"""
TurboQuant decode 精度离线分析脚本
从服务捕获的真实数据重现 decode，对比 kernel 输出与参考值
"""
import torch, math, sys, pickle, os
import numpy as np
sys.path.insert(0, '/opt/torchenv/lib/python3.12/site-packages')

from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
from vllm.v1.attention.ops.triton_turboquant_decode import triton_turboquant_decode_attention
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard

SAVE_FILE = '/tmp/tq_capture.pkl'


def load_tensor(data, key_bytes, key_shape, dtype=np.float16):
    arr = np.frombuffer(data[key_bytes], dtype=dtype)
    return torch.from_numpy(arr.reshape(data[key_shape]).copy()).cuda()


def load_tensor_u8(data, key_bytes, key_shape):
    arr = np.frombuffer(data[key_bytes], dtype=np.uint8)
    return torch.from_numpy(arr.reshape(data[key_shape]).copy()).cuda()


if not os.path.exists(SAVE_FILE):
    print(f"No capture file at {SAVE_FILE}")
    sys.exit(1)

print(f"Loading from {SAVE_FILE} ({os.path.getsize(SAVE_FILE)/1e6:.1f}MB)...")
data = pickle.load(open(SAVE_FILE, 'rb'))

# Reconstruct tensors
kv_cache = load_tensor_u8(data, 'kv_bytes', 'kv_shape')
query = load_tensor(data, 'query', 'query_shape', dtype=np.float16)
bt_arr = np.frombuffer(data['block_table'], dtype=np.int32)
block_table = torch.from_numpy(bt_arr.reshape(data['bt_shape']).copy()).cuda()
seq_lens = torch.tensor(data['seq_lens'], dtype=torch.int32, device='cuda')
blk_idx = data['blk_idx']

# TQ config
mse_bits = data['mse_bits']
key_packed_size = data['key_packed_size']
value_quant_bits = data['value_quant_bits']
norm_correction = data['norm_correction']
key_fp8 = data['key_fp8']
scale = data['scale']

# Pi and centroids
Pi_arr = np.frombuffer(data['Pi'], dtype=np.float32)
Pi = torch.from_numpy(Pi_arr.reshape(data['Pi_shape']).copy()).cuda()
PiT = Pi

c_arr = np.frombuffer(data['centroids'], dtype=np.float32)
centroids = torch.from_numpy(c_arr.reshape(data['centroids_shape']).copy()).cuda()

B, Hq, D = query.shape
Hk = kv_cache.shape[2]
seq_len = seq_lens[0].item()

print(f"kv_cache: {kv_cache.shape}, query: {query.shape}")
print(f"block_table[0,:4]: {block_table[0,:4].tolist()}, seq_len: {seq_len}")
print(f"kv[{blk_idx},0,0,:6]: {kv_cache[blk_idx,0,0,:6].tolist()}")
print(f"query[0,0,:3]: {query[0,0,:3].tolist()}")
print(f"TQ config: mse_bits={mse_bits}, value_quant_bits={value_quant_bits}, "
      f"norm_correction={norm_correction}, scale={scale:.4f}")

# ==============================================================================
# Test 1: TQ decode kernel (same as service)
# ==============================================================================
print("\n=== Test 1: TQ decode kernel ===")
result_tq = triton_turboquant_decode_attention(
    query=query,
    kv_cache=kv_cache.contiguous(),
    block_table=block_table,
    seq_lens=seq_lens,
    Pi=Pi, centroids=centroids, scale=scale,
    mse_bits=mse_bits, key_packed_size=key_packed_size,
    value_quant_bits=value_quant_bits, key_fp8=key_fp8,
    norm_correction=norm_correction, PiT=PiT,
)
print(f"TQ output[0,0,:5]: {result_tq[0,0,:5].tolist()}")
print(f"std={result_tq.float().std():.4f}, mean_abs={result_tq.float().abs().mean():.4f}")
print(f"NaN:{result_tq.isnan().any().item()}, Inf:{result_tq.isinf().any().item()}")

# ==============================================================================
# Test 2: dequant + standard attention (reference)
# ==============================================================================
print("\n=== Test 2: dequant + standard attention ===")
from vllm.v1.attention.ops.triton_turboquant_decode import _tq_full_dequant_kv
import triton

alloc_len = math.ceil(seq_len / kv_cache.shape[1]) * kv_cache.shape[1]
BLOCK_D = triton.next_power_of_2(D)
mse_bytes = math.ceil(D * mse_bits / 8)
val_data_bytes = math.ceil(D * value_quant_bits / 8)

k_dq = torch.zeros(1, Hk, alloc_len, D, dtype=torch.float16, device='cuda')
v_dq = torch.zeros(1, Hk, alloc_len, D, dtype=torch.float16, device='cuda')

grid = (alloc_len, Hk)
_tq_full_dequant_kv[grid](
    kv_cache.contiguous(), block_table, centroids,
    k_dq, v_dq,
    k_dq.stride(0), k_dq.stride(1), k_dq.stride(2),
    v_dq.stride(0), v_dq.stride(1), v_dq.stride(2),
    kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
    block_table.stride(0),
    HEAD_DIM=D, BLOCK_SIZE=kv_cache.shape[1], NUM_KV_HEADS=Hk,
    MSE_BYTES=mse_bytes, KPS=key_packed_size,
    VQB=value_quant_bits, VAL_DATA_BYTES=val_data_bytes,
    MSE_BITS=mse_bits, KEY_FP8=1 if key_fp8 else 0,
    BLOCK_D=BLOCK_D, NORM_CORRECTION=1 if norm_correction else 0,
    FP8_E4B15=0, num_warps=4,
)

# Inverse-rotate keys back to RoPE space
k_trimmed = k_dq[0, :, :seq_len, :]  # (Hk, seq_len, D)
k_flat = k_trimmed.reshape(-1, D).float()
k_flat = k_flat @ Pi  # inverse WHT rotation
k_rot_back = k_flat.reshape(Hk, seq_len, D).half()  # (Hk, seq_len, D)
v_trimmed = v_dq[0, :, :seq_len, :]  # (Hk, seq_len, D)

# Standard attention: each query head attends to its KV group
kv_group = Hq // Hk
ref_list = []
q_r = query[0].float()  # (Hq, D)
k_r = k_rot_back.permute(1, 0, 2).float()  # (seq_len, Hk, D)
v_r = v_trimmed.permute(1, 0, 2).float()  # (seq_len, Hk, D)
for h in range(Hq):
    kv_h = h // kv_group
    s = torch.einsum('d,td->t', q_r[h], k_r[:, kv_h]) * scale
    a = torch.softmax(s, dim=0)
    ref_list.append(torch.einsum('t,td->d', a, v_r[:, kv_h]))
ref = torch.stack(ref_list, dim=0).half().unsqueeze(0)

print(f"Ref output[0,0,:5]: {ref[0,0,:5].tolist()}")
print(f"std={ref.float().std():.4f}, mean_abs={ref.float().abs().mean():.4f}")

# ==============================================================================
# Compare TQ vs Reference
# ==============================================================================
print("\n=== Comparison ===")
mae = (result_tq - ref).abs().mean().item()
cos_sim = torch.nn.functional.cosine_similarity(
    result_tq[0,0].float().unsqueeze(0),
    ref[0,0].float().unsqueeze(0)
).item()
print(f"MAE(TQ vs dequant_ref): {mae:.4f}")
print(f"Cosine similarity: {cos_sim:.4f}")
if mae < 0.15:
    print("PASS: TQ matches dequant reference")
else:
    print("FAIL: TQ output differs significantly from dequant reference")

# ==============================================================================
# Test 3: Compare kv_cache block content
# ==============================================================================
print("\n=== kv_cache block content ===")
print(f"Block {blk_idx} non-zero bytes: {(kv_cache[blk_idx] != 0).sum().item()}")
print(f"kv[{blk_idx},0,0,:8]: {kv_cache[blk_idx,0,0,:8].tolist()}")
print(f"kv[{blk_idx},1,0,:8]: {kv_cache[blk_idx,1,0,:8].tolist()}")
print(f"Expected non-zero for {min(seq_len, kv_cache.shape[1])} tokens × {Hk} heads × 134 bytes = "
      f"{min(seq_len, kv_cache.shape[1]) * Hk * 134}")

# ==============================================================================
# Test 4: If service also saved output, compare
# ==============================================================================
if 'tq_output' in data:
    print("\n=== Comparing with service output ===")
    svc_out_arr = np.frombuffer(data['tq_output'], dtype=np.float16)
    svc_out = torch.from_numpy(svc_out_arr.reshape(data['tq_output_shape']).copy()).cuda()
    mae_vs_svc = (result_tq - svc_out).abs().mean().item()
    print(f"Service output[0,0,:5]: {svc_out[0,0,:5].tolist()}")
    print(f"MAE(offline_TQ vs service_TQ): {mae_vs_svc:.6f}")
    if mae_vs_svc < 0.001:
        print("IDENTICAL: offline and service kernels produce same result")
    else:
        print("DIFFERENT: offline and service outputs diverge!")
else:
    print("\nNo service output in capture (run with TQ_CAPTURE=1)")
