"""
TurboQuant decode 精度分析 - compact 版本
对比服务 output 与离线 kernel 结果
"""
import torch, math, sys, pickle, os
import numpy as np
sys.path.insert(0, '/opt/torchenv/lib/python3.12/site-packages')

from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
from vllm.v1.attention.ops.triton_turboquant_decode import triton_turboquant_decode_attention
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard

SAVE_FILE = '/tmp/tq_compact.pkl'

if not os.path.exists(SAVE_FILE):
    print(f"No file at {SAVE_FILE}")
    sys.exit(1)

data = pickle.load(open(SAVE_FILE, 'rb'))
blk_idx = data['blk_idx']
seq_len = data['seq_lens'][0]

# Reconstruct kv_cache: full zeros + relevant block
kv_full_shape = data['kv_full_shape']
kv_cache = torch.zeros(kv_full_shape, dtype=torch.uint8, device='cuda')
kv_block = np.frombuffer(data['kv_block'], dtype=np.uint8).reshape(data['kv_block_shape'])
kv_cache[blk_idx] = torch.from_numpy(kv_block.copy()).cuda()

query = torch.from_numpy(np.frombuffer(data['query'], dtype=np.float16)
                         .reshape(data['query_shape']).copy()).cuda()
bt = torch.from_numpy(np.frombuffer(data['block_table'], dtype=np.int32)
                      .reshape(data['bt_shape']).copy()).cuda()
seq_lens = torch.tensor(data['seq_lens'], dtype=torch.int32, device='cuda')

Pi = torch.from_numpy(np.frombuffer(data['Pi'], dtype=np.float32)
                      .reshape(data['Pi_shape']).copy()).cuda()
c = torch.from_numpy(np.frombuffer(data['centroids'], dtype=np.float32)
                     .reshape(data['centroids_shape']).copy()).cuda()

B, Hq, D = query.shape
Hk = kv_cache.shape[2]

print(f"kv_cache: {kv_cache.shape}, query: {query.shape}")
print(f"blk={blk_idx}, seq_len={seq_len}, scale={data['scale']:.4f}")
print(f"kv[{blk_idx},0,0,:6]: {kv_cache[blk_idx,0,0,:6].tolist()}")
print(f"query[0,0,:3]: {query[0,0,:3].tolist()}")

# ==== Service output ====
svc_out = torch.from_numpy(
    np.frombuffer(data['tq_output'], dtype=np.float16)
    .reshape(data['tq_output_shape']).copy()
).cuda()
print(f"\nService TQ output:")
print(f"  [0,0,:5]: {svc_out[0,0,:5].tolist()}")
print(f"  std={data['tq_out_std']:.4f}, mean_abs={data['tq_out_mean_abs']:.4f}")

# ==== Offline TQ kernel ====
print("\nRunning offline TQ kernel...")
result_offline = triton_turboquant_decode_attention(
    query=query,
    kv_cache=kv_cache.contiguous(),
    block_table=bt,
    seq_lens=seq_lens,
    Pi=Pi, centroids=c,
    scale=data['scale'],
    mse_bits=data['mse_bits'],
    key_packed_size=data['key_packed_size'],
    value_quant_bits=data['value_quant_bits'],
    key_fp8=data['key_fp8'],
    norm_correction=data['norm_correction'],
    PiT=Pi,
)
print(f"Offline TQ output:")
print(f"  [0,0,:5]: {result_offline[0,0,:5].tolist()}")
print(f"  std={result_offline.float().std():.4f}, mean_abs={result_offline.float().abs().mean():.4f}")

# ==== Compare ====
mae_svc_vs_offline = (svc_out - result_offline).abs().mean().item()
cos_sim = torch.nn.functional.cosine_similarity(
    svc_out[0,0].float().unsqueeze(0),
    result_offline[0,0].float().unsqueeze(0)
).item()
print(f"\n=== Service vs Offline ===")
print(f"MAE: {mae_svc_vs_offline:.6f}")
print(f"Cosine sim: {cos_sim:.4f}")
if mae_svc_vs_offline < 0.001:
    print("IDENTICAL: service and offline produce same TQ output")
    print("→ Bug is AFTER the TQ decode kernel (e.g. output shape/dtype issue)")
else:
    print("DIFFERENT: service and offline TQ kernels diverge")
    print("→ Bug is IN the TQ decode kernel execution")

# ==== dequant reference ====
print("\nRunning dequant reference...")
import triton
from vllm.v1.attention.ops.triton_turboquant_decode import _tq_full_dequant_kv

alloc_len = math.ceil(seq_len / kv_cache.shape[1]) * kv_cache.shape[1]
BLOCK_D = triton.next_power_of_2(D)
mse_bytes = math.ceil(D * data['mse_bits'] / 8)
val_data_bytes = math.ceil(D * data['value_quant_bits'] / 8)
k_dq = torch.zeros(1, Hk, alloc_len, D, dtype=torch.float16, device='cuda')
v_dq = torch.zeros(1, Hk, alloc_len, D, dtype=torch.float16, device='cuda')
grid = (alloc_len, Hk)
_tq_full_dequant_kv[grid](
    kv_cache.contiguous(), bt, c,
    k_dq, v_dq,
    k_dq.stride(0), k_dq.stride(1), k_dq.stride(2),
    v_dq.stride(0), v_dq.stride(1), v_dq.stride(2),
    kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
    bt.stride(0),
    HEAD_DIM=D, BLOCK_SIZE=kv_cache.shape[1], NUM_KV_HEADS=Hk,
    MSE_BYTES=mse_bytes, KPS=data['key_packed_size'],
    VQB=data['value_quant_bits'], VAL_DATA_BYTES=val_data_bytes,
    MSE_BITS=data['mse_bits'], KEY_FP8=1 if data['key_fp8'] else 0,
    BLOCK_D=BLOCK_D, NORM_CORRECTION=1 if data['norm_correction'] else 0,
    FP8_E4B15=0, num_warps=4,
)
# Inverse WHT rotation
k_flat = k_dq[0, :, :seq_len, :].reshape(-1, D).float() @ Pi
k_rot_back = k_flat.reshape(Hk, seq_len, D).half()
v_trim = v_dq[0, :, :seq_len, :]

kv_group = Hq // Hk
ref_list = []
q_r = query[0].float()
k_r = k_rot_back.permute(1,0,2).float()
v_r = v_trim.permute(1,0,2).float()
for h in range(Hq):
    kv_h = h // kv_group
    s = torch.einsum('d,td->t', q_r[h], k_r[:,kv_h]) * data['scale']
    a = torch.softmax(s, dim=0)
    ref_list.append(torch.einsum('t,td->d', a, v_r[:,kv_h]))
ref = torch.stack(ref_list, dim=0).half().unsqueeze(0)

print(f"Dequant ref: [0,0,:5]={ref[0,0,:5].tolist()}")
mae_svc_vs_ref = (svc_out - ref).abs().mean().item()
mae_offline_vs_ref = (result_offline - ref).abs().mean().item()
print(f"MAE(service vs ref): {mae_svc_vs_ref:.4f}")
print(f"MAE(offline vs ref): {mae_offline_vs_ref:.4f}")

print(f"\n=== CONCLUSION ===")
if mae_offline_vs_ref < 0.01:
    print("Offline TQ kernel is CORRECT (matches dequant ref)")
    if mae_svc_vs_offline < 0.01:
        print("Service TQ kernel output MATCHES offline → Problem is elsewhere in integration")
        print("Likely candidates: attention output shape/dtype, hidden states accumulation")
    else:
        print("Service TQ kernel output DIFFERS from offline → Bug in service kernel execution")
else:
    print("Offline TQ kernel has issues too → True kernel bug")
