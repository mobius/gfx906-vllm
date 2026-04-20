"""
从服务里捕获真实的 kv_cache 数据并在独立测试里重现 decode
"""
import torch, math, sys, pickle, os, numpy as np
sys.path.insert(0, '/opt/torchenv/lib/python3.12/site-packages')

from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
from vllm.v1.attention.ops.triton_turboquant_decode import triton_turboquant_decode_attention
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard

SAVE_FILE = '/tmp/tq_capture.pkl'

D = 128; Hk = 2; Hq = 12; block_size = 16
cfg = TurboQuantConfig.from_cache_dtype('turboquant_4bit_nc', D)
c = get_centroids(D, cfg.centroid_bits).cuda().float()
Pi = _build_hadamard(D, 'cuda')
PiT = Pi
scale = 1.0 / math.sqrt(D)

if os.path.exists(SAVE_FILE):
    print(f"Loading captured data from {SAVE_FILE}")
    data = pickle.load(open(SAVE_FILE, 'rb'))
    
    # Reconstruct tensors from numpy arrays
    kv_bytes = np.frombuffer(data['kv_bytes'], dtype=np.uint8)
    kv_data = torch.from_numpy(kv_bytes.reshape(data['kv_shape']).copy()).cuda()
    
    query_bytes = np.frombuffer(data['query'], dtype=np.float16)
    query_data = torch.from_numpy(query_bytes.reshape(data['query_shape']).copy()).cuda()
    
    bt_bytes = np.frombuffer(data['block_table'], dtype=np.int32)
    bt = torch.from_numpy(bt_bytes.reshape(data['bt_shape']).copy()).cuda()
    
    sl = torch.tensor(data['seq_lens'], dtype=torch.int32, device='cuda')
    blk_idx = data.get('blk_idx', bt[0, 0].item())
    
    print(f"kv_cache shape: {kv_data.shape}")
    print(f"query shape: {query_data.shape}")
    print(f"block_table: {bt[0, :4].tolist()}")
    print(f"seq_lens: {sl.tolist()}")
    print(f"kv_cache[{blk_idx}, 0, 0, :6]: {kv_data[blk_idx, 0, 0, :6].tolist()}")
    print(f"kv_cache is_contiguous: {kv_data.is_contiguous()}")
    
    out = triton_turboquant_decode_attention(
        query=query_data, kv_cache=kv_data.contiguous(), block_table=bt,
        seq_lens=sl, Pi=Pi, centroids=c, scale=scale,
        mse_bits=cfg.mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.value_quant_bits, key_fp8=cfg.key_fp8,
        norm_correction=cfg.norm_correction, PiT=PiT,
    )
    print(f"TQ decode output[0, 0, :5]: {out[0, 0, :5].tolist()}")
    print(f"TQ decode output[0, 1, :5]: {out[0, 1, :5].tolist()}")
    has_nan = out.isnan().any().item()
    has_inf = out.isinf().any().item()
    print(f"NaN: {has_nan}, Inf: {has_inf}, std: {out.float().std().item():.4f}, mean: {out.float().abs().mean().item():.4f}")
    
    # Check if output values are reasonable
    if has_nan or has_inf:
        print("FAIL: NaN/Inf detected")
    elif out.float().abs().mean().item() < 0.001:
        print("FAIL: Output near zero")
    elif out.float().std().item() > 10:
        print("WARN: Output std very large, possible issue")
    else:
        print("PASS: Output looks reasonable")
else:
    print(f"No capture file found at {SAVE_FILE}")
