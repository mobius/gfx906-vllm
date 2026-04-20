"""Test TurboQuant decode at different seq_len values"""
import torch, math, sys
sys.path.insert(0, '/opt/torchenv/lib/python3.12/site-packages')
from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
from vllm.v1.attention.ops.triton_turboquant_decode import triton_turboquant_decode_attention
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard

D, Hk, Hq = 128, 2, 2
block_size = 16
cfg = TurboQuantConfig.from_cache_dtype('turboquant_4bit_nc', D)
c = get_centroids(D, cfg.centroid_bits).cuda().float()
c_sorted, _ = c.sort()
midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
Pi = _build_hadamard(D, 'cuda')
PiT = Pi
kv_cache = torch.zeros(64, block_size, Hk, cfg.slot_size_aligned, dtype=torch.uint8, device='cuda')
torch.manual_seed(42)
scale = 1.0 / math.sqrt(D)

for seq_len in [1, 5, 6, 10, 20, 32, 33, 50]:
    kv_cache.zero_()
    keys = torch.randn(seq_len, Hk, D, device='cuda', dtype=torch.float16)
    vals = torch.randn(seq_len, Hk, D, device='cuda', dtype=torch.float16)
    slots = torch.arange(seq_len, dtype=torch.int64, device='cuda')
    triton_turboquant_store(key=keys, value=vals, kv_cache=kv_cache,
        slot_mapping=slots, PiT=PiT, midpoints=midpoints,
        mse_bits=cfg.mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.value_quant_bits, key_fp8=cfg.key_fp8)
    
    query = torch.randn(1, Hq, D, device='cuda', dtype=torch.float16)
    bt = torch.zeros(1, 64, dtype=torch.int32, device='cuda')
    sl = torch.tensor([seq_len], dtype=torch.int32, device='cuda')
    
    out = triton_turboquant_decode_attention(
        query=query, kv_cache=kv_cache, block_table=bt, seq_lens=sl,
        Pi=Pi, centroids=c, scale=scale, mse_bits=cfg.mse_bits,
        key_packed_size=cfg.key_packed_size, value_quant_bits=cfg.value_quant_bits,
        key_fp8=cfg.key_fp8, norm_correction=cfg.norm_correction, PiT=PiT,
    )
    
    scores = torch.einsum('bhd,thd->bht', query.float(), keys.float()) * scale
    attn = torch.softmax(scores, dim=-1)
    ref = torch.einsum('bht,thd->bhd', attn, vals.float()).half()
    
    mae = (out - ref).abs().mean().item()
    status = "PASS" if mae < 0.15 else "FAIL"
    print(f'seq_len={seq_len:3d}: MAE={mae:.4f} {status}')
