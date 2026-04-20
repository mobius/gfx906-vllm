"""
TurboQuant 数值正确性验证脚本 v2
"""
import torch, sys, math

def test_roundtrip():
    print("=== TurboQuant Round-trip Test (4bit_nc) ===")
    
    Hk = 2; Hq = 2; D = 128
    block_size = 16; num_blocks = 32
    
    sys.path.insert(0, '/opt/torchenv/lib/python3.12/site-packages')
    from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
    from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
    from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
    from vllm.v1.attention.ops.triton_turboquant_decode import triton_turboquant_decode_attention
    from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
    
    cfg = TurboQuantConfig.from_cache_dtype('turboquant_4bit_nc', D)
    centroids = get_centroids(D, cfg.centroid_bits).cuda().float()
    c_sorted, _ = centroids.sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
    
    Pi = _build_hadamard(D, 'cuda')
    PiT = Pi
    
    slot_size = cfg.slot_size_aligned
    kv_cache = torch.zeros(num_blocks, block_size, Hk, slot_size, dtype=torch.uint8, device='cuda')
    
    # seq_len=4 tokens: prefill 4 tokens then decode 1 step
    seq_len = 4
    torch.manual_seed(42)
    key = torch.randn(seq_len, Hk, D, device='cuda', dtype=torch.float16)
    value = torch.randn(seq_len, Hk, D, device='cuda', dtype=torch.float16)
    slot_mapping = torch.arange(seq_len, dtype=torch.int32, device='cuda')
    
    # Store all prefill tokens
    triton_turboquant_store(
        key=key, value=value, kv_cache=kv_cache, slot_mapping=slot_mapping,
        PiT=PiT, midpoints=midpoints,
        mse_bits=cfg.mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.value_quant_bits, key_fp8=cfg.key_fp8,
    )
    print(f"Stored {seq_len} tokens")
    
    # Decode: query attends over stored 4 tokens
    query = torch.randn(1, Hq, D, device='cuda', dtype=torch.float16)
    block_table = torch.zeros(1, num_blocks, dtype=torch.int32, device='cuda')
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device='cuda')
    scale = 1.0 / math.sqrt(D)
    
    output = triton_turboquant_decode_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        Pi=Pi,
        centroids=centroids,
        scale=scale,
        mse_bits=cfg.mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.value_quant_bits,
        key_fp8=cfg.key_fp8,
        norm_correction=cfg.norm_correction,
        PiT=PiT,
    )
    print(f"Decode output shape: {output.shape}")
    print(f"Output[:5]: {output[0,0,:5].tolist()}")
    
    has_nan = output.isnan().any().item()
    has_inf = output.isinf().any().item()
    out_std = output.float().std().item()
    print(f"NaN: {has_nan}, Inf: {has_inf}, Std: {out_std:.4f}")
    
    # 验证输出不是乱码：std 应该在合理范围
    if has_nan or has_inf:
        print("FAIL: NaN/Inf in output")
        return False
    if out_std < 0.001 or out_std > 100:
        print(f"FAIL: Output std out of range ({out_std:.4f})")
        return False
    
    # 参考：用标准 attention 计算结果对比
    # key/value: [seq_len, Hk, D] → attention output
    q_r = query.float()  # [1, Hq, D]
    k_r = key.float()    # [4, Hk, D]
    v_r = value.float()  # [4, Hk, D]
    # GQA: Hq==Hk here
    scores = torch.einsum('bhd,thd->bht', q_r, k_r) * scale  # [1, Hq, 4]
    attn = torch.softmax(scores, dim=-1)
    ref_out = torch.einsum('bht,thd->bhd', attn, v_r).half()
    print(f"Reference output[:5]: {ref_out[0,0,:5].tolist()}")
    
    mae = (output - ref_out).abs().mean().item()
    print(f"MAE vs reference: {mae:.4f}")
    
    if mae < 0.3:
        print("PASS: Output matches reference within tolerance")
        return True
    else:
        print(f"FAIL: MAE too large ({mae:.4f})")
        return False

if __name__ == '__main__':
    try:
        ok = test_roundtrip()
        sys.exit(0 if ok else 1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
