#!/bin/bash
# TurboQuant + Gemma4 patch + vLLM 启动脚本
# 用法：在 Docker 容器内运行此脚本
# 支持模型：Qwen2.5（AWQ）、Gemma4（AWQ）等

set -e

VLLM_PATH=/opt/torchenv/lib/python3.12/site-packages/vllm
SRC=/workspace/vllm
WHEELS_DIR=/workspace/wheels

# ── 步骤 0a：升级 transformers（Gemma4 需要 5.x）──────────────────────────────
if python3 -c "import transformers; assert tuple(int(x) for x in transformers.__version__.split('.')[:2]) >= (5, 0)" 2>/dev/null; then
  echo "[transformers] version OK: $(python3 -c 'import transformers; print(transformers.__version__)')"
else
  echo "[transformers] upgrading to >=5.0 for Gemma4 support..."
  pip install 'transformers>=5.5.0' -q 2>&1 | grep -E 'Successfully|ERROR' | head -2
fi

# ── 步骤 0b：安装 triton-gfx906 wheel（若有）────────────────────────────────
TQ_WHEEL=$(ls "${WHEELS_DIR}"/triton-*.whl 2>/dev/null | head -1)
if [ -n "${TQ_WHEEL}" ]; then
  echo "[triton] Installing triton-gfx906 from ${TQ_WHEEL}..."
  pip install "${TQ_WHEEL}" --force-reinstall 2>&1 | tail -5
  python3 -c "
import triton, os
amd_dir = os.path.join(os.path.dirname(triton.__file__), 'backends', 'amd')
print('[triton] version:', triton.__version__, '| AMD backend:', os.path.exists(amd_dir))
" 2>&1 || true
else
  echo "[triton] No wheel found in ${WHEELS_DIR}, using existing triton"
fi

echo "[patch] Applying gfx906 + TurboQuant + Gemma4 patches to vLLM ${VLLM_PATH}..."

FILES=(
  # ── gfx906 / TurboQuant core ─────────────────────────────────────────────
  config/cache.py
  attention/layer.py
  model_executor/layers/quantization/moe_wna16.py
  attention/ops/triton_reshape_and_cache_flash.py
  model_executor/layers/quantization/turboquant/__init__.py
  model_executor/layers/quantization/turboquant/centroids.py
  model_executor/layers/quantization/turboquant/config.py
  model_executor/layers/quantization/turboquant/quantizer.py
  model_executor/models/qwen2.py
  platforms/rocm.py
  utils/torch_utils.py
  v1/attention/backends/registry.py
  v1/attention/backends/turboquant_attn.py
  v1/attention/ops/__init__.py
  v1/attention/ops/triton_turboquant_decode.py
  v1/attention/ops/triton_turboquant_store.py
  v1/kv_cache_interface.py
  v1/core/single_type_kv_cache_manager.py
  # ── Gemma4 support ───────────────────────────────────────────────────────
  model_executor/models/gemma4.py
  model_executor/models/gemma4_mm.py
  model_executor/models/registry.py
  model_executor/layers/rotary_embedding/__init__.py
  model_executor/layers/rotary_embedding/gemma4_rope.py
  reasoning/gemma4_reasoning_parser.py
  reasoning/gemma4_utils.py
  model_executor/layers/attention/__init__.py
  model_executor/layers/fused_moe/__init__.py
  model_executor/layers/fused_moe/activation.py
  model_executor/layers/fused_moe/router/__init__.py
  model_executor/layers/fused_moe/router/gate_linear.py
  model_executor/layers/fused_moe/router/base_router.py
  model_executor/layers/fused_moe/router/fused_moe_router.py
  model_executor/layers/fused_moe/router/fused_topk_router.py
  model_executor/layers/fused_moe/router/fused_topk_bias_router.py
  model_executor/layers/fused_moe/router/grouped_topk_router.py
  model_executor/layers/fused_moe/router/router_factory.py
  model_executor/layers/fused_moe/router/custom_routing_router.py
  model_executor/layers/fused_moe/router/routing_simulator_router.py
  model_executor/layers/fused_moe/router/zero_expert_router.py
  model_executor/custom_op.py
  model_executor/models/interfaces.py
)

for f in "${FILES[@]}"; do
  dir=$(dirname "${VLLM_PATH}/${f}")
  mkdir -p "${dir}"
  cp "${SRC}/${f}" "${VLLM_PATH}/${f}"
done

# 清理 __pycache__ 避免旧字节码缓存干扰
find "${VLLM_PATH}" -name "*.pyc" -path "*turboquant*" -delete 2>/dev/null || true
find "${VLLM_PATH}" -name "*.pyc" -path "*cache*" -delete 2>/dev/null || true
find "${VLLM_PATH}" -name "*.pyc" -path "*registry*" -delete 2>/dev/null || true

# ── 可选：让 MoE expert 层走 fp16（绕过 MoeWNA16 量化路径）─────────────────
# 设置 VLLM_GEMMA4_FP16_MOE=1 来启用
if [ "${VLLM_GEMMA4_FP16_MOE:-0}" = "1" ]; then
  echo "[patch] Applying fp16 MoE patch to awq.py (MoE experts will run in fp16)..."
  python3 -c "
import re
path = '${VLLM_PATH}/model_executor/layers/quantization/awq.py'
with open(path) as f:
    content = f.read()
old = '''        elif isinstance(layer, FusedMoE):
            # Lazy import to avoid circular import.
            from .moe_wna16 import MoeWNA16Config
            config = {
                \"quant_method\": \"awq\",
                \"bits\": self.weight_bits,
                \"group_size\": self.group_size,
                \"zero_point\": self.zero_point,
                \"lm_head\": False,
                \"modules_to_not_convert\": self.modules_to_not_convert,
            }
            logger.warning_once(
                \"[vllm-gfx906] You are using modified MoeWNA16 kernel, \"
                \"this is differ from the offical vLLM.\")
            return MoeWNA16Config.from_config(config).get_quant_method(
                layer, prefix)'''
new = '''        elif isinstance(layer, FusedMoE):
            # gfx906-vllm fp16-moe mode: MoE expert layers run in fp16
            return None'''
if old in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('[patch] awq.py: MoE experts set to fp16')
else:
    print('[patch] WARNING: awq.py fp16-moe pattern not found (already patched?)')
" 2>&1
fi

echo "[patch] Done. Verifying imports..."
python3 -c "
from vllm.config.cache import CacheDType
from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionBackend
print('[patch] turboquant_attn imported OK')
print('[patch] CacheDType turboquant values:', [v for v in CacheDType.__args__ if 'turboquant' in v])
" 2>&1

echo "[patch] Starting vLLM server..."
exec python3 -m vllm.entrypoints.openai.api_server "$@"
