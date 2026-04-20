#!/bin/bash
# TurboQuant patch + vLLM 启动脚本
# 用法：在 Docker 容器内运行此脚本

set -e

VLLM_PATH=/opt/torchenv/lib/python3.12/site-packages/vllm
SRC=/workspace/vllm
WHEELS_DIR=/workspace/wheels

# ── 步骤 0：安装 triton-gfx906 wheel（若有）────────────────────────────────
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

echo "[patch] Applying TurboQuant patches to vLLM ${VLLM_PATH}..."

FILES=(
  config/cache.py
  attention/layer.py
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

echo "[patch] Done. Verifying imports..."
python3 -c "
from vllm.config.cache import CacheDType
from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionBackend
print('[patch] turboquant_attn imported OK')
print('[patch] CacheDType turboquant values:', [v for v in CacheDType.__args__ if 'turboquant' in v])
" 2>&1

echo "[patch] Starting vLLM server..."
exec python3 -m vllm.entrypoints.openai.api_server "$@"
