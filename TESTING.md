# 测试说明 - Transformers 5 兼容性修复

## 修复内容

已修复 transformers 5.0 兼容性问题，现在支持：
- ✅ Transformers 4.56.0 - 4.x
- ✅ Transformers 5.0.0+

## 快速测试

### 1. 在 Linux 环境中测试

```bash
# 激活环境
conda activate vllm_test

# 升级 transformers（如果需要）
pip install "transformers>=4.56.0,<5"

# 或者安装 transformers 5（测试兼容性）
pip install "transformers>=5.0.0,<6"
```

### 2. 测试导入

```bash
python -c "from vllm import LLM; print('✅ Import successful!')"
```

### 3. 测试模型加载

```bash
# 使用您的原始命令
VLLM_USE_MODELSCOPE=true vllm serve Qwen/Qwen3.5-0.8B --port 8000 --tensor-parallel-size 1 --max-model-len 8192
```

### 4. 预期结果

- ✅ 不再出现 `ImportError: cannot import name 'ALLOWED_LAYER_TYPES'`
- ✅ 模型正常加载
- ✅ 服务正常启动

## 详细测试

### 测试 Transformers 4.x

```bash
# 安装 transformers 4.x
pip install "transformers==4.56.2"

# 测试
python -c "
from vllm.transformers_utils.config import ALLOWED_ATTENTION_LAYER_TYPES
print(f'✅ Using ALLOWED_ATTENTION_LAYER_TYPES from: {ALLOWED_ATTENTION_LAYER_TYPES.__name__}')
"
```

### 测试 Transformers 5.x

```bash
# 安装 transformers 5.x
pip install "transformers==5.0.0"

# 测试
python -c "
from vllm.transformers_utils.config import ALLOWED_ATTENTION_LAYER_TYPES
print(f'✅ Using ALLOWED_ATTENTION_LAYER_TYPES from: {ALLOWED_ATTENTION_LAYER_TYPES.__name__}')
"
```

## 验证修复

查看源码确认修复已应用：

```bash
# 检查 transformers_utils/config.py
grep -A 8 "Transformers v5" vllm/transformers_utils/config.py

# 应该看到：
# try:
#     # Transformers v5
#     from transformers.configuration_utils import ALLOWED_ATTENTION_LAYER_TYPES
# except ImportError:
#     # Transformers v4
#     from transformers.configuration_utils import (
#         ALLOWED_LAYER_TYPES as ALLOWED_ATTENTION_LAYER_TYPES,
#     )
```

## 故障排除

### 问题 1: 仍然出现 ImportError

**原因**: 可能使用了旧版本的 vllm-gfx906

**解决**:
```bash
cd ~/vllm-gfx906
git pull origin gfx906/main
pip install -e .
```

### 问题 2: 其他依赖冲突

**原因**: requirements 中的其他包版本不兼容

**解决**:
```bash
# 检查 requirements/rocm.txt
cat requirements/rocm.txt

# 逐个安装调试
pip install transformers==5.0.0
pip install torch==2.9.0
# ... 其他依赖
```

### 问题 3: 模型加载失败

**原因**: 可能是模型文件或配置问题

**解决**:
```bash
# 检查模型是否可访问
python -c "
from modelscope import snapshot_download
model_path = snapshot_download('Qwen/Qwen3.5-0.8B')
print(f'Model path: {model_path}')
"
```

## 性能测试

### 基础推理测试

```python
from vllm import LLM

# 初始化
llm = LLM(
    model="Qwen/Qwen3.5-0.8B",
    max_model_len=8192,
    tensor_parallel_size=1,
)

# 推理
outputs = llm.generate(["Hello, my name is", "The future of AI is"])

# 打印结果
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print("-" * 50)
```

### 批量测试

```bash
# 使用 vLLM benchmark
python -m vllm.benchmarks.benchmark_serving \
    --model Qwen/Qwen3.5-0.8B \
    --dataset-name sharegpt \
    --dataset-path ./data/sharegpt.json \
    --num-prompts 10
```

## 版本信息

- **vllm-gfx906**: 基于 v0.6.x
- **Transformers**: 4.56.0 - 5.x
- **Python**: 3.10+
- **ROCm**: 6.3+

## 反馈

如果测试中遇到问题，请提供：
1. 完整的错误信息
2. transformers 版本 (`pip show transformers`)
3. Python 版本 (`python --version`)
4. ROCm 版本 (`rocminfo | grep -i version`)

---

**更新日期**: 2026-03-08
**维护者**: Sisyphus (AI Agent)
**修复提交**: cc7a762cc, 0bca9aa91
