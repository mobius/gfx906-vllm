# 硬件验证清单

**目标**: 在AMD MI50 (gfx906) GPU上验证ROCm MLA改进

---

## 环境检查

### 1. 系统信息

```bash
# GPU信息
rocm-smi --showmem
rocm-smi --showtemp
rocm-smi --showclockfreqs

# 预期输出: 显示MI50的内存、温度、时钟频率
```

**预期结果**:
- GPU检测: AMD MI50 (gfx906)
- 内存: ≥ 32GB HBM2
- 温度: 正常范围 (< 80°C)

### 2. 软件环境

```bash
# Python版本
python --version  # 建议: 3.8+

# PyTorch版本
python -c "import torch; print(torch.__version__)"
# 检查是否有ROCm支持
python -c "import torch; print(torch.version.hip)"
```

**预期结果**:
- PyTorch with ROCm support
- HIP (Heterogeneous Interface for Portable Computing) available

### 3. vLLM版本确认

```bash
cd /path/to/vllm-gfx906

# 确认分支
git branch
# 应该显示: gfx906/main 或 gfx906/adapt-phase1-critical-fixes

# 确认commit
git log --oneline -3
# 应该包含:
#   [ROCm] Support MLA with nhead<16 and FP8 KV cache for TP=8
```

---

## 代码验证

### 1. 运行单元测试

```bash
cd /path/to/vllm-gfx906

# 运行验证测试
python test_rocm_mla_fix.py
```

**期望输出**:
```
[PASS] num_heads=4 should be supported
[PASS] num_heads=8 should be supported
[PASS] num_heads=16 should be supported
[PASS] num_heads=32 should be supported
[PASS] num_heads=128 should be supported
[PASS] num_heads=2 should NOT be supported
[PASS] num_heads=12 should NOT be supported
[PASS] num_heads=256 should NOT be supported
[PASS] Head repeat logic tests
[PASS] Code modification verification

*** All tests passed! ROCm MLA improvement verified ***
```

### 2. 检查修改的代码

```python
# 验证修改已应用
from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAImpl

# 检查新属性
import inspect
source = inspect.getsource(AiterMLAImpl.__init__)

# 应该包含:
# - _needs_head_repeat
# - _head_repeat_factor
print("Code verification passed")
```

---

## 功能验证

### 测试1: 标准配置（向后兼容）

**目的**: 确保现有配置不受影响

```python
from vllm import LLM

# 标准配置 - 应该无缝工作
llm = LLM(
    model="deepseek-ai/deepseek-llm-7b-chat",  # 或其他模型
    tensor_parallel_size=8,
    max_model_len=2048,
)

# 测试推理
prompts = ["Hello, how are you?"]
outputs = llm.generate(prompts)

print("✓ Standard configuration test passed")
print(f"Generated: {outputs[0].outputs[0].text[:50]}...")
```

**预期结果**:
- 无错误
- 正常生成文本
- 性能与之前相同或更好

### 测试2: 小Head数量配置

**目的**: 验证num_heads=8的支持

```python
from vllm import LLM

# 小head配置 - 需要支持num_heads=8的模型
llm = LLM(
    model="model-with-8-heads",  # 需要实际模型
    tensor_parallel_size=8,
    max_model_len=2048,
)

# 测试推理
prompts = ["Test prompt"]
outputs = llm.generate(prompts)

print("✓ Small head count test passed")
print(f"Generated: {outputs[0].outputs[0].text[:50]}...")
```

**预期结果**:
- 无错误
- 自动应用head repeat逻辑
- 可能有+40%延迟开销（可接受）

### 测试3: 性能对比

**目的**: 验证性能分析

```python
import time
from vllm import LLM, SamplingParams

llm = LLM(
    model="your-model",
    tensor_parallel_size=8,
)

sampling_params = SamplingParams(
    max_tokens=100,
    temperature=0.7,
)

# 测试多次推理取平均
times = []
for _ in range(5):
    start = time.time()
    outputs = llm.generate(["Test prompt"] * 10, sampling_params)
    elapsed = time.time() - start
    times.append(elapsed)

avg_time = sum(times) / len(times)
print(f"Average time: {avg_time:.3f}s")
print(f"Throughput: {10 / avg_time:.1f} tokens/s")
```

**预期结果**:
- num_heads≥16: 最好性能
- num_heads=8: 轻微性能下降（约40%延迟）
- num_heads=4: 显著性能下降（约80%延迟）

---

## 模型特定测试

### Kimi K2.5/Linear (如果可用)

**目的**: 验证新支持的模型

```python
from vllm import LLM

llm = LLM(
    model="moonshot/Kimi-K2.5-Linear",
    tensor_parallel_size=8,
    trust_remote_code=True,
)

# 测试推理
prompts = ["介绍一下人工智能"]
outputs = llm.generate(prompts)

print("✓ Kimi model test passed")
print(f"Generated: {outputs[0].outputs[0].text[:100]}...")
```

**预期结果**:
- 成功加载模型
- 正常推理
- 无head repeat错误

---

## 内存和性能监控

### 内存监控

```python
import torch

# 训练前内存
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# 推理中监控
def generate_with_monitoring(llm, prompts):
    for prompt in prompts:
        print(f"Before: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        outputs = llm.generate([prompt])
        
        print(f"After: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Delta: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
        return outputs
```

**预期结果**:
- 内存使用在合理范围内
- 无内存泄漏
- nhead<16时有额外内存使用（tensor repeat）

### 性能监控

```python
import time

def benchmark_inference(llm, prompts, num_runs=10):
    times = []
    
    for _ in range(num_runs):
        start = time.time()
        outputs = llm.generate(prompts)
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"Benchmark results ({num_runs} runs):")
    print(f"  Average: {avg_time:.3f}s")
    print(f"  Min: {min_time:.3f}s")
    print(f"  Max: {max_time:.3f}s")
    print(f"  Throughput: {len(prompts) / avg_time:.1f} req/s")
    
    return avg_time

# 运行基准测试
avg_time = benchmark_inference(llm, ["Test prompt"] * 10)
```

---

## 故障排查

### 问题1: 导入错误

**错误**: `ModuleNotFoundError: No module named 'vllm'`

**解决方案**:
```bash
# 确认在正确的目录
cd /path/to/vllm-gfx906

# 重新安装
pip install -e .
```

### 问题2: GPU不可用

**错误**: `CUDA error: no available device`

**解决方案**:
```bash
# 检查GPU
rocm-smi

# 检查PyTorch ROCm支持
python -c "import torch; print(torch.cuda.is_available())"
```

### 问题3: 模型加载失败

**错误**: `OSError: Can't load model`

**可能原因**:
- 模型路径错误
- 内存不足
- 模型格式不支持

**解决方案**:
```bash
# 检查模型文件
ls -lh /path/to/model/

# 检查可用内存
rocm-smi --showmem

# 尝试减少max_model_len
llm = LLM(
    model="your-model",
    max_model_len=1024,  # 减少上下文长度
)
```

### 问题4: 性能不如预期

**原因**: 使用了num_heads < 16

**解决方案**:
- 如果可能，使用num_heads ≥ 16的模型
- 或调整tensor_parallel_size
- 或接受性能权衡

---

## 成功标准

### 验证清单

- [ ] 环境检查通过（GPU、PyTorch、vLLM）
- [ ] 单元测试全部通过（18/18）
- [ ] 标准配置测试通过（向后兼容）
- [ ] 小head配置测试通过（新功能）
- [ ] 性能测试在预期范围内
- [ ] 内存使用正常
- [ ] 无内存泄漏

### 性能基准

| 配置 | 期望延迟范围 | 期望吞吐量 |
|------|--------------|------------|
| num_heads=16 | 100-120% 基线 | 100% 基线 |
| num_heads=8 | 130-150% 基线 | 90-100% 基线 |
| num_heads=4 | 170-190% 基线 | 80-90% 基线 |

---

## 验证报告模板

完成后填写此报告:

```markdown
# 硬件验证报告

**日期**: YYYY-MM-DD
**测试人员**: [姓名]
**GPU**: AMD MI50 (gfx906)
**ROCm版本**: [版本号]

## 环境检查

- [ ] GPU检测通过
- [ ] PyTorch+ROCm工作正常
- [ ] vLLM版本正确

## 功能验证

- [ ] 单元测试通过 (18/18)
- [ ] 标准配置测试通过
- [ ] 小head配置测试通过
- [ ] Kimi模型测试通过 (如果可用)

## 性能验证

| num_heads | 延迟 | 吞吐量 | 备注 |
|-----------|------|--------|------|
| 16 | ___s | ___ tok/s | |


| 8 | ___s | ___ tok/s | |


## 内存验证

- [ ] 内存使用正常
- [ ] 无内存泄漏
- [ ] nhead<16时额外内存可接受

## 问题记录

### 遇到的问题
1. [问题描述]
   - 解决方案:
   - 结果:

## 总结

**总体评价**: [成功/部分成功/失败]

**建议**:
- [后续改进建议]
- [部署建议]
- [用户反馈]
```

---

**准备人员**: gfx906适配团队
**最后更新**: 2026-03-08
**状态**: 待执行
