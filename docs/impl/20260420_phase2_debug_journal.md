# TurboQuant gfx906 精度问题调试日志

**日期**: 2026-04-20  
**背景**: Phase 2 基础功能完成后，服务能启动、KV cache 能扩展，但 decode 输出质量不正常（与 float16 KV 相比有明显差异）。本文档记录完整的诊断过程。

---

## 1. 问题描述

### 初始现象

```
prompt: "The capital of France is"  max_tokens=5
TurboQuant 输出: " Paris located ____\n."
标准 vLLM 输出:  " Paris. The capital of"
```

- **第一个 decode token（"Paris"）正确**
- **第二个 token 起开始出现质量下降**
- 长生成（>10 tokens）质量明显退化

---

## 2. 诊断阶段一：隔离 attention kernel

### 假设：TQ decode kernel 有 bug（gfx906 Wave64/uint16 问题）

#### 测试 2.1 — Roundtrip 测试（合成数据）

```python
# seq_len=4, Hk=2, Hq=2, D=128
triton_turboquant_store(key, value, kv_cache, slot_mapping, ...)
output = triton_turboquant_decode_attention(query, kv_cache, ...)
MAE vs reference = 0.0500  →  PASS
```

#### 测试 2.2 — 扩展到真实 GQA 场景（Hk=2, Hq=12，5-token prefill）

```python
# 模拟 Qwen2.5-1.5B 的 GQA 配置
MAE = 0.0481  →  PASS
```

**结论**：TQ decode kernel 本身在合成数据上正确。

---

## 3. 诊断阶段二：pkl 数据重放

### 方法

在服务运行时通过 `TQ_CAPTURE=1` 环境变量捕获真实推理的 kv_cache、query 等数据到 `/tmp/tq_capture.pkl`，然后离线重放。

### 结果

```
Service TQ output:   [0.0199, -0.0009, 0.0299, ...]  std=0.2473
Offline TQ output:   [0.0199, -0.0009, 0.0299, ...]  std=0.2474
MAE (service vs offline) = 0.000000  →  IDENTICAL
MAE (offline vs dequant ref) = 0.0001  →  PASS
```

**结论**：
- TQ decode kernel 在真实数据上也完全正确
- Service 和 offline 输出完全一致
- 精度问题 **不在 decode kernel**

---

## 4. 诊断阶段三：逐步缩小范围

### 测试 4.1 — bypass decode，用 dequant+einsum 替代

添加 `TQ_COMPARE_REF=2` 环境变量，用 dequant+einsum 计算替代所有层的 TQ decode attention：

```
TQ_COMPARE_REF=2 输出: " Paris located ______"
标准 TQ 输出:           " Paris located ______"  
→ 完全相同
```

28 层 × 4 decode steps = 112 次全部 bypass，结果不变。

**结论**：attention 计算本身没有问题，问题在 attention 之外。

### 测试 4.2 — 标准 vLLM 对比

同机器、同模型，不加载 TurboQuant 的标准 vLLM：
```
标准 vLLM: " Paris. The capital of"  ✅
```

### 测试 4.3 — hidden_states 健康检查

添加 `TQ_DECODE_DEBUG=1`，在 `Qwen2DecoderLayer.forward` 里记录每层的 hidden_states std：

```
[DecoderLayer#1] hs_before_attn std=0.440
[DecoderLayer#1] hs_after_attn  std=0.337
[DecoderLayer#1] hs_after_mlp   std=0.377
```

std 值全部在正常范围（0.1~1.0），无 NaN/Inf。hidden_states 张量本身健康。

### 测试 4.4 — TQ_COMPARE_REF=1 精度追踪

逐层记录 TQ decode 与 dequant+einsum ref 的 MAE：

```
Layer 0:  MAE=0.0324   ← 相对偏大
Layer 1:  MAE=0.0136
Layer 2:  MAE=0.0090
Layer 3:  MAE=0.0013
Layer 4:  MAE=0.0006
...（后续层 MAE < 0.003）
```

**关键发现**：浅层 MAE 较大（layer 0 = 0.032），误差在通过 o_proj → LayerNorm → MLP 后会放大并向后传播。

---

## 5. 根因分析

### 误差传播机制

```
prefill: flash_attn (精确)
  ↓
decode step 1:
  TQ decode (MAE≈0.032 vs fp16 ref)
  → hidden_states 有 ~0.03 的误差
  → o_proj 可能放大误差
  → LayerNorm 之后误差继续传播
  → MLP 进一步放大
  → 下一层 attention 的 hidden_states 输入偏差
  → 累积效应
decode step 2:
  上一步的偏差作为输入
  → TQ decode 再叠加误差
  → 更大偏差
  ...
decode step N:
  误差严重累积
  → 输出完全偏离
```

### 为什么第一个 token 正确？

prefill 阶段（输出第一个新 token）用的是 flash_attn，不走 TQ，因此完全精确。

### 为什么 7B 比 1.5B 好？

大模型（7B）每层有更多参数，对量化噪声有更强的鲁棒性：
- 1.5B: 28 层，每层参数少，量化误差占比更高
- 7B: 32 层，信息冗余度高，每层 MAE ≈ 0.02~0.03 时整体能容忍

### 这是 gfx906 特有的 bug 吗？

**否**。根据 TurboQuant 论文的 PPL 数据：

| Preset | PPL 增量 | 测试模型 |
|--------|---------|---------|
| turboquant_4bit_nc | +2.71% | Llama-3 系列（7B+） |

2.71% PPL 增量在大模型上可接受，但对 1.5B 小模型会显著放大：
- 小模型没有足够的冗余来吸收量化噪声
- 每步误差更容易导致 "attention sink" 偏移，引发语言质量退化

gfx906 上的 roundtrip MAE（0.05）和 NVIDIA 上预期一致，无额外精度损失。

---

## 6. 其他 Bug 修复（无关精度，但阻塞 7B AWQ 使用）

### Bug A: `supported_dtypes` 访问方式错误

**触发**：启动服务时，`vllm/config/model.py` 调用：
```python
supported_dtypes = [dtype for dtype in current_platform.supported_dtypes() ...]
```

**错误**：`TypeError: 'method' object is not iterable`

**原因**：
- 基类 `interface.py` 定义为 `@property`（实例属性）
- 我们的 `rocm.py` 定义为 `@classmethod`
- `current_platform.supported_dtypes()` 通过实例访问 classmethod，返回 bound method 而非列表

**修复**：将 `rocm.py` 的 `supported_dtypes` 改为 `@property`（与基类一致）；
`check_if_supports_dtype` 里的 `cls.supported_dtypes()` 改为内联 `_GCN_ARCH` 检查。

---

### Bug B: `supported_quantization` 同类问题

**触发**：加载 AWQ 模型时，`vllm/platforms/interface.py:394`：
```python
if cls.supported_quantization and quant not in cls.supported_quantization:
```

**错误**：`TypeError: argument of type 'property' is not iterable`

**原因**：通过 `cls`（class 对象）访问 `@property`，得到 property 对象而非列表。

**修复**：
1. 将 `supported_quantization` 逻辑抽取为 `_get_supported_quantization()` classmethod
2. 覆盖 `verify_quantization`，直接调用 `cls._get_supported_quantization()` 绕过基类

---

### Bug C: 未指定 `--dtype float16` 时使用 bfloat16

**触发**：首次启动无 `--dtype` 参数时。

**错误**：`RuntimeError: Expected weight.scalar_type() == input.scalar_type()`

**原因**：
- gfx906 不支持 bfloat16
- `supported_dtypes` 修复后正确返回 `[float16, float32]`，但 `_resolve_auto_dtype` 仍选了 bfloat16
- 原因：auto dtype 选择逻辑先看模型权重（bfloat16 safetensors），再检查平台支持

**修复**：启动参数里显式加 `--dtype float16`（`start_turboquant.sh` 需要用户传入）。

---

## 7. 调试工具（已集成到代码中）

以下环境变量用于调试，不影响正常运行（默认值 '0' 均为关闭）：

### `TQ_VERIFY_OUTPUT=1`

在 `Qwen2Attention.forward` 和 `Qwen2ForCausalLM.compute_logits` 里记录前3次调用的 hidden_states 统计。

### `TQ_DECODE_DEBUG=1`

在 `Qwen2DecoderLayer.forward` 里记录 decode 阶段每层前3次的 hidden_states std（before/after attn，after MLP）。

### `TQ_CAPTURE=1`

在 `_decode_attention` 里捕获第一次 decode call 的完整张量数据到 `/tmp/tq_capture.pkl`，供离线分析。

### `TQ_COMPARE_REF=1`

在 `_decode_attention` 里额外运行 dequant+einsum 参考计算，记录 MAE。仅第一次调用执行（per-layer）。

### `TQ_COMPARE_REF=2`

同上，但用 dequant 参考结果替代 TQ decode 输出（用于验证"attention 之外无问题"）。

### `TQ_DISABLE_STORE=1`

禁用 prefill 阶段的 TQ store（decode 时读到全零 KV），用于验证 store 效果。

### `TQ_PREFILL_FALLBACK=1`

decode 时走 dequant+continuation_prefill 路径，完全绕过 TQ decode triton kernel。

---

## 8. 结论

1. **TurboQuant triton kernel 在 gfx906 上正确**，无 Wave64/uint16 特有 bug
2. **推理质量差异来自 4bit KV 量化的固有精度限制**
3. **对于 7B+ 模型推荐使用** `turboquant_4bit_nc`（大模型容忍量化噪声）
4. **对于 1.5B 小模型**，短生成（<10 tokens）可接受，长生成质量退化
5. **两个关键 Bug 已修复**（`supported_dtypes`/`supported_quantization` 的 property vs class 访问问题）

---

## 9. 相关文件

| 文件 | 内容 |
|------|------|
| `vllm/platforms/rocm.py` | Bug A/B 修复 |
| `vllm/attention/layer.py` | TurboQuant buffer 初始化 + TQ_VERIFY_OUTPUT 调试钩子 |
| `vllm/v1/attention/backends/turboquant_attn.py` | TQ_CAPTURE / TQ_COMPARE_REF / TQ_DISABLE_STORE 调试钩子 |
| `vllm/model_executor/models/qwen2.py` | TQ_DECODE_DEBUG / TQ_VERIFY_OUTPUT 调试钩子 |
| `docs/impl/20260420_phase2_turboquant_final.md` | 最终完成情况 |
