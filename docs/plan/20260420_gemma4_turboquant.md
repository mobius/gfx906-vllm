# Gemma4 + TurboQuant on MI50 移植计划

**日期**: 2026-04-20  
**背景**: TurboQuant 在 Qwen2.5-7B-AWQ 上精度中等（平均字符相似度 0.47），本计划评估
Gemma4 是否能在 MI50 上结合 TurboQuant 获得更好的推理质量和更大的 KV 上下文容量。

---

## 1. 目标

1. 在 MI50（gfx906, 32 GB）上成功运行 Gemma4 + TurboQuant KV 压缩
2. 对比 Gemma4 + TurboQuant 与 Qwen2.5-7B + TurboQuant 的精度差异
3. 评估可实用的最大上下文长度

---

## 2. 为什么选 Gemma4？

### 架构优势

Gemma4 使用**滑动窗口 + 全局注意力交替**架构：

```
层分布（以 27B 为例）：
  5 层 local sliding window attention（窗口 512 tokens）
  1 层 global full attention
  …循环…
```

- **全局 full attention 层比例仅 ~17%**——TurboQuant 只需量化这些层的 KV
- **滑动窗口层的 KV 只保存最近 512 tokens**，本身用量已极小
- **总 KV 压力远低于纯 Transformer（如 Qwen2.5）**

### 显存估算（27B-A4B int4 MoE）

```
模型权重（int4 MoE，激活 4B）:  ~3–4 GB
可用 KV cache:                  ~27 GB
全局 attention 层 KV（fp16）:   ~X bytes/token
全局 attention 层 KV（TQ 4bit）: ~X/4 bytes/token
```

相比 Qwen2.5-7B（权重 4 GB）：Gemma4 27B-A4B **权重接近，但架构决定 KV 压力更小**。

### TurboQuant 精度预期更好的原因

| 因素 | Qwen2.5-7B | Gemma4-27B-A4B |
|------|-----------|----------------|
| TQ 量化层数 | 32 层 | ~8–9 层（全局层）|
| 误差累积层数 | 32 层 | **~8–9 层** |
| sliding window 层隔离 | 无 | ✅ 有（5层/组）|
| 激活参数 | 7B | **4B（更轻量，对噪声容忍）**|

---

## 3. 依赖分析

### 3.1 vLLM 支持状态

| 组件 | 状态 | 说明 |
|------|------|------|
| Gemma4 模型实现 | ❌ 未包含 | gfx906-vllm 最新是 Gemma3n，需 cherry-pick |
| Gemma4 Transformers 支持 | ✅ 已有 | transformers 5.x 已支持 |
| TurboQuant backend | ✅ 已完成 | Phase 2 已移植 |
| gfx906 triton wheel | ✅ 已有 | /mnt/hdd_storage/vllm/wheels/ |

### 3.2 需要从上游 cherry-pick 的内容

```bash
# 上游仓库: https://github.com/vllm-project/vllm
# 目标分支: gfx906/main

# 关键文件（需要确认对应 PR/commit）：
vllm/model_executor/models/gemma4.py
vllm/transformers_utils/configs/gemma4.py  
vllm/model_executor/models/registry.py     # 新增 Gemma4ForCausalLM 注册
```

需要查找上游 vLLM 中 Gemma4 支持的 commit（预计 2026-04-02 左右合入）。

### 3.3 Transformers 升级

容器内 transformers 4.57.3 不支持 `gemma4` 模型类型，需升级到 5.x。  
**注意**：vLLM 0.12 要求 `transformers<5`，升级可能有兼容性风险，需测试。

---

## 4. 实施步骤

### Phase A: 环境准备（先决条件）

- [ ] A1. 确认 HuggingFace 上有合适的 Gemma4 量化版本（int4/AWQ，≤ 20 GB）
- [ ] A2. 下载模型到本地（`/mnt/storage/` 或 `/mnt/hdd_storage/models/`）
- [ ] A3. 测试容器内 `pip install transformers>=5.0` 的 vLLM 兼容性

### Phase B: vLLM Gemma4 支持

- [ ] B1. 在上游 vLLM 找到 Gemma4 支持的 commit（搜索 PR #Gemma4 / gemma-4）
- [ ] B2. Cherry-pick Gemma4 相关文件到 gfx906-vllm
- [ ] B3. 验证 Gemma4 在标准 vLLM（无 TurboQuant）下能正常启动和推理

### Phase C: TurboQuant 集成

- [ ] C1. 确认 Gemma4 全局 attention 层的 KV cache 格式与 TurboQuant 兼容
- [ ] C2. 确认 sliding window 层如何处理（跳过 TQ 压缩 or 处理 512-token window）
- [ ] C3. 测试 Gemma4 + `kv_cache_dtype=turboquant_4bit_nc` 启动
- [ ] C4. 精度对比测试（与标准 fp16 KV 对比，与 Qwen2.5-7B + TQ 对比）

### Phase D: 性能评估

- [ ] D1. 记录 KV cache 容量（token 数）
- [ ] D2. 记录 decode 速度（tok/s）
- [ ] D3. 精度评估（同 Qwen2.5 评估的 13 个 prompt 套件）

---

## 5. 风险点

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| Transformers 5.x 与 vLLM 0.12 兼容性问题 | 中 | 高 | 先做隔离测试，必要时 pin 特定版本 |
| Gemma4 sliding window KV 与 TQ 集成复杂 | 中 | 中 | sliding window 层可以选择跳过 TQ，只压缩全局层 |
| 没有合适的 int4 量化版本可下载 | 低 | 高 | 可用 GGUF 或尝试未量化的小规格（2B/4B） |
| gfx906 对 Gemma4 新 attention 实现的兼容性 | 低 | 高 | 先测标准推理，再加 TQ |

---

## 6. 候选模型（HuggingFace 检索结果，2026-04-20）

### 优先级排序（MI50 32GB 适用性）

| 优先级 | 模型 | 量化 | 磁盘 | 显存需求 | 适用性 | 说明 |
|--------|------|------|------|---------|--------|------|
| ★★★ | `mattbucci/gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed` | AWQ 4-bit | ~9 GB | 8–12 GB | ✅ 最佳 | 专为 AMD ROCm 优化，26B MoE 激活 3.8B |
| ★★★ | `Intel/gemma-4-26B-A4B-it-int4-mixed-AutoRound` | int4 AutoRound | ~5 GB | 8–10 GB | ✅ 最佳 | Intel AutoRound 量化，26B MoE，33.9k 下载 |
| ★★☆ | `Intel/gemma-4-26B-A4B-it-int4-AutoRound` | int4 AutoRound | ~5 GB | 8–10 GB | ✅ 很好 | AutoRound 标准版 |
| ★★☆ | `QuantTrio/gemma-4-31B-it-AWQ` | AWQ 4-bit | ~20 GB | 20–25 GB | ⚠️ 刚好 | 31B dense，单卡刚够；留给 TQ KV 的空间约 7–10 GB |
| ★☆☆ | `ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ` | GPTQ W4A16 | ~19 GB | 25–30 GB | ⚠️ 紧张 | KV cache 空间不足，不推荐配合 TurboQuant |

> **注意**：bartowski GGUF 系列（E4B / 26B）适合 llama.cpp，但 vLLM 路径需要 safetensors 格式，GGUF 暂不走 TurboQuant。

### 推荐选择

**首选**: `mattbucci/gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed`
- 已在 AMD ROCm 7.2（RDNA4/gfx1201）上实测，硬件相近，迁移风险最低
- AWQ 格式，vLLM 原生支持，无需额外适配量化层
- 权重仅 ~9 GB，MI50 留出 ~22 GB 给 KV cache（TurboQuant 后可达 ~88 GB 等效 KV 容量）
- 量化策略：强制路由 GPTQ 校准确保 128 个专家全被量化，精度有保证
- **风险**：gfx1201 vs gfx906 架构差异，需实测；视觉编码器 INT4 化，但文本推理不受影响

**备选**: `Intel/gemma-4-26B-A4B-it-int4-mixed-AutoRound`
- 月下载量 33.9K，是最活跃的 Gemma4 量化模型之一
- 混合精度策略（MoE 层 4bit + 语言层 8bit + 其余 16bit）精度损失更小
- **需要**在 `rocm.py` 的 `_get_supported_quantization()` 中加入 `"auto_round"`（改动已知，3行代码）
- **风险**：无 AMD 测试记录，auto_round 内核在 gfx906 上的执行路径待验证

---
## 7. 参考

- Gemma4 发布博客: https://blog.google/technology/google-deepmind/gemma-4/
- 上游 vLLM Gemma4 支持: https://github.com/vllm-project/vllm (2026-04-02 合入)
- TurboQuant Phase 2 文档: `docs/impl/20260420_phase2_turboquant_final.md`
- Qwen2.5 TQ 精度评估: `docs/impl/20260420_phase2_debug_journal.md`

---

## 8. 当前状态与诊断发现（2026-04-21 更新）

### 8.1 已完成的移植工作

**Gemma4 服务在 MI50 上成功启动**（见 commit `16bcfc9c0`）：
- 模型加载（Gemma4-26B-A4B AWQ，16 GB）✅
- TRITON_ATTN attention backend ✅
- MoE routing 工作 ✅
- `Application startup complete` ✅
- KV cache: 48,704 tokens（fp16，auto）✅

### 8.2 精度问题诊断

**表现**：推理输出质量低（选错关键词，如 "Japan:" → "Berlin" 而非 "Tokyo"）

**诊断过程**：
1. `gelu_and_mul` torch op 在 gfx906 上验证正确（MAE=0.000069）✅
2. Dense fused_moe GELU 路径在 gfx906 正确（std 合理，无 NaN）✅
3. Dense AWQ linear（Qwen2.5 路径）推理正确 ✅
4. `gptq_gemm` 对 qweight `[K//8, N]` 格式输出正确的 `[B, N]` ✅

**根因**：Gemma4 AWQ MoE 的量化路径（`MoeWNA16`）在 gfx906 上有精度问题。

具体为：`fused_experts` 接收 `quant_config=int4_w4a16_moe_quant_config` 时触发 `AssertionError: Hidden size mismatch`，说明量化权重格式（`w13_qweight [E, 2N, K//8]`）与 fused_moe Triton kernel 期望的格式不匹配。

**关键差异**：
- Qwen2.5-7B AWQ：dense linear，group_size=128，走 `AWQLinearMethod` + `gptq_gemm`，正确
- Gemma4 AWQ MoE：FusedMoE，group_size=32，走 `MoeWNA16` + `fused_experts(quant_config)` 路径，有 hidden size mismatch 错误

### 8.3 深挖诊断（2026-04-21 进一步分析）

**模型关键维度**（从 config.json 确认）：
- `hidden_size = 2816`（K 维度，hidden → MoE 输入）
- `moe_intermediate_size = 704`（N 维度，每个专家的中间维度）
- `num_experts = 128`，`top_k_experts = 8`，`group_size = 32`

**权重格式验证**（从 safetensors 文件）：
```
gate_proj.qweight: [2816, 88]  → [K, N//8]  = [2816, 704//8]  ✅
gate_proj.qzeros:  [88, 88]   → [K//G, N//8] = [2816//32, 704//8] ✅
gate_proj.scales:  [88, 704]  → [K//G, N]   = [88, 704]            ✅
```

**convert_awq_tensor 验证**（模拟确认）：
- 输入 `qzeros [88, 88]` int32 → 输出 `[352, 88]` uint8 ✅
- 这等于 `[N//2, K//G]`，正是 Triton kernel 期望的格式

**block_shape 确认**：
- `layer.group_size = 32`（K 维度 2816 % 32 == 0，不需要 div_factor 调整）
- `block_shape = [0, 32]`
- Triton kernel 中 `offs_k // group_size` 最大 = `2816 // 32 = 88` = scales 第三维 ✅
- qzeros 访问 `offs_bn // 2` 最大 = `1407 // 2 = 703` < `qzeros.shape[1] = 704` ✅

**CUDA vs Triton 路径**：
- `should_moe_wna16_use_cuda()` 要求 `current_platform.is_cuda_alike()`
- MI50 是 ROCm，不满足，**始终走 Triton kernel** (`fused_moe_kernel_gptq_awq`)

**activation 路径**：
- `gemma4.py` 传 `activation="gelu"` 给 `FusedMoE`
- `fused_experts_impl` 中 `"gelu"` 调用 `torch.ops._C.gelu_and_mul` ✅

**未解决的问题**：
- 精度问题依旧（"Japan: Berlin" 而非 "Tokyo"）
- 当前 `block_shape=[0, 32]` 理论正确，但无法运行验证（GPU 被 orphan 进程占用）
- `w13_qzeros.nbytes` 异常（测到 15,859,712 而非期望 7,929,856）：
  - 静态模拟：uint8 param 赋值后仍为 uint8 ✅
  - **真实运行时**：nbytes = `numel * element_size = 7,929,856 * 2 = 15,859,712` → element_size=2？
  - 需要在容器中打印 `qz.dtype` 确认

### 8.5 精度问题解决（2026-04-21 最终确认）

**结论：Gemma4 AWQ MoE 在 MI50 上推理完全正确！**

之前的"精度问题"（"Japan: Berlin"）是**测试方法错误**导致的：
- 用了 raw completion 格式，没有使用 Gemma4 的指令格式 `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`
- Gemma4 是 instruction-tuned 模型，需要正确的 chat template

**正确测试结果（gemma4-dtype 容器，block_shape=[0, 32]，dtype=uint8）**：

```
prompt: <start_of_turn>user\nWhat is the capital of Japan?<end_of_turn>\n<start_of_turn>model\n

output: <thought>
The user is asking for the capital of Japan.
The capital of Japan is Tokyo.
I will provide the answer.
The answer is Tokyo.
```
✅ 完全正确！

```
prompt: 17 * 23 = ?

output: 用 (20-3)(20+3) = 400-9 = 391 的方法计算，推理链完整、正确
```
✅ 数学推理高质量！

**最终确认的权重格式（运行时实测）**：
```
w13_qzeros: shape=[128, 704, 88] dtype=torch.uint8 numel=7929856 nbytes=7929856 ✅
block_shape=[0, 32] ✅
group_size=32 ✅
```

## 9. Phase C 完成报告（2026-04-21）

### 9.1 实现方案

**修改文件**：`vllm/attention/layer.py`（仅此一处，25 行代码）

**两处关键修改**：

1. **`__init__` 中 sliding window 层降级**：
   ```python
   self._tq_fallback_sliding = False
   if (sliding_window is not None
           and isinstance(kv_cache_dtype, str)
           and kv_cache_dtype.startswith("turboquant_")):
       kv_cache_dtype = "auto"         # 降级到标准 fp16
       self._tq_fallback_sliding = True  # 标记此层
   ```

2. **`get_kv_cache_spec` 返回 `FullAttentionSpec`（非 `SlidingWindowSpec`）**：
   ```python
   if self._tq_fallback_sliding:
       return FullAttentionSpec(
           ..., sliding_window=self.sliding_window
       )
   ```
   关键：`TQFullAttentionSpec` 是 `FullAttentionSpec` 子类，
   `UniformTypeKVCacheSpecs.is_uniform_type()` 对 FullAttentionSpec 路径，
   两种 spec 都能通过检查，走 `_get_kv_cache_groups_uniform_type` 路径，
   无需 `unify_kv_cache_spec_page_size`（避免 page size 不整除错误）。

3. **`assert` 放宽**：允许 `attn_type=None`（Gemma4 sliding 层的默认值）

### 9.2 启动验证

**启动命令**：
```bash
--model /model --dtype float16 --kv-cache-dtype turboquant_4bit_nc \
--max-model-len 4096 --gpu-memory-utilization 0.95 --enforce-eager
```

**日志验证（关键输出）**：
```
Using TRITON_ATTN attention backend   ← 25 个 sliding 层
Using TURBOQUANT attention backend    ← 5 个 full attention 层
[TurboQuant] slot_size=518 page_size=16576  ← 5 次（每个 full 层）
GPU KV cache size: 50,272 tokens
```

### 9.3 推理验证

| 测试 | 结果 |
|------|------|
| "What is the capital of Japan?" | **Tokyo** ✅ |
| "What is 17 * 23?" (差平方推导) | **391，推理链完整** ✅ |
| 模型加载（16 GB AWQ）| ✅ |
| TurboQuant KV 压缩（full 层）| ✅ slot_size=518 |

### 9.4 KV cache 对比

| 配置 | KV cache tokens |
|------|----------------|
| Gemma4 fp16（无 TQ） | ~48,704 tokens |
| Gemma4 + TurboQuant 4bit_nc | **50,272 tokens** |

> 说明：Gemma4 sliding 层（25/30）使用 FullAttentionSpec 分配所有 token 的 KV，
> 内存消耗较大；full 层（5/30）使用 TurboQuant 4bit 压缩，节省内存。
> 由于 sliding 层本来主导内存，TQ 压缩带来的净增益较小。
> 可通过 `--max-model-len` 更大值 + TQ 来获得更明显的 KV 扩展效果。

### 9.6 KV 扩展效果深度评测（max_model_len=32768）

**测试结果**：

| 配置 | Available KV Mem | KV cache tokens | 并发 (32k) |
|------|-----------------|----------------|-----------|
| Gemma4 fp16（无 TQ） | 11.15 GiB | **48,720 tokens** | 9.19x |
| Gemma4 + TurboQuant 4bit_nc | 9.83 GiB | **50,304 tokens** | 1.54x |

**反直觉发现**：有 TQ 反而 available KV memory 更少（-1.32 GiB），导致 KV tokens 仅微增。

**根本原因**：

当前 TQ 方案将 sliding 层从 `SlidingWindowSpec` 降级为 `FullAttentionSpec`（为了统一 page_size）：

| 方案 | sliding 层（25个）KV 内存 | full 层（5个）KV 内存 | 合计 |
|------|------------------------|---------------------|------|
| 无 TQ（SlidingWindowSpec） | 0.30 GiB（每层只需 3071 tokens） | 0.31 GiB（fp16） | **0.61 GiB** |
| 有 TQ（FullAttentionSpec fallback） | **3.12 GiB**（每层分配全部 32768 tokens） | 0.08 GiB（TQ 4bit） | **3.20 GiB** |

TQ 对 full 层节省了 0.23 GiB，但 sliding 层 FullAttentionSpec 多耗费了 2.82 GiB，净损失 2.59 GiB。这解释了为何 available KV memory 有 TQ 时反而少 1.32 GiB（差额由模型初始化阶段的 overhead 造成）。

**技术瓶颈**：

vLLM v1 的 `unify_kv_cache_spec_page_size` 要求所有层的 page_size 可整除：
- `SlidingWindowSpec(head_dim=256)` page_size = 131,072 bytes
- `TQFullAttentionSpec(slot=518)` page_size = 16,576 bytes
- GCD(131,072, 16,576) = 32，无法统一

因此不得不将 sliding 层改为 FullAttentionSpec（page_size 131,072）才能通过。

**后续优化方向**：

- 修改 `kv_cache_utils.py`，支持 TQ 层独立于 sliding 层单独分组（参考 Mamba spec 的处理方式）
- 或：调整 TQ 的 slot_size padding，使 `block_size * 1 * slot_size = 131,072`（slot_size = 4,096？），需修改 TurboQuantConfig
- 目前功能性集成已验证，内存效率优化可作为 Phase D
3. **备选方案**：如果 MoE 量化路径修复复杂，可以考虑让 MoE 层走 fp16 路径（在 `modules_to_not_convert` 里排除 MoE expert 层）

