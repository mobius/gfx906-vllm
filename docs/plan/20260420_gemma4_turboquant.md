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
