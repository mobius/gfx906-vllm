# Phase 3: 分阶段升级执行计划

**创建日期**: 2026-03-08
**状态**: 🔄 规划中
**目标**: v0.11.1 → v0.13.0 → v0.15.0 → v0.17.0

---

## ⚠️ 策略调整

### 发现的问题

1. **版本标签缺失**:
   - upstream vLLM仓库没有标准的git标签（v0.11.1, v0.12.0等）
   - 版本信息可能通过GitHub releases管理
   - 无法直接使用基于标签的升级路径

2. **commit数量巨大**:
   - 2024年全年：3,352个commits
   - Q1 2024：392个commits
   - 无法准确定位v0.13.0的commit ID

3. **架构差异显著**:
   - Phase 2发现：V1 vs V0引擎差异
   - 配置文件结构重组
   - 依赖关系复杂

### 修订策略

**原计划**: v0.11.1 → v0.13.0 → v0.15.0 → v0.17.0
**新策略**: 基于时间范围的渐进式升级

#### 方案A: 按季度升级（推荐）

```
当前 (v0.11.1, 2024年初)
    ↓ Phase 3A
2024年Q2末 (约500-800 commits)
    ↓ Phase 3B
2024年Q3末 (约500-800 commits)
    ↓ Phase 3C
2024年Q4末 (约500-800 commits)
    ↓
2025年Q1 (最新upstream)
```

**优势**:
- 明确的时间边界
- 可预测的commit数量
- 便于验证和回滚

#### 方案B: 关键功能点升级

选择特定功能引入的commit作为升级节点：
- FlashAttention重大改进
- torch.compile集成
- V1引擎稳定化

#### 方案C: 延迟大规模升级

- 继续选择性同步（Phase 2B）
- 等待更明确的时机
- 等待业务需求驱动

---

## 📊 当前状态分析

### gfx906 (v0.11.1)

**已知信息**:
- 基于v0.11.1（2024年初）
- 当前commit: `0fb53d72d` (Phase 2结束)
- 分支: `gfx906/phase2-selective-sync`
- 已完成Phase 1和Phase 2

### upstream (最新)

**commit信息**:
- 最新commit: `ee8a29511` (2024-12-19)
- 包含DeepSeek-R1量化修复
- 2024年全年：3,352个commits

**关键修复** (Phase 2已识别):
- Ray reinit error: `47826cacf` ✅ 已应用
- Compressed tensor: `ac7979940` ⚠️ 不适用
- Speculative decoding: `b63ba8483` ❌ 架构差异
- Multi-step: `9e5ec35b1` ❌ 文件不存在
- Quantization: `b3195bc9e` ⚸️ 需验证

---

## 🎯 推荐方案: Phase 3A - Q2 2024升级

### 目标

升级到2024年Q2末（约2024年6月30日）的upstream状态

### 时间表

- **准备**: 1天
- **执行**: 2-3天
- **测试**: 2-3天
- **总计**: 1周

### 执行步骤

#### Day 1: 分析和准备

```bash
# 1. 查找Q2末的commit
cd D:/VLLM/vllm
git log --oneline --all --since="2024-06-01" --until="2024-07-01" | tail -1

# 2. 分析Q2的ROCm相关变更
git log --oneline --all --since="2024-04-01" --until="2024-07-01" \
  --grep="AMD\|ROCm\|rocm" > ../vllm-gfx906/q2-2024-rocm-fixes.txt

# 3. 评估变更数量
git log --oneline --all --since="2024-04-01" --until="2024-07-01" | wc -l

# 4. 创建升级分支
cd ../vllm-gfx906
git checkout -b gfx906/phase3a-upgrade-q2-2024
```

#### Day 2-3: 执行升级

```bash
# 1. 获取upstream最新代码
cd D:/VLLM/vllm
git fetch origin
git checkout main

# 2. 找到Q2末的commit ID
TARGET_COMMIT=$(git log --oneline --all --since="2024-06-01" --until="2024-07-01" | tail -1 | awk '{print $1}')

# 3. 在gfx906中准备merge
cd ../vllm-gfx906
git merge $TARGET_COMMIT --no-commit --no-ff
```

#### 保护文件处理

```bash
# 这些文件必须保留gfx906的版本
GFX906_PROTECTED_FILES=(
    "requirements/rocm.txt"
    "vllm/platforms/rocm.py"
    "vllm/transformers_utils/config.py"
    "vllm/config/model.py"
)

# 解决冲突时保留gfx906版本
for file in "${GFX906_PROTECTED_FILES[@]}"; do
    git checkout --ours $file
done
```

#### Day 4-5: 测试和验证

```bash
# 1. 运行Phase 1的测试
python test_rocm_mla_fix.py

# 2. 运行ROCm相关测试
python -m pytest tests/ -k "rocm or amd" -v

# 3. 性能基准测试
python benchmark_performance.py

# 4. 功能验证
python -c "from vllm import LLM; print('Import successful')"
```

---

## 📋 决策矩阵

### 继续Phase 3A的条件

✅ **继续** 如果:
- Q2变更数量 < 1000 commits
- ROCm相关修复 < 50个
- 测试通过率 > 80%
- 可在1周内完成

❌ **停止** 如果:
- Q2变更数量 > 1500 commits
- 出现架构级冲突
- 关键功能损坏
- 时间超出预期

### 停止后的选项

1. **回滚到Phase 2**:
   ```bash
   git reset --hard gfx906/phase2-selective-sync
   ```

2. **尝试更小的范围**:
   - 只升级Q2的ROCm修复
   - 选择性merge特定commits

3. **暂停升级**:
   - 继续选择性同步
   - 等待更合适的时机

---

## 📚 相关文档

### 参考文档

1. **STAGED_UPGRADE_EVALUATION.md** - 分阶段升级评估
2. **PHASE2_SELECTIVE_SYNC_SUMMARY.md** - Phase 2总结
3. **MAINTENANCE_LOG.md** - 维护日志

### 关键经验

**从Phase 2学到的**:
- 单commit移植适用于简单修复
- 架构差异是主要障碍
- 需要平衡风险和收益

**应用到Phase 3**:
- 采用更保守的范围（按季度）
- 重视测试验证
- 准备回滚计划

---

## 🚀 立即行动

### 下一步（今天）

1. ✅ **确认策略**: 与团队讨论Phase 3A计划
2. ⏸️ **分析Q2变更**: 统计commit数量和ROCm修复
3. ⏸️ **评估风险**: 识别潜在冲突点
4. ⏸️ **制定详细计划**: 具体到每天的task

### 本周目标

- [ ] 完成Q2变更分析
- [ ] 创建Phase 3A分支
- [ ] 执行升级到Q2末
- [ ] 运行测试验证
- [ ] 决策继续或停止

---

**文档创建**: 2026-03-08
**状态**: 等待团队确认
**下一步**: 分析Q2 2024变更
