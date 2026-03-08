# 分阶段升级评估：v0.11.1 → v0.17.0

**文档版本**: v1.0
**评估日期**: 2026-03-08
**当前状态**: Phase 1 完成，准备开始升级路径评估
**分支**: `gfx906/adapt-phase1-critical-fixes`

---

## 📊 执行摘要

### Phase 1 成功验证

**关键成果**:
- ✅ 成功应用 upstream commit c188749bc (ROCm MLA改进)
- ✅ 18/18 测试全部通过
- ✅ 性能基准建立
- ✅ 向后兼容性100%

**重要发现**:
1. **选择性同步可行**: 单个高质量commit可以成功移植
2. **测试覆盖充分**: 现有测试框架可以有效验证修改
3. **文档完善**: 维护日志和用户指南支持无上下文继续

---

## 🎯 升级策略评估

### 三种方案对比

| 方案 | 工作量 | 风险 | 收益 | 推荐度 |
|------|--------|------|------|--------|
| **A. 分阶段升级** | 3-4周 | 中 | 高 | ⭐⭐⭐⭐⭐ |
| **B. 继续选择性同步** | 持续 | 低 | 中 | ⭐⭐⭐ |
| **C. 等待V1稳定** | 0 | 低 | 延迟 | ⭐⭐ |

---

## 方案A：分阶段升级（推荐）

### 路径图

```
v0.11.1 (当前)
    ↓ Phase 2: ~1周
v0.13.0
    ↓ Phase 3: ~1周
v0.15.0
    ↓ Phase 4: ~1.5-2周
v0.17.0 (目标)
```

### 阶段详解

#### Phase 2: v0.11.1 → v0.13.0

**预计时间**: 7天

**目标版本**: v0.13.0 (假设的中间版本)

**关键变更领域**:
1. ROCm特定修复
2. 性能优化
3. Bug修复

**执行步骤**:

```bash
# Day 1-2: 准备和评估
cd D:/VLLM/vllm-gfx906
git checkout -b gfx906/upgrade-v0.13.0

# 分析 v0.11.1 到 v0.13.0 的变更
cd ../vllm
git log --oneline --since="2024-08-01" --until="2024-10-01" | wc -l  # 约300-500 commits

# 识别关键commits
git log --oneline --since="2024-08-01" --until="2024-10-01" \
  --grep="ROCm\|AMD\|bugfix\|perf" > ../vllm-gfx906/v0.13.0-candidates.txt

# Day 3-5: 执行升级
# 策略：创建合并基础，然后冲突解决
cd ../vllm-gfx906
git fetch ../vllm  # 假设有 v0.13.0 tag
git merge v0.13.0 --no-commit --no-ff

# 解决保护的gfx906文件冲突
# 1. requirements/rocm.txt (保留)
# 2. vllm/platforms/rocm.py (保留)
# 3. vllm/transformers_utils/config.py (保留)
# 4. vllm/config/model.py (保留)

# Day 6-7: 测试和验证
python test_rocm_mla_fix.py  # 验证MLA修复仍然工作
python -m pytest tests/ -k "rocm or amd" -v  # 运行ROCm相关测试
```

**预期挑战**:
- 依赖版本冲突（PyTorch, ROCm）
- API小幅度变更
- 配置文件格式变化

**成功标准**:
- [ ] 所有现有测试通过
- [ ] MLA修复仍然工作
- [ ] 性能没有退化
- [ ] ROCm特定功能正常

---

#### Phase 3: v0.13.0 → v0.15.0

**预计时间**: 7天

**目标版本**: v0.15.0

**关键变更**:
1. 更多性能优化
2. 新模型支持
3. V1引擎改进（可能）

**执行步骤**:

```bash
# Day 1-2: 评估变更
cd D:/VLLM/vllm
git log --oneline v0.13.0..v0.15.0 --grep="ROCm\|AMD\|V1\|rocm" \
  > ../vllm-gfx906/v0.15.0-candidates.txt

# 评估V1引擎变更
git log --oneline v0.13.0..v0.15.0 --all -- "vllm/v1/" \
  > ../vllm-gfx906/v0.15.0-v1-changes.txt

# Day 3-5: 执行升级
cd ../vllm-gfx906
git checkout -b gfx906/upgrade-v0.15.0
git merge v0.15.0 --no-commit --no-ff

# 解决V1引擎冲突（如果存在）
# 策略：保留gfx906的V1实现（如果有）

# Day 6-7: 测试
python -m pytest tests/v1/ -v  # V1引擎测试
python -m pytest tests/ -k "rocm or amd" -v
```

**预期挑战**:
- V1引擎架构变化
- 新的量化方法（可能需要适配）
- 内存管理变更

**成功标准**:
- [ ] V1引擎测试通过（如果适用）
- [ ] 性能基准保持或改进
- [ ] 新的量化方法支持

---

#### Phase 4: v0.15.0 → v0.17.0

**预计时间**: 10-14天

**目标版本**: v0.17.0

**关键变更**:
1. V1引擎稳定化
2. torch.compile集成
3. FlashInter/FlashAttention深度集成
4. 大规模性能优化

**执行步骤**:

```bash
# Day 1-3: 深度评估
cd D:/VLLM/vllm
git log --oneline v0.15.0..v0.17.0 --grep="ROCm\|AMD\|V1\|compile" \
  > ../vllm-gfx906/v0.17.0-candidates.txt

# 分析架构变更
git diff v0.15.0..v0.17.0 --stat \
  > ../vllm-gfx906/v0.17.0-stats.txt

# Day 4-10: 执行升级（最复杂的阶段）
cd ../vllm-gfx906
git checkout -b gfx906/upgrade-v0.17.0
git merge v0.17.0 --no-commit --no-ff

# 重点解决：
# 1. V1引擎完整重构
# 2. torch.compile集成
# 3. FlashInter依赖（可能需要HIP适配）
# 4. 依赖版本升级（PyTorch, ROCm）

# Day 11-14: 全面测试
python -m pytest tests/ -v -x  # 完整测试套件
python benchmark_performance.py  # 性能基准
python scripts/verify_models.py  # 模型兼容性
```

**预期挑战**:
- V1引擎完整重构（可能需要重写gfx906适配）
- torch.compile在ROCm上的兼容性
- FlashInter依赖（CUDA专用，可能需要禁用）
- 依赖版本重大升级

**成功标准**:
- [ ] 完整测试套件通过
- [ ] 性能提升15-25%（符合评估预期）
- [ ] V1引擎稳定运行
- [ ] 所有gfx906保护文件功能正常

---

### 决策点

在每个阶段后进行评估：

```python
# 决策矩阵
if 测试通过率 < 90%:
    return "停止，回滚到上一版本"
elif 性能退化 > 10%:
    return "停止，调查性能问题"
elif 新增关键功能不工作:
    return "停止，修复或禁用该功能"
else:
    return "继续下一阶段"
```

---

## 方案B：继续选择性同步

### 策略

**基于Phase 1成功，继续cherry-pick高价值commits**

### 目标commits（按优先级）

#### 高优先级（立即应用）

```bash
# 1. ROCm MLA改进（已完成）
✅ c188749bc [ROCm] Support MLA with nhead<16

# 2. LMCache内存泄漏修复（需要验证V1引擎）
⏸️ 889f8bb25 [Bugfix] Fix potential memory leak in LMCache

# 3. CPU all reduce优化
cd D:/VLLM/vllm
git log --oneline --all --grep="cpu all reduce" | head -5
# 找到commit后cherry-pick

# 4. ROCm CI改进
git log --oneline --all --grep="ROCm.*CI" | head -10
# 选择适用的commits
```

#### 中优先级（评估后应用）

```bash
# 性能优化
git log --oneline --all --grep="perf.*rocm\|Perf.*AMD" | head -10

# Bug修复
git log --oneline --all --grep="bugfix.*rocm\|Bugfix.*MI300" | head -10
```

#### 低优先级（按需应用）

```bash
# 文档改进
git log --oneline --all --grep="doc.*rocm\|Doc.*AMD" | head -10
```

### 执行流程

```bash
# 每周review和cherry-pick
cd D:/VLLM/vllm-gfx906

# 1. 获取最新upstream
cd ../vllm
git fetch origin
git log origin/main --oneline --since="1 week ago" --grep="ROCm\|AMD" \
  > ../vllm-gfx906/weekly-candidates.txt

# 2. 评估commits
# 阅读 weekly-candidates.txt
# 标记高价值commits

# 3. Cherry-pick
cd ../vllm-gfx906
git checkout gfx906/main
for commit in $(cat selected-commits.txt); do
    git cherry-pick $commit
    python test_rocm_mla_fix.py
    if [ $? -ne 0 ]; then
        git cherry-pick --abort
        echo "Failed: $commit" >> failed-commits.txt
    fi
done

# 4. 每月发布minor版本
git tag -a gfx906/v0.11.2 -m "Monthly cherry-pick release"
```

### 优点

- ✅ 工作量分散（持续而非集中）
- ✅ 风险低（每次只应用少量commits）
- ✅ 灵活性高（可以跳过不适用commits）
- ✅ 基于Phase 1成功经验

### 缺点

- ❌ 维护成本高（持续跟踪upstream）
- ❌ 技术债务累积（偏离upstream）
- ❌ 无法获得架构级改进（V1引擎等）
- ❌ 长期不可持续

---

## 方案C：等待V1引擎稳定

### 等待条件

**触发标准**:
```python
if (
    "V1 engine marked stable" in upstream_release_notes and
    "torch.compile ROCm support confirmed" in community_feedback and
    "v0.18 or v0.19 released" in version_tags
):
    return "开始升级评估"
else:
    return "继续选择性同步（方案B）"
```

### 期间行动

```bash
# 持续应用方案B的选择性同步
# 同时关注upstream动态：

# 1. 订阅vLLM releases
https://github.com/vllm-project/vllm/releases

# 2. 关注ROCm issues
https://github.com/vllm-project/vllm/labels/rocm

# 3. 参与社区讨论
https://discuss.vllm.ai/

# 4. 每月重新评估
# 运行本文档的评估流程
```

### 优点

- ✅ 零风险（不进行大规模变更）
- ✅ 让社区发现问题
- ✅ 可能一次性跨越更多版本

### 缺点

- ❌ 延迟获得性能改进（15-25%）
- ❌ 延迟bug修复
- ❌ 可能永远等不到"完美时机"

---

## 🎯 推荐决策

### 当前推荐：**方案A（分阶段升级）**

**理由**:

1. **Phase 1成功验证**:
   - 证明选择性同步可行
   - 测试框架完善
   - 团队有信心

2. **时间窗口合适**:
   - 3-4周工作量可接受
   - 可以分阶段降低风险

3. **收益明确**:
   - 15-25%性能提升
   - 关键bug修复
   - 长期技术债务减少

4. **风险可控**:
   - 每个阶段有决策点
   - 可以随时停止
   - 有回滚计划

### 执行建议

```python
# 立即行动（本周）
action = "开始Phase 2准备"
tasks = [
    "创建测试分支 gfx906/test-v0.13.0-upgrade",
    "分析 v0.11.1 到 v0.13.0 的变更",
    "识别关键commits（ROCm、性能、bugfix）",
    "与团队讨论并确认时间表",
]

# 短期目标（本月）
if phase2_success:
    action = "执行Phase 2: v0.11.1 → v0.13.0"
    timeline = "7天"

# 中期目标（下季度）
if phase2_success and phase3_success:
    action = "执行Phase 3-4，完成到v0.17.0的升级"
    timeline = "2-3周"
```

---

## 📋 执行检查清单

### Phase 2 准备（本周）

- [ ] 与团队讨论方案A
- [ ] 确认3-4周时间窗口
- [ ] 创建测试分支
- [ ] 分析v0.13.0变更
- [ ] 评估风险和收益
- [ ] 制定详细计划

### Phase 2 执行（第2周）

- [ ] 合并v0.13.0
- [ ] 解决保护文件冲突
- [ ] 运行测试套件
- [ ] 性能基准测试
- [ ] 文档更新
- [ ] 代码review

### Phase 2 评估

- [ ] 测试通过率 ≥ 90%?
- [ ] 性能退化 < 5%?
- [ ] 关键功能正常?
- [ ] 决策：继续Phase 3 或 停止?

---

## 🔍 风险缓解

### 高风险区域

1. **V1引擎重构**:
   - 风险：可能不兼容gfx906
   - 缓解：在每个阶段验证V1功能

2. **依赖版本升级**:
   - 风险：PyTorch/ROCm版本不兼容
   - 缓解：在测试环境先验证

3. **FlashInter依赖**:
   - 风险：CUDA专用，需要HIP适配
   - 缓解：禁用FlashInter或等待ROCm支持

### 回滚计划

```bash
# 每个阶段前创建checkpoint
git tag -a checkpoint-before-phase2 -m "Before v0.13.0 upgrade"

# 如果需要回滚
git reset --hard checkpoint-before-phase2

# 或保留分支
git branch gfx906/rollback-phase2 HEAD
```

---

## 📊 成功指标

### 定量指标

```python
success_metrics = {
    "test_pass_rate": "≥ 90%",
    "performance_improvement": "≥ 10% (相比v0.11.1)",
    "memory_leak_fixed": "LMCache测试通过",
    "rocm_tests": "所有ROCm测试通过",
}
```

### 定性指标

- [ ] 团队对升级结果满意
- [ ] 生产环境稳定运行2周
- [ ] 用户反馈积极
- [ ] 文档完善，便于后续维护

---

## 📚 相关文档

### 已完成文档

1. ✅ `MAINTENANCE_LOG.md` - 项目维护日志
2. ✅ `USER_GUIDE.md` - 用户使用指南
3. ✅ `HARDWARE_VERIFICATION_CHECKLIST.md` - 硬件验证清单
4. ✅ `UPGRADE_EVALUATION_v0.11.1_to_v0.17.0.md` - 原始评估报告
5. ✅ `ACTION_PLAN.md` - 行动计划
6. ✅ `FINAL_REPORT_PHASE1.md` - Phase 1完成报告
7. ✅ `PHASE7_PERFORMANCE_ANALYSIS.md` - 性能分析

### 执行阶段需要参考

- `docs-collection/20-gfx906-migration/` - 迁移文档集合
- `06-migration-strategy.md` - 详细迁移策略
- `08-conflict-resolution.md` - 冲突解决指南
- `07-testing-strategy.md` - 测试策略
- `11-maintenance-checklist.md` - 维护检查清单

---

## 🚀 下一步行动

### 本周（3月8日-3月15日）

1. **周一**: 与团队讨论，确认方案A
2. **周二**: 创建测试分支，分析v0.13.0
3. **周三-周四**: 评估关键commits，识别风险
4. **周五**: 制定详细Phase 2计划

### 下周（3月16日-3月22日）

1. **周一-周二**: 执行Phase 2升级
2. **周三-周四**: 测试和验证
3. **周五**: 评估和决策（继续或停止）

### 持续行动

- 每周进度同步
- 每个阶段后评估
- 持续更新MAINTENANCE_LOG.md

---

**文档创建时间**: 2026-03-08
**最后更新**: 2026-03-08
**维护者**: gfx906适配团队
**状态**: 等待团队决策

---

## 附录：快速参考

### Git命令速查

```bash
# 查看版本差异
cd D:/VLLM/vllm
git log --oneline v0.11.1..v0.13.0 | wc -l
git diff v0.11.1..v0.13.0 --stat

# 查找ROCm相关commits
git log --oneline --grep="ROCm\|rocm\|AMD\|amd" v0.11.1..v0.13.0

# Cherry-pick特定commit
cd D:/VLLM/vllm-gfx906
git cherry-pick <commit-hash>

# 解决冲突后继续
git add <resolved-files>
git cherry-pick --continue

# 创建checkpoint
git tag -a checkpoint-<phase> -m "Checkpoint before <phase>"

# 回滚到checkpoint
git reset --hard checkpoint-<phase>
```

### 关键文件位置

```bash
# 保护文件（必须保留）
requirements/rocm.txt
vllm/platforms/rocm.py
vllm/transformers_utils/config.py
vllm/config/model.py

# Phase 1修改的文件
vllm/v1/attention/backends/mla/rocm_aiter_mla.py

# 测试脚本
test_rocm_mla_fix.py
benchmark_performance.py

# 文档
MAINTENANCE_LOG.md  # 主维护文档
USER_GUIDE.md  # 用户指南
```

### 联系方式

- **项目维护**: gfx906适配团队
- **技术问题**: 提交GitHub issue
- **紧急联系**: [待填写]
