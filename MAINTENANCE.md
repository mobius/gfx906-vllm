# vllm-gfx906 维护计划

## 项目概述

这是基于 vllm-project/vllm 修改以支持 AMD gfx906 GPU (Radeon VII / MI50 / MI60) 的 fork 版本。

**状态**: 活跃维护（接手自 nlzy/vllm-gfx906）

**上游项目**: https://github.com/vllm-project/vllm

## 已知问题与修复

### 2026-03-08: Transformers 5 兼容性修复

**问题**: `ImportError: cannot import name 'ALLOWED_LAYER_TYPES'`

**原因**: transformers 5.0 将 `ALLOWED_LAYER_TYPES` 重命名为 `ALLOWED_ATTENTION_LAYER_TYPES`

**修复文件**:
- `vllm/transformers_utils/config.py` (line 18)
- `vllm/config/model.py` (line 14)

**修复方案**:
```python
try:
    # Transformers v5
    from transformers.configuration_utils import ALLOWED_ATTENTION_LAYER_TYPES
except ImportError:
    # Transformers v4
    from transformers.configuration_utils import (
        ALLOWED_LAYER_TYPES as ALLOWED_ATTENTION_LAYER_TYPES,
    )
```

## 定期同步计划

### 同步频率

- **关键安全修复**: 立即同步
- **主要版本更新**: 每 2 周检查一次
- **依赖更新**: 每月检查一次

### 同步流程

1. **检查上游更新**
   ```bash
   cd D:\VLLM\vllm-gfx906
   git fetch upstream main
   git log HEAD..upstream/main --oneline
   ```

2. **分析变更**
   - 查看 requirements/common.txt 中的依赖变化
   - 检查是否有影响 ROCm/gfx906 的修改
   - 关注 transformers 版本更新

3. **测试兼容性**
   - 在本地环境测试关键功能
   - 确认 ROCm 兼容性
   - 验证模型加载

4. **合并变更**
   ```bash
   git merge upstream/main
   # 解决冲突，保留 gfx906 特定修改
   ```

5. **提交记录**
   - 在 CHANGELOG.md 记录同步内容
   - 标注任何 gfx906 特定的修改

## gfx906 特定修改列表

### requirements/rocm.txt

**用途**: ROCm 平台特定依赖

**关键修改**:
- 移除了 gRPC 相关依赖（gfx906 兼容性问题）
- 固定 numba == 0.61.2
- 使用特定的 fastsafetensors 版本
- 移除 amd-quark（需要更新的 ROCm）

### 构建系统

**setup.py**: 可能包含 gfx906 特定的编译选项

### 内核修改

可能在以下目录：
- `vllm/vllm_flash_attn/`
- CMake/编译配置

## 依赖管理

### 当前关键依赖

- **transformers**: >= 4.56.0, < 5 (已添加 v5 兼容性)
- **torch**: == 2.9.0
- **ROCM**: 6.3+ (仅内核模式驱动)

### 注意事项

1. **transformers 版本**
   - 官方限制 < 5
   - 代码已支持 v5（通过 try/except）
   - 可以考虑更新限制为 < 6

2. **ROCm 版本**
   - 当前基于 ROCm 6.3
   - 注意 ROCm 更新可能破坏兼容性

## 测试清单

每次同步后需要测试：

### 基础功能
- [ ] 模型加载成功
- [ ] 推理执行正常
- [ ] 多 GPU 支持

### 模型兼容性
- [ ] Qwen 系列模型
- [ ] LLaMA 系列模型
- [ ] 其他常用模型

### 性能
- [ ] 吞吐量测试
- [ ] 内存使用正常

## 贡献指南

### 提交规范

使用语义化提交信息：
- `fix: 修复 xxx 问题`
- `sync: 同步上游到版本 x.x.x`
- `gfx906: 添加 gfx906 特定修改 xxx`

### 分支策略

- `main`: 稳定版本
- `dev`: 开发分支
- `sync/upstream`: 上游同步分支

## 相关资源

- **上游项目**: https://github.com/vllm-project/vllm
- **gfx906 硬件**: AMD Radeon VII / MI50 / MI60
- **ROCm 文档**: https://rocm.docs.amd.com/
- **原 fork**: https://github.com/nlzy/vllm-gfx906 (已归档)

## 版本历史

- **v0.6.x** (2026-03-08): 修复 transformers 5 兼容性
- **v0.6.x** (基于原 fork): 从 nlzy/vllm-gfx906 接手维护
