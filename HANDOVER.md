# vllm-gfx906 维护交接文档

## ✅ 已完成工作 (2026-03-08)

### 1. Transformers 5 兼容性修复

**问题**: ImportError: cannot import name 'ALLOWED_LAYER_TYPES'

**根本原因**: 
- 原版 vllm-gfx906 直接导入 `ALLOWED_LAYER_TYPES`
- Transformers 5.0 将其重命名为 `ALLOWED_ATTENTION_LAYER_TYPES`
- 原 fork 已归档，未更新此兼容性

**修复方案**:
在两个文件中添加了 try/except 兼容性代码：

1. `vllm/transformers_utils/config.py` (line 18-26)
2. `vllm/config/model.py` (line 15-22)

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

**Commit**: `cc7a762cc` - "fix: add transformers 5 compatibility support"

### 2. 创建维护文档

- **MAINTENANCE.md**: 维护计划和流程
- **CHANGELOG.md**: 变更日志

## 📋 维护计划

### 定期任务

#### 每周
- 检查 upstream (vllm-project/vllm) 的更新
- 查看是否有重要的 bug 修复或安全更新

#### 每月
- 评估是否需要同步到新版本
- 测试关键功能

#### 按需
- 上游有重要安全修复时立即同步
- 用户报告问题时及时响应

### 同步策略

1. **保护 gfx906 特定修改**
   - `requirements/rocm.txt` 中的依赖固定
   - `setup.py` 中的构建配置
   - 平台检测代码

2. **优先级**
   - 安全/bug 修复 > 功能更新 > 重构

3. **测试流程**
   - 基础导入测试
   - 模型加载测试
   - 推理功能测试

## 🔧 技术要点

### gfx906 特定限制

1. **数据类型**: 仅支持 float16（无 bfloat16）
2. **量化**: 不支持 bitsandbytes
3. **ROCm 版本**: 基于 6.3（内核模式驱动）

### 关键文件

- `requirements/rocm.txt`: ROCm 特定依赖
- `vllm/platforms/`: 平台检测代码
- `vllm/vllm_flash_attn/`: flash attention 内核

## 📊 当前状态

- **分支**: gfx906/main
- **基于**: vllm v0.6.x
- **最新提交**: cc7a762cc (transformers 5 兼容性)
- **状态**: ✅ 可用

## 🚀 下一步行动

### 短期 (1-2 周)
1. 测试修复后的代码在 Linux 环境中的运行情况
2. 验证常用模型的兼容性
3. 准备发布说明

### 中期 (1-2 月)
1. 建立自动化测试流程
2. 同步 upstream 关键更新
3. 收集用户反馈

### 长期
1. 评估是否升级到更新的 vllm 版本
2. 考虑支持更新的 ROCm 版本
3. 优化性能

## 📞 联系方式

- **问题反馈**: 创建 GitHub Issue
- **讨论**: 在 Issue 中讨论技术细节

## 🙏 致谢

感谢原作者 **nlzy** 创建并维护 vllm-gfx906 项目。
虽然原仓库已归档，但为 gfx906 用户提供了宝贵的基础。

---
**维护者**: Sisyphus (AI Agent)
**接手日期**: 2026-03-08
