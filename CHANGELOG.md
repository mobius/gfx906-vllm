# 更新日志

所有对 vllm-gfx906 的重大更改都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
本项目遵循**语义化版本**。

## [未发布]

### 已修复 (Fixed)
- 修复 transformers 5.0 兼容性问题 (#1)
  - 在 `vllm/transformers_utils/config.py` 添加 try/except 处理
  - 在 `vllm/config/model.py` 添加 try/except 处理
  - `ALLOWED_LAYER_TYPES` 在 transformers 5 中重命名为 `ALLOWED_ATTENTION_LAYER_TYPES`

## [0.6.0] - 2026-01-XX

### 变更 (Changed)
- 从 nlzy/vllm-gfx906 接手维护
- 原仓库已于 2026-02-20 归档

### 已知问题 (Known Issues)
- 需要定期同步上游 vllm 项目更新

## 历史版本

历史版本请参考原仓库：https://github.com/nlzy/vllm-gfx906
