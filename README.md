# vLLM for AMD gfx906

[![vLLM](https://img.shields.io/badge/vLLM-gfx906-red)](https://github.com/vllm-project/vllm)
[![ROCm](https://img.shields.io/badge/ROCm-6.3+-purple)](https://rocm.docs.amd.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](https://github.com/vllm-project/vllm/blob/main/LICENSE)

**专为 AMD gfx906 GPU（Radeon VII、Radeon Pro VII、Instinct MI50/MI60）优化的 LLM 推理引擎**

## 版权声明

本项目基于 [vLLM](https://github.com/vllm-project/vllm) 开发，感谢原项目所有贡献者。

- **原始项目**: [vLLM](https://github.com/vllm-project/vllm) by UC Berkeley Sky Computing Lab
- **许可证**: Apache License 2.0
- **本项目**: 包含针对AMD gfx906 GPU的优化和修复

详见 [NOTICE](NOTICE) 文件了解完整的归属声明。

## 项目状态

⚠️ **本项目由社区维护中。** 原作者已归档其工作，但本分支持续更新以保持与最新 vLLM 版本的兼容性。

## 最新更新

### ✅ 2026年3月更新

- **修复 HIPBLAS 兼容性问题** - 解决了 ROCm/gfx906 上的 `HIPBLAS_STATUS_INTERNAL_ERROR` 错误
- **更新至最新的 vLLM V1 引擎** - 性能提升和新功能支持
- **测试 Qwen3.5-0.8B** - 成功运行 Qwen/Qwen3.5-0.8B（非多模态版本）
- **通用 Triton GEMM** - 使用 Triton 矩阵乘法替代 hipBLAS 回退，提高稳定性

### 支持的模型

当前已测试并可用：
- ✅ Qwen/Qwen3.5-0.8B（非多模态）
- ✅ Qwen 系列模型（需使用 `--limit-mm-per-prompt '{"image": 0, "video": 0}'` 参数）

## 项目简介

这是 [vLLM](https://github.com/vllm-project/vllm) 的修改版本，专门用于 AMD gfx906 系列 GPU。它包含了针对 gfx906 架构的 ROCm 兼容性优化和变通方案。

**原作者**: [nalanzeyu](https://github.com/nalanzeyu/vllm-gfx906)

## 系统要求

- **硬件**: AMD gfx906 GPU（Radeon VII、Radeon Pro VII、Instinct MI50、Instinct MI60）
- **ROCm**: 6.3+（需要内核模式驱动）
- **Python**: 3.10+
- **Triton**: triton-gfx906 v3.5.0+gfx906（[安装指南](https://github.com/nlzy/triton-gfx906/tree/v3.5.0+gfx906)）
- **操作系统**: Linux（在 Ubuntu 上测试）

## 安装方式

### 从源码构建

```bash
# 安装系统依赖
sudo apt install python3-venv python3-dev

# 克隆仓库
git clone https://github.com/ttdxq/vllm-gfx906.git
cd vllm-gfx906

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装 ROCm 版本的 PyTorch
pip install torch==2.9 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

# 安装依赖
pip install -r requirements/rocm-build.txt -r requirements/rocm.txt

# 安装 vLLM
pip install --no-build-isolation -e .
```

## 使用方法

### 启动服务

```bash
# 对于多模态模型（禁用多模态功能）
VLLM_USE_MODELSCOPE=true vllm serve Qwen/Qwen3.5-0.8B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --enforce-eager

# 指定 GPU 内存使用率
vllm serve Qwen/Qwen3.5-0.8B \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85
```

### 测试 API

```bash
# 检查模型可用性
curl http://localhost:8000/v1/models

# 发送对话请求
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-0.8B",
    "messages": [{"role": "user", "content": "你好！请介绍一下你自己"}],
    "max_tokens": 100
  }'
```

### Python 客户端示例

```python
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-not-required"
)

# 发送对话请求
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-0.8B",
    messages=[
        {"role": "user", "content": "写一首关于编程的俳句"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
```

## 核心修改

### 1. ROCm GEMM 优化
- **问题**: `torch.nn.functional.linear` 在 gfx906 上触发 `HIPBLAS_STATUS_INTERNAL_ERROR`
- **解决方案**: 使用通用 Triton 矩阵乘法实现，避免 hipBLAS 调用
- **修改文件**: `vllm/model_executor/layers/utils.py`

### 2. CacheConfig 兼容性
- 修复混合模型的 `mamba_cache_mode` 属性问题
- 更新 Qwen3.5 模型加载逻辑

### 3. 平台检测
- 改进 gfx906 的 ROCm 平台检测
- 优化注意力后端选择机制

## 量化支持

基于原项目测试结果：
- ✅ **GPTQ** - 推荐
- ✅ **AWQ** - 推荐
- ✅ **W4A16 INT** - 支持（通过 llm-compressor）
- ⚠️ **MoE 量化模型** - 速度显著较慢，不推荐
- ⚠️ **非量化模型** - 略慢，但可用

详细信息请参阅 [Issue #29](https://github.com/nlzy/vllm-gfx906/issues/29)。

## 已知限制

1. **不支持多模态** - 必须使用 `--limit-mm-per-prompt` 禁用
2. **首次推理较慢** - Triton 内核首次运行时需要编译
3. **内存占用较大** - KV cache 预分配会占用大量 GPU 内存
4. **实验性质** - 使用风险自负

## 性能优化建议

1. **减小 `max-model-len`** - 对于小模型可以节省内存
2. **使用 `--gpu-memory-utilization`** - 显式管理内存使用
3. **启用 `--enforce-eager`** - 如果遇到 CUDA 图问题
4. **重启前清理进程** - 使用 `pkill -9 -f "vllm serve"` 杀死现有进程

## 故障排除

### 服务在首次请求时挂起
这是正常现象 - Triton 正在编译内核。请等待 1-2 分钟。

### GPU 内存错误
- 杀死现有进程: `pkill -9 -f "vllm serve"`
- 降低 GPU 内存使用率: `--gpu-memory-utilization 0.7`

### HIPBLAS 错误
最新版本应该已修复。如果遇到，请提交 issue。

## 贡献

欢迎贡献！你可以：
- 报告 bug
- 提出新功能建议
- 提交 pull request
- 改进文档

## 致谢

- **原始 vLLM**: [UC Berkeley Sky Computing Lab](https://sky.cs.berkeley.edu)
- **ROCm 移植**: [Said-Akbar/vllm-rocm](https://github.com/Said-Akbar/vllm-rocm)
- **gfx906 分支**: [nalanzeyu/vllm-gfx906](https://github.com/nalanzeyu/vllm-gfx906)
- **Triton for gfx906**: [nlzy/triton-gfx906](https://github.com/nlzy/triton-gfx906)

## 许可证

Apache License 2.0 - 详见 [LICENSE](LICENSE)

## 引用

如果你在研究中使用了本分支，请同时引用原始 vLLM 论文并说明 gfx906 适配：

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## 联系方式

- **问题反馈**: [GitHub Issues](https://github.com/ttdxq/vllm-gfx906/issues)
- **讨论交流**: [GitHub Discussions](https://github.com/ttdxq/vllm-gfx906/discussions)

---

**注意**: 本项目为社区维护的分支。使用风险自负，特别是作为硬件购买的参考依据。
