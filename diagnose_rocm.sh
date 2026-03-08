#!/bin/bash
# 诊断脚本 - 检查 setup.py 的 ROCm 检测逻辑

echo "=== 诊断 vllm-gfx906 ROCm 安装问题 ==="
echo

echo "1. 检查当前的 VLLM_TARGET_DEVICE:"
echo "   VLLM_TARGET_DEVICE=$VLLM_TARGET_DEVICE"
echo

echo "2. 检查 PyTorch 版本信息:"
python3 -c "
import torch
print(f'   torch.version.cuda: {torch.version.cuda}')
print(f'   torch.version.hip: {torch.version.hip}')
"
echo

echo "3. 检查 setup.py 中的关键行:"
echo "   第 887 行 (应该显示 'elif _is_cuda() and not _is_hip():'):"
grep -n "elif _is_cuda()" setup.py | head -2
echo

echo "4. 检查函数定义:"
echo "   _is_cuda() 定义:"
grep -A 2 "def _is_cuda()" setup.py
echo

echo "   _is_hip() 定义:"
grep -A 3 "def _is_hip()" setup.py
echo

echo "=== 测试逻辑 ==="
python3 -c "
import torch

# 模拟 setup.py 中的逻辑
VLLM_TARGET_DEVICE = None
if torch.version.hip is not None:
    VLLM_TARGET_DEVICE = 'rocm'
elif torch.version.cuda is not None:
    VLLM_TARGET_DEVICE = 'cuda'
else:
    VLLM_TARGET_DEVICE = 'cpu'

print(f'自动检测的 VLLM_TARGET_DEVICE: {VLLM_TARGET_DEVICE}')

def _is_cuda():
    has_cuda = torch.version.cuda is not None
    return VLLM_TARGET_DEVICE == 'cuda' and has_cuda

def _is_hip():
    return (VLLM_TARGET_DEVICE == 'cuda' or VLLM_TARGET_DEVICE == 'rocm') and torch.version.hip is not None

print(f'_is_cuda(): {_is_cuda()}')
print(f'_is_hip(): {_is_hip()}')
print(f'_is_cuda() and not _is_hip(): {_is_cuda() and not _is_hip()}')
"
echo

echo "=== 建议 ==="
echo "如果 _is_cuda() 和 _is_hip() 都为 True，需要检查 VLLM_TARGET_DEVICE 的值"
echo "如果 setup.py 第 887 行显示不正确，需要重新拉取代码"
