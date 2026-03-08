#!/usr/bin/env python3
"""
从本地upstream vLLM同步到vllm-gfx906

用法：
    python sync_from_upstream.py
"""

import os
import shutil
import filecmp
from pathlib import Path
from datetime import datetime

# 配置
UPSTREAM_VLLM = Path("D:/VLLM/vllm")
GFX906_VLLM = Path("D:/VLLM/vllm-gfx906")
SYNC_LOG = Path("D:/VLLM/vllm-gfx906/SYNC_LOG.md")

# gfx906特定文件（必须保留，不能覆盖）
GFX906_PROTECTED_FILES = {
    # requirements
    "requirements/rocm.txt",
    # 平台检测
    "vllm/platforms/rocm.py",
    "vllm/vllm_flash_attn/flash_attention.py",
    # Transformers 5兼容性修复
    "vllm/transformers_utils/config.py",
    "vllm/config/model.py",
    # 文档
    "HANDOVER.md",
    "MAINTENANCE.md",
    "CHANGELOG.md",
}

# 需要同步的关键文件（从upstream）
SYNC_PRIORITY_FILES = {
    # Python核心代码
    "vllm/worker/worker.py",
    "vllm/engine/",
    "vllm/model_executor/",
    # 修复相关
    "vllm/attention/",
    "vllm/layers/",
    # 配置
    "vllm/config/",
    # 工具
    "vllm/transformers_utils/",
    # 测试
    "tests/",
}

# 必须跳过的文件（upstream特定，不适用于gfx906）
SKIP_FILES = {
    # CUDA特定
    "**/*_cuda.py",
    "**/*_cuda.cu",
    "**/tensorrt/",
    # bitsandbytes（不支持ROCm）
    "**/bitsandbytes*",
    # 其他GPU支持
    "**/tpu/",
    "**/xpu/",
    "**/npu/",
}


def log_message(message):
    """记录日志到控制台和文件"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}\n"
    print(log_line.strip())

    with open(SYNC_LOG, "a", encoding="utf-8") as f:
        f.write(log_line)


def compare_files(upstream_file, gfx906_file):
    """比较两个文件的差异"""
    if not upstream_file.exists():
        return "upstream文件不存在"
    if not gfx906_file.exists():
        return "gfx906文件不存在（需要新增）"

    if filecmp.cmp(upstream_file, gfx906_file, shallow=False):
        return "相同"
    else:
        return "不同（需要同步）"


def should_sync_file(relative_path):
    """判断是否应该同步这个文件"""
    # 检查是否是保护文件
    for protected in GFX906_PROTECTED_FILES:
        if str(relative_path) in protected or str(relative_path).startswith(
            protected.rstrip("/")
        ):
            return False, "gfx906保护文件"

    # 检查是否应该跳过
    for skip_pattern in SKIP_FILES:
        if skip_pattern in str(relative_path):
            return False, f"匹配跳过模式: {skip_pattern}"

    return True, "可以同步"


def scan_differences():
    """扫描upstream和gfx906之间的差异"""
    log_message("=" * 80)
    log_message("开始扫描upstream和gfx906的差异")
    log_message("=" * 80)

    # 扫描upstream中的文件
    upstream_files = list(UPSTREAM_VLLM.rglob("*.py"))
    upstream_files.extend(UPSTREAM_VLLM.rglob("*.txt"))

    sync_candidates = []
    protected_files = []
    skip_files = []

    for upstream_file in upstream_files:
        # 计算相对路径
        try:
            relative_path = upstream_file.relative_to(UPSTREAM_VLLM)
        except ValueError:
            continue

        # 判断是否应该同步
        should_sync, reason = should_sync_file(relative_path)

        if not should_sync:
            if "保护文件" in reason:
                protected_files.append((relative_path, reason))
            else:
                skip_files.append((relative_path, reason))
            continue

        # 检查文件是否不同
        gfx906_file = GFX906_VLLM / relative_path
        comparison = compare_files(upstream_file, gfx906_file)

        if comparison == "不同（需要同步）":
            sync_candidates.append((relative_path, upstream_file, gfx906_file))
        elif comparison == "gfx906文件不存在（需要新增）":
            sync_candidates.append((relative_path, upstream_file, gfx906_file))

    # 输出报告
    log_message(f"扫描完成：")
    log_message(f"  候选同步文件: {len(sync_candidates)}")
    log_message(f"  保护文件（不覆盖）: {len(protected_files)}")
    log_message(f"  跳过文件: {len(skip_files)}")

    return sync_candidates, protected_files, skip_files


def sync_file(upstream_file, gfx906_file, dry_run=True):
    """同步单个文件"""
    if dry_run:
        log_message(f"[DRY RUN] 会同步: {gfx906_file}")
        return True

    try:
        # 确保目标目录存在
        gfx906_file.parent.mkdir(parents=True, exist_ok=True)

        # 备份现有文件
        if gfx906_file.exists():
            backup_file = gfx906_file.with_suffix(
                f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy2(gfx906_file, backup_file)
            log_message(f"  备份: {backup_file}")

        # 复制文件
        shutil.copy2(upstream_file, gfx906_file)
        log_message(f"  ✓ 同步成功: {gfx906_file}")
        return True

    except Exception as e:
        log_message(f"  ✗ 同步失败: {gfx906_file} - {e}")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("vLLM Upstream → gfx906 同步工具")
    print("=" * 80 + "\n")

    # 初始化日志
    log_message("同步工具启动")
    log_message(f"Upstream: {UPSTREAM_VLLM}")
    log_message(f"Gfx906: {GFX906_VLLM}")

    # 第一步：扫描差异
    print("\n第一步：扫描文件差异...")
    candidates, protected, skipped = scan_differences()

    # 第二步：显示关键文件
    print("\n第二步：关键文件差异:")
    print("=" * 80)

    # 按优先级排序
    priority_candidates = []
    for rel_path, upstream, gfx906 in candidates:
        for pattern in SYNC_PRIORITY_FILES:
            if pattern in str(rel_path):
                priority_candidates.append((rel_path, upstream, gfx906))
                break

    # 显示优先级高的前20个文件
    for i, (rel_path, upstream, gfx906) in enumerate(priority_candidates[:20], 1):
        print(f"\n{i}. {rel_path}")
        print(f"   upstream: {upstream}")
        print(f"   gfx906:   {gfx906}")

    # 第三步：询问是否继续
    print("\n" + "=" * 80)
    print(f"\n发现 {len(candidates)} 个需要同步的文件")
    print(f"其中 {len(priority_candidates)} 个为高优先级文件")

    response = input("\n是否继续？(yes/no/dry-run): ").strip().lower()

    if response == "dry-run":
        print("\n执行DRY RUN模式（不会实际修改文件）...")
        log_message("DRY RUN模式开始")

        success_count = 0
        fail_count = 0

        for rel_path, upstream, gfx906 in candidates:
            if sync_file(upstream, gfx906, dry_run=True):
                success_count += 1
            else:
                fail_count += 1

        log_message(f"DRY RUN完成: {success_count}个文件会同步, {fail_count}个文件失败")

    elif response == "yes":
        print("\n开始同步...")
        log_message("实际同步开始")

        success_count = 0
        fail_count = 0

        for i, (rel_path, upstream, gfx906) in enumerate(candidates, 1):
            print(f"\n[{i}/{len(candidates)}] 同步: {rel_path}")

            if sync_file(upstream, gfx906, dry_run=False):
                success_count += 1
            else:
                fail_count += 1

        log_message(f"同步完成: {success_count}成功, {fail_count}失败")
        print(f"\n同步完成: {success_count}成功, {fail_count}失败")

    else:
        print("\n取消同步")
        log_message("同步被用户取消")


if __name__ == "__main__":
    main()
