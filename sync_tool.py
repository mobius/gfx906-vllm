#!/usr/bin/env python3
"""
vLLM upstream到gfx906的同步工具

功能：扫描差异、选择性同步文件、保护gfx906特定修改
"""

import os
import shutil
import filecmp
from pathlib import Path
from datetime import datetime

# 配置路径
UPSTREAM_VLLM = Path("D:/VLLM/vllm")
GFX906_VLLM = Path("D:/VLLM/vllm-gfx906")
SYNC_LOG = Path("D:/VLLM/vllm-gfx906/SYNC_LOG.md")

# gfx906特定文件，保留不覆盖
GFX906_PROTECTED_FILES = {
    "requirements/rocm.txt",
    "vllm/platforms/rocm.py",
    "vllm/vllm_flash_attn/flash_attention.py",
    "vllm/transformers_utils/config.py",
    "vllm/config/model.py",
    "HANDOVER.md",
    "MAINTENANCE.md",
    "CHANGELOG.md",
}

# 跳过的文件模式
SKIP_PATTERNS = [
    "*_cuda.py",
    "*_cuda.cu",
    "tensorrt",
    "bitsandbytes",
    "tpu",
    "xpu",
    "npu",
]


def log_message(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {msg}\n"
    print(log_line.strip())
    with open(SYNC_LOG, "a", encoding="utf-8") as f:
        f.write(log_line)


def should_sync_file(rel_path):
    for protected in GFX906_PROTECTED_FILES:
        if str(rel_path) in protected:
            return False, "protected"
    for pattern in SKIP_PATTERNS:
        if pattern in str(rel_path):
            return False, f"skip:{pattern}"
    return True, "ok"


def scan_differences():
    log_message("开始扫描差异")

    candidates = []

    for upstream_file in UPSTREAM_VLLM.rglob("*"):
        if not upstream_file.is_file():
            continue

        try:
            rel_path = upstream_file.relative_to(UPSTREAM_VLLM)
        except ValueError:
            continue

        should_sync, reason = should_sync_file(rel_path)
        if not should_sync:
            continue

        gfx906_file = GFX906_VLLM / rel_path

        if not gfx906_file.exists():
            candidates.append((rel_path, upstream_file, gfx906_file, "new"))
        elif not filecmp.cmp(upstream_file, gfx906_file, shallow=False):
            candidates.append((rel_path, upstream_file, gfx906_file, "diff"))

    log_message(f"发现 {len(candidates)} 个需要同步的文件")
    return candidates


def sync_file(upstream_file, gfx906_file, dry_run=True):
    try:
        gfx906_file.parent.mkdir(parents=True, exist_ok=True)

        if gfx906_file.exists():
            backup = gfx906_file.with_suffix(".bak")
            shutil.copy2(gfx906_file, backup)
            if not dry_run:
                log_message(f"  备份: {backup}")

        if not dry_run:
            shutil.copy2(upstream_file, gfx906_file)
            log_message(f"  同步: {gfx906_file}")
        else:
            log_message(f"  [DRY RUN] 会同步: {gfx906_file}")

        return True
    except Exception as e:
        log_message(f"  失败: {gfx906_file} - {e}")
        return False


def main():
    print("\nvLLM同步工具")
    print(f"Upstream: {UPSTREAM_VLLM}")
    print(f"Gfx906: {GFX906_VLLM}\n")

    log_message("同步工具启动")

    candidates = scan_differences()

    if not candidates:
        print("没有需要同步的文件")
        return

    print(f"\n发现 {len(candidates)} 个需要同步的文件")
    print("\n前10个文件:")
    for i, (rel, up, gfx, status) in enumerate(candidates[:10], 1):
        print(f"{i}. {rel} [{status}]")

    response = input("\n选择: (d)ry-run/(y)es/(n)o: ").strip().lower()

    if response == "d":
        print("\nDRY RUN模式...")
        log_message("DRY RUN")
        for rel, up, gfx, status in candidates:
            sync_file(up, gfx, dry_run=True)
        print(f"\nDRY RUN完成: {len(candidates)} 个文件")

    elif response == "y":
        print("\n开始同步...")
        log_message("开始同步")

        success = 0
        failed = 0

        for i, (rel, up, gfx, status) in enumerate(candidates, 1):
            print(f"\n[{i}/{len(candidates)}] {rel}")
            if sync_file(up, gfx, dry_run=False):
                success += 1
            else:
                failed += 1

        log_message(f"同步完成: {success}成功, {failed}失败")
        print(f"\n完成: {success}成功, {failed}失败")

    else:
        print("\n取消")


if __name__ == "__main__":
    main()
