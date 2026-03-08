#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理.bak备份文件
删除之前sync_tool.py创建的所有.bak文件
"""

import os
import sys
from pathlib import Path


def clean_backup_files(directory="."):
    """清理目录中的所有.bak文件"""

    path = Path(directory)
    if not path.exists():
        print(f"Error: Directory {directory} does not exist")
        return False

    # 查找所有.bak文件
    bak_files = list(path.rglob("*.bak"))

    if not bak_files:
        print("No .bak files found")
        return True

    print(f"Found {len(bak_files)} .bak files")
    print(
        f"Total size: {sum(f.stat().st_size for f in bak_files) / 1024 / 1024:.2f} MB"
    )

    # 确认删除
    print("\nDeleting .bak files...")
    deleted_count = 0
    failed_count = 0

    for bak_file in bak_files:
        try:
            bak_file.unlink()
            deleted_count += 1
            if deleted_count % 100 == 0:
                print(f"  Deleted {deleted_count}/{len(bak_files)} files...")
        except Exception as e:
            print(f"  Error deleting {bak_file}: {e}")
            failed_count += 1

    print(f"\nCleanup complete:")
    print(f"  Deleted: {deleted_count} files")
    print(f"  Failed: {failed_count} files")
    print(f"  Remaining: {len(list(path.rglob('*.bak')))} files")

    return failed_count == 0


def main():
    print("=" * 60)
    print("gfx906 Backup Files Cleanup")
    print("=" * 60)

    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    print(f"Working directory: {os.getcwd()}")

    # 执行清理
    success = clean_backup_files()

    print("\n" + "=" * 60)
    if success:
        print("✓ Cleanup completed successfully!")
        return 0
    else:
        print("✗ Cleanup completed with some errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
