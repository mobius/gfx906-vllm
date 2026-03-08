#!/usr/bin/env python3
"""验证P0级别修复的正确性"""

import re
import sys
from pathlib import Path


def check_requirements():
    """检查requirements/common.txt中的关键更新"""
    print("Checking dependency updates...")
    
    req_file = Path("requirements/common.txt")
    if not req_file.exists():
        print("[X] requirements/common.txt not found")
        return False
    
    content = req_file.read_text()
    checks = []
    
    # 1. protobuf (CVE-2026-0994)
    if re.search(r'protobuf\s*>=\s*5\.29\.6.*CVE-2026-0994', content):
        print("[OK] protobuf security update applied (CVE-2026-0994)")
        checks.append(True)
    else:
        print("[X] protobuf security update not found")
        checks.append(False)
    
    # 2. ijson
    if re.search(r'ijson.*Required for mistral streaming tool parser', content):
        print("[OK] ijson dependency added")
        checks.append(True)
    else:
        print("[X] ijson dependency not found")
        checks.append(False)
    
    # 3. compressed-tensors
    if re.search(r'compressed-tensors\s*==\s*0\.13\.0', content):
        print("[OK] compressed-tensors updated to 0.13.0")
        checks.append(True)
    else:
        print("[!] compressed-tensors not 0.13.0")
        checks.append(False)
    
    # 4. mistral_common
    if re.search(r'mistral_common\[image\]\s*>=\s*1\.9\.1', content):
        print("[OK] mistral_common updated to 1.9.1")
        checks.append(True)
    else:
        print("[!] mistral_common not 1.9.1")
        checks.append(False)
    
    # 5. xgrammar
    if re.search(r'xgrammar\s*==\s*0\.1\.29', content):
        print("[OK] xgrammar updated to 0.1.29")
        checks.append(True)
    else:
        print("[!] xgrammar not 0.1.29")
        checks.append(False)
    
    # 6. opencv
    if re.search(r'opencv-python-headless\s*>=\s*4\.13\.0', content):
        print("[OK] opencv-python-headless updated to 4.13.0")
        checks.append(True)
    else:
        print("[!] opencv-python-headless not 4.13.0")
        checks.append(False)
    
    # 7. aiohttp
    if re.search(r'aiohttp\s*>=\s*3\.13\.3', content):
        print("[OK] aiohttp updated to 3.13.3")
        checks.append(True)
    else:
        print("[!] aiohttp not 3.13.3")
        checks.append(False)
    
    return all(checks), checks


def check_openai_api_files():
    """检查OpenAI API服务文件"""
    print("\nChecking OpenAI API service files...")
    
    openai_dir = Path("vllm/entrypoints/openai")
    if not openai_dir.exists():
        print("[X] OpenAI API directory not found")
        return False
    
    required_files = [
        "serving_chat.py",
        "serving_completion.py",
        "serving_engine.py",
        "serving_models.py",
        "serving_responses.py",
        "serving_tokenization.py",
        "serving_tokens.py",
        "serving_transcription.py",
        "speech_to_text.py",
        "protocol.py",
    ]
    
    checks = []
    for file in required_files:
        file_path = openai_dir / file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"[OK] {file} ({size} bytes)")
            checks.append(True)
        else:
            print(f"[X] {file} missing")
            checks.append(False)
    
    return all(checks), checks


def main():
    print("="*60)
    print("P0 Fix Verification")
    print("="*60)
    
    req_ok, req_checks = check_requirements()
    api_ok, api_checks = check_openai_api_files()
    
    print("\n" + "="*60)
    print("Verification Results")
    print("="*60)
    
    passed = sum(req_checks) + sum(api_checks)
    total = len(req_checks) + len(api_checks)
    
    print(f"\nDependency checks: {sum(req_checks)}/{len(req_checks)} passed")
    print(f"API file checks: {sum(api_checks)}/{len(api_checks)} passed")
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if req_ok and api_ok:
        print("\n[SUCCESS] All P0 fixes verified!")
        return 0
    else:
        print("\n[WARNING] Some checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
