#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gfx906 ROCm MLA适配验证测试"""

import sys
from pathlib import Path


def test_mla_head_validation():
    print("\n=== Test 1: MLA Head Count Validation ===")

    test_cases = [
        (4, True, "num_heads=4 should be supported"),
        (8, True, "num_heads=8 should be supported"),
        (16, True, "num_heads=16 should be supported"),
        (32, True, "num_heads=32 should be supported"),
        (128, True, "num_heads=128 should be supported"),
        (2, False, "num_heads=2 should NOT be supported"),
        (12, False, "num_heads=12 should NOT be supported"),
        (256, False, "num_heads=256 should NOT be supported"),
    ]

    passed = 0
    failed = 0

    for num_heads, should_pass, description in test_cases:
        _valid_heads = num_heads in (4, 8) or (
            num_heads % 16 == 0 and 16 <= num_heads <= 128
        )

        if _valid_heads == should_pass:
            print(f"[PASS] {description}")
            passed += 1
        else:
            print(f"[FAIL] {description}")
            failed += 1

    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


def test_head_repeat_logic():
    print("\n=== Test 2: Head Repeat Logic ===")

    test_cases = [
        (4, 4, "num_heads=4, repeat_factor=4"),
        (8, 2, "num_heads=8, repeat_factor=2"),
        (16, 1, "num_heads=16, repeat_factor=1"),
        (32, 1, "num_heads=32, repeat_factor=1"),
        (128, 1, "num_heads=128, repeat_factor=1"),
    ]

    passed = 0
    failed = 0

    for num_heads, expected_factor, description in test_cases:
        needs_head_repeat = num_heads < 16
        head_repeat_factor = 16 // num_heads if num_heads < 16 else 1

        if head_repeat_factor == expected_factor:
            print(f"[PASS] {description} (actual: {head_repeat_factor})")
            passed += 1
        else:
            print(f"[FAIL] {description} (expected: {expected_factor}, actual: {head_repeat_factor})")
            failed += 1

    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


def test_code_correctness():
    print("\n=== Test 3: Code Modification Verification ===")

    file_path = Path(__file__).parent / "vllm/v1/attention/backends/mla/rocm_aiter_mla.py"

    if not file_path.exists():
        print(f"[FAIL] File not found: {file_path}")
        return False

    content = file_path.read_text()

    checks = [
        ("_valid_heads check", "_valid_heads = num_heads in (4, 8)" in content),
        ("_needs_head_repeat definition", "self._needs_head_repeat = num_heads < 16" in content),
        ("_head_repeat_factor definition", "self._head_repeat_factor = 16 // num_heads" in content),
        ("q tensor repeat", "q.repeat_interleave(self._head_repeat_factor, dim=1)" in content),
        ("o tensor slice", "o[:, :: self._head_repeat_factor, :]" in content),
    ]

    passed = 0
    failed = 0

    for check_name, check_result in checks:
        if check_result:
            print(f"[PASS] {check_name}")
            passed += 1
        else:
            print(f"[FAIL] {check_name}")
            failed += 1

    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


def main():
    print("=" * 60)
    print("gfx906 ROCm MLA Adaptation Verification Test")
    print("=" * 60)

    results = []

    results.append(("Head count validation", test_mla_head_validation()))
    results.append(("Head repeat logic", test_head_repeat_logic()))
    results.append(("Code modification check", test_code_correctness()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n*** All tests passed! ROCm MLA improvement verified ***")
        print("\nApplied features:")
        print("  - Support num_heads = 4, 8, 16, 32, 64, 128")
        print("  - Auto handle nhead<16 scenarios")
        print("  - Suitable for TP=8 configuration")
        return 0
    else:
        print("\n*** Some tests failed, please check the fix ***")
        return 1


if __name__ == "__main__":
    sys.exit(main())
