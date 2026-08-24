"""
IndSolve — Verification Lab Command-Line Runner
Executes all mathematical test suites and exits with code 0 on complete pass, 1 on failure.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from solver.test_suite import run_full_verification_lab


def main():
    out = run_full_verification_lab()
    total_tests = out["total_tests"]
    total_passed = out["total_passed"]
    all_passed = (total_passed == total_tests)

    print("=" * 75)
    print(f"[*] INDSOLVE VERIFICATION LAB: {total_passed} / {total_tests} TESTS PASSED")
    print("=" * 75)
    for fam, s in out["family_stats"].items():
        status_flag = "PASS" if s["passed"] == s["total"] else "FAIL"
        print(f"  [{s['passed']:>2}/{s['total']:<2}] [{status_flag}] {fam:<25}: {s['desc']}")
    print("=" * 75)

    if all_passed:
        print("[SUCCESS] 100% of verification tests passed. Solver is mathematically sound.")
        sys.exit(0)
    else:
        print(f"[FAILURE] {total_tests - total_passed} test(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
