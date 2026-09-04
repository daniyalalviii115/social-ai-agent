"""
Master Test Suite Runner (Phase 7)
====================================
Runs every phase's individual validation script in sequence and
reports a single pass/fail summary across the entire project.

This does NOT reimplement the individual tests — it invokes the
exact test files already built and verified in Phases 2-6, and
aggregates their pass/fail status. If any phase's test script
raises an exception or exits non-zero, that phase is marked FAILED
and the runner continues to the next phase (does not stop early),
so you get a full picture of project health in one run.
"""

import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable

# (display name, relative path to test script)
TEST_SUITE = [
    ("Phase 2 — Simulator",            os.path.join("core", "test_simulator.py")),
    ("Phase 3 — ML Modules",           os.path.join("ml", "test_ml_modules.py")),
    ("Phase 4 — Decision Modules",     os.path.join("decision", "test_decision_modules.py")),
    ("Phase 5 — Content Modules",      os.path.join("content", "test_content_modules.py")),
    ("Phase 6 — Dashboard (static)",   os.path.join("dashboard", "test_dashboard_imports.py")),
    ("Phase 8/9 — Upgrade Modules",    os.path.join("core", "test_upgrade_modules.py")),
]


def run_test_file(display_name: str, relative_path: str) -> dict:
    """
    Runs a single test script as a subprocess, captures its output,
    and returns a result dict with pass/fail status and timing.
    """
    full_path = os.path.join(PROJECT_ROOT, relative_path)

    if not os.path.exists(full_path):
        return {
            "name": display_name,
            "path": relative_path,
            "passed": False,
            "duration": 0.0,
            "stdout": "",
            "stderr": f"Test file not found at {full_path}",
        }

    t0 = time.time()
    try:
        result = subprocess.run(
            [PYTHON_EXE, full_path],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute safety cap per test (GA tests can be slow)
        )
        duration = time.time() - t0
        passed = result.returncode == 0
        return {
            "name": display_name,
            "path": relative_path,
            "passed": passed,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": display_name,
            "path": relative_path,
            "passed": False,
            "duration": time.time() - t0,
            "stdout": exc.stdout or "",
            "stderr": f"Test timed out after 600 seconds.",
        }
    except Exception as exc:
        return {
            "name": display_name,
            "path": relative_path,
            "passed": False,
            "duration": time.time() - t0,
            "stdout": "",
            "stderr": f"Unexpected error running test: {exc}",
        }


def main():
    print("=" * 70)
    print("  MASTER TEST SUITE — Autonomous Social Media AI Agent")
    print("=" * 70)
    print(f"  Running {len(TEST_SUITE)} phase test suites...\n")

    results = []
    for display_name, relative_path in TEST_SUITE:
        print(f"  ▶ Running: {display_name}  ({relative_path})")
        result = run_test_file(display_name, relative_path)
        results.append(result)

        status = "✅ PASSED" if result["passed"] else "❌ FAILED"
        print(f"    {status}  ({result['duration']:.2f}s)")

        if not result["passed"]:
            # Show the tail of stdout/stderr to help diagnose without
            # flooding the console with the full log.
            if result["stdout"]:
                tail = result["stdout"].strip().splitlines()[-15:]
                print("    --- stdout (last 15 lines) ---")
                for line in tail:
                    print(f"    {line}")
            if result["stderr"]:
                tail = result["stderr"].strip().splitlines()[-15:]
                print("    --- stderr (last 15 lines) ---")
                for line in tail:
                    print(f"    {line}")
        print()

    # -----------------------------------------------------------
    # Summary
    # -----------------------------------------------------------
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = total - passed_count
    total_duration = sum(r["duration"] for r in results)

    print("=" * 70)
    print("  MASTER TEST SUITE SUMMARY")
    print("=" * 70)
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"    [{status}]  {r['name']:<35}  {r['duration']:>6.2f}s")
    print("-" * 70)
    print(f"  Total suites : {total}")
    print(f"  Passed       : {passed_count}")
    print(f"  Failed       : {failed_count}")
    print(f"  Total time   : {total_duration:.2f}s")
    print("=" * 70)

    if failed_count == 0:
        print("\n  🎉 ALL PHASE TEST SUITES PASSED — PROJECT IS FULLY VERIFIED\n")
        sys.exit(0)
    else:
        print(f"\n  ⚠️  {failed_count} SUITE(S) FAILED — SEE OUTPUT ABOVE FOR DETAILS\n")
        sys.exit(1)


if __name__ == "__main__":
    main()