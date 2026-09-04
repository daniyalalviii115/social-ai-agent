import os
import sys
import ast
import traceback

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_PASS = 0
_FAIL = 0

def _check(condition: bool, test_name: str, detail: str = "") -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  [PASS] {test_name}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {test_name}" + (f" — {detail}" if detail else ""))

print()
print("=" * 65)
print("  PHASE 6 VALIDATION: Static Analysis & Imports")
print("=" * 65)

# 1. AST syntax check
app_path = os.path.join(_PROJECT_ROOT, "dashboard", "app.py")
try:
    with open(app_path, "r", encoding="utf-8") as f:
        source = f.read()
    ast.parse(source)
    _check(True, "dashboard/app.py is syntactically valid Python")
except SyntaxError as e:
    _check(False, "dashboard/app.py is syntactically valid Python", detail=str(e))
except Exception as e:
    _check(False, "Failed to read dashboard/app.py", detail=str(e))

# 2. Dependency imports check
import_modules = [
    "config", 
    "core.simulator", 
    "ml.naive_bayes", 
    "ml.kmeans_cluster", 
    "decision.fuzzy_engine", 
    "decision.genetic_optimizer", 
    "content.llm_generator", 
    "content.banner_renderer"
]

for mod in import_modules:
    try:
        __import__(mod)
        _check(True, f"Successfully imported '{mod}'")
    except ImportError as e:
        _check(False, f"Successfully imported '{mod}'", detail=str(e))
    except Exception as e:
        _check(False, f"Error importing '{mod}'", detail=str(e))

# 3. CSV requirements check
try:
    import config
    csv_files = ["posts.csv", "engagement.csv", "engagement_tiered.csv", "comments.csv", "fuzzy_results.csv"]
    for csv_file in csv_files:
        path = os.path.join(config.DATA_DIR, csv_file)
        _check(os.path.exists(path), f"Required CSV exists: {csv_file}")
except Exception as e:
    _check(False, "Failed to check CSV files", detail=str(e))

# 4. PNG requirements check
try:
    import config
    png_files = ["cluster_scatter.png", "fuzzy_membership.png", "ga_convergence.png"]
    for png_file in png_files:
        path = os.path.join(config.OUTPUT_DIR, png_file)
        _check(os.path.exists(path), f"Required PNG exists: {png_file}")
except Exception as e:
    _check(False, "Failed to check PNG files", detail=str(e))

print()
print("=" * 65)
total = _PASS + _FAIL
print(f"  Tests passed : {_PASS} / {total}")
print(f"  Tests failed : {_FAIL} / {total}")
print("=" * 65)

if _FAIL == 0:
    print()
    print("  ALL PHASE 6 STATIC VALIDATION TESTS PASSED")
    print()
else:
    print()
    print(f"  {_FAIL} TEST(S) FAILED — review output above for details.")
    sys.exit(1)
