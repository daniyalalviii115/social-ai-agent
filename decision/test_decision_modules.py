"""
decision/test_decision_modules.py — Phase 4 Validation Test Suite
==================================================================
Validates the correctness and integration of:
  - decision/fuzzy_engine.py  (StrategyFuzzyEngine)
  - decision/genetic_optimizer.py (ContentGeneticOptimizer)

Run with:
    .\\venv\\Scripts\\python.exe decision\\test_decision_modules.py
"""

import os
import sys
import traceback

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
import pandas as pd  # noqa: E402

from decision.fuzzy_engine import StrategyFuzzyEngine  # noqa: E402
from decision.genetic_optimizer import ContentGeneticOptimizer  # noqa: E402
from core.simulator import (  # noqa: E402
    HOOK_TYPES,
    POSTING_HOURS,
    VISUAL_STYLES,
    CONTENT_TONES,
)

_PASS = 0
_FAIL = 0


def _check(condition: bool, test_name: str, detail: str = "") -> None:
    """Print PASS / FAIL for a single assertion."""
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  [PASS] {test_name}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {test_name}" + (f" — {detail}" if detail else ""))


# ===========================================================================
# SECTION 1: StrategyFuzzyEngine — compute_shift_rate correctness
# ===========================================================================

print()
print("=" * 65)
print("  SECTION 1: StrategyFuzzyEngine Tests")
print("=" * 65)

try:
    engine = StrategyFuzzyEngine()

    # Test 1a — negative sentiment + Flop tier (should produce HIGH shift rate)
    result_neg_flop = engine.compute_shift_rate(-0.9, "Flop")
    _check(
        isinstance(result_neg_flop, dict),
        "compute_shift_rate returns dict",
    )
    _check(
        "shift_rate" in result_neg_flop,
        "result contains 'shift_rate' key",
    )
    _check(
        "trigger_pivot" in result_neg_flop,
        "result contains 'trigger_pivot' key",
    )
    _check(
        "sentiment_input" in result_neg_flop and "tier_input" in result_neg_flop,
        "result contains 'sentiment_input' and 'tier_input' keys",
    )

    sr_neg_flop = result_neg_flop["shift_rate"]
    _check(
        0.0 < sr_neg_flop < 1.0,
        f"negative/Flop shift_rate strictly between 0 and 1 (got {sr_neg_flop:.4f})",
    )

    # Test 1b — neutral sentiment + Average tier (mid range expected)
    result_neu_avg = engine.compute_shift_rate(0.0, "Average")
    sr_neu_avg = result_neu_avg["shift_rate"]
    _check(
        0.0 < sr_neu_avg < 1.0,
        f"neutral/Average shift_rate strictly between 0 and 1 (got {sr_neu_avg:.4f})",
    )

    # Test 1c — positive sentiment + Viral tier (should produce LOW shift rate)
    result_pos_viral = engine.compute_shift_rate(0.9, "Viral")
    sr_pos_viral = result_pos_viral["shift_rate"]
    _check(
        0.0 < sr_pos_viral < 1.0,
        f"positive/Viral shift_rate strictly between 0 and 1 (got {sr_pos_viral:.4f})",
    )

    # Sanity check: negative/Flop must have a HIGHER shift_rate than positive/Viral
    _check(
        sr_neg_flop > sr_pos_viral,
        f"Fuzzy logic direction: neg/Flop ({sr_neg_flop:.4f}) > pos/Viral ({sr_pos_viral:.4f})",
        detail=f"neg_flop={sr_neg_flop:.4f}, pos_viral={sr_pos_viral:.4f}",
    )

    # Check trigger_pivot type
    _check(
        isinstance(result_neg_flop["trigger_pivot"], bool),
        "trigger_pivot is bool type",
    )

    # Check tier_input mapping
    _check(
        result_neg_flop["tier_input"] == 0.0,
        "Flop tier maps to ordinal 0.0",
    )
    result_viral = engine.compute_shift_rate(0.5, "Viral")
    _check(
        result_viral["tier_input"] == 2.0,
        "Viral tier maps to ordinal 2.0",
    )

except Exception:
    _FAIL += 1
    print(f"  [FAIL] StrategyFuzzyEngine raised an exception:")
    traceback.print_exc()


# ===========================================================================
# SECTION 2: fuzzy_results.csv existence and schema
# ===========================================================================

print()
print("=" * 65)
print("  SECTION 2: fuzzy_results.csv Validation")
print("=" * 65)

try:
    fuzzy_csv_path = os.path.join(config.DATA_DIR, "fuzzy_results.csv")

    _check(
        os.path.exists(fuzzy_csv_path),
        f"fuzzy_results.csv exists at {fuzzy_csv_path}",
    )

    if os.path.exists(fuzzy_csv_path):
        fuzzy_df = pd.read_csv(fuzzy_csv_path)

        required_cols = {"post_id", "avg_sentiment_score", "performance_tier",
                         "shift_rate", "trigger_pivot"}
        _check(
            required_cols.issubset(set(fuzzy_df.columns)),
            f"fuzzy_results.csv has all required columns: {required_cols}",
            detail=f"actual columns: {list(fuzzy_df.columns)}",
        )

        # trigger_pivot must only contain boolean values (True/False)
        if "trigger_pivot" in fuzzy_df.columns:
            unique_vals = set(fuzzy_df["trigger_pivot"].unique())
            _check(
                unique_vals.issubset({True, False}),
                f"trigger_pivot contains only boolean values (found: {unique_vals})",
            )

        # shift_rate must be in (0, 1) for all rows
        if "shift_rate" in fuzzy_df.columns:
            _check(
                (fuzzy_df["shift_rate"] > 0.0).all() and
                (fuzzy_df["shift_rate"] < 1.0).all(),
                "All shift_rate values are strictly between 0 and 1",
                detail=f"min={fuzzy_df['shift_rate'].min():.4f}, "
                       f"max={fuzzy_df['shift_rate'].max():.4f}",
            )

        _check(
            len(fuzzy_df) > 0,
            f"fuzzy_results.csv has {len(fuzzy_df)} rows (non-empty)",
        )

except Exception:
    _FAIL += 1
    print(f"  [FAIL] fuzzy_results.csv validation raised exception:")
    traceback.print_exc()


# ===========================================================================
# SECTION 3: ContentGeneticOptimizer — logic and convergence
# ===========================================================================

print()
print("=" * 65)
print("  SECTION 3: ContentGeneticOptimizer Tests")
print("=" * 65)

try:
    # Instantiate with no fitness lookup (uses defaults) for a quick test
    optimizer = ContentGeneticOptimizer(fitness_lookup_df=None)

    # Override to fewer generations for speed in tests
    optimizer.generations    = 10
    optimizer.population_size = 15

    result = optimizer.evolve(verbose=False)

    _check(
        isinstance(result, dict),
        "evolve() returns dict",
    )

    required_keys = {"best_individual", "best_fitness", "fitness_history", "final_population"}
    _check(
        required_keys.issubset(set(result.keys())),
        f"evolve() result contains all required keys: {required_keys}",
    )

    # fitness_history length == number of generations run
    _check(
        len(result["fitness_history"]) == optimizer.generations,
        f"fitness_history length == generations ({optimizer.generations})",
        detail=f"got {len(result['fitness_history'])}",
    )

    # fitness_history must be non-decreasing (running-best property)
    fh = result["fitness_history"]
    is_nondecreasing = all(fh[i] <= fh[i + 1] or abs(fh[i] - fh[i + 1]) < 1e-12
                           for i in range(len(fh) - 1))
    _check(
        is_nondecreasing,
        "fitness_history is non-decreasing (running-best per generation)",
        detail=f"history: {[round(v, 6) for v in fh]}",
    )

    # best_individual contains all 4 required gene keys
    best = result["best_individual"]
    gene_keys = {"hook_type", "posting_hour", "visual_style", "content_tone"}
    _check(
        gene_keys.issubset(set(best.keys())),
        f"best_individual contains all 4 gene keys: {gene_keys}",
    )

    # Each gene value must be drawn from the correct source list
    _check(
        best.get("hook_type") in HOOK_TYPES,
        f"best_individual.hook_type in HOOK_TYPES (got: {best.get('hook_type')})",
    )
    _check(
        best.get("posting_hour") in POSTING_HOURS,
        f"best_individual.posting_hour in POSTING_HOURS (got: {best.get('posting_hour')})",
    )
    _check(
        best.get("visual_style") in VISUAL_STYLES,
        f"best_individual.visual_style in VISUAL_STYLES (got: {best.get('visual_style')})",
    )
    _check(
        best.get("content_tone") in CONTENT_TONES,
        f"best_individual.content_tone in CONTENT_TONES (got: {best.get('content_tone')})",
    )

    # best_fitness is a float
    _check(
        isinstance(result["best_fitness"], float),
        f"best_fitness is float (got: {type(result['best_fitness']).__name__})",
    )

    # final_population is the correct size
    _check(
        len(result["final_population"]) == optimizer.population_size,
        f"final_population has correct size {optimizer.population_size}",
        detail=f"got {len(result['final_population'])}",
    )

except Exception:
    _FAIL += 1
    print(f"  [FAIL] ContentGeneticOptimizer raised an exception:")
    traceback.print_exc()


# ===========================================================================
# SECTION 4: Output files exist
# ===========================================================================

print()
print("=" * 65)
print("  SECTION 4: Output Files Existence")
print("=" * 65)

try:
    mf_path = os.path.join(config.OUTPUT_DIR, "fuzzy_membership.png")
    ga_path = os.path.join(config.OUTPUT_DIR, "ga_convergence.png")

    _check(
        os.path.exists(mf_path),
        f"fuzzy_membership.png exists at {mf_path}",
    )
    _check(
        os.path.exists(ga_path),
        f"ga_convergence.png exists at {ga_path}",
    )

    # Check non-zero file size
    if os.path.exists(mf_path):
        _check(
            os.path.getsize(mf_path) > 0,
            f"fuzzy_membership.png is non-empty ({os.path.getsize(mf_path)} bytes)",
        )
    if os.path.exists(ga_path):
        _check(
            os.path.getsize(ga_path) > 0,
            f"ga_convergence.png is non-empty ({os.path.getsize(ga_path)} bytes)",
        )

except Exception:
    _FAIL += 1
    print(f"  [FAIL] Output file check raised exception:")
    traceback.print_exc()


# ===========================================================================
# Final summary
# ===========================================================================

print()
print("=" * 65)
total = _PASS + _FAIL
print(f"  Tests passed : {_PASS} / {total}")
print(f"  Tests failed : {_FAIL} / {total}")
print("=" * 65)

if _FAIL == 0:
    print()
    print("  ALL PHASE 4 VALIDATION TESTS PASSED")
    print()
else:
    print()
    print(f"  {_FAIL} TEST(S) FAILED — review output above for details.")
    print()
    sys.exit(1)
