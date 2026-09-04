"""
ml/test_ml_modules.py — Phase 3 Automated Validation Tests
===========================================================
Validates both Phase 3 ML modules:
  - SentimentAnalyzer  (ml/naive_bayes.py)
  - PerformanceClusterer (ml/kmeans_cluster.py)

Asserts:
  1. NB accuracy > 0.50 on held-out test set.
  2. All three sentiment classes ("positive", "negative", "neutral") detected.
  3. Single inference result dict has the expected structure.
  4. score_post_sentiment() ratios sum to 1.0 (within floating-point tolerance).
  5. Tiered DataFrame contains only "Flop", "Average", "Viral" values.
  6. predict_tier() returns a valid tier string.
  7. Cluster engagement_rate ordering: Flop_mean < Average_mean < Viral_mean.
  8. data/engagement_tiered.csv exists on disk.
  9. outputs/cluster_scatter.png exists on disk.

Run from social-ai-agent/ with the venv activated:
    .\\venv\\Scripts\\python.exe ml\\test_ml_modules.py
"""

import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from ml.naive_bayes import SentimentAnalyzer, train_and_save_model
from ml.kmeans_cluster import PerformanceClusterer, cluster_and_save


def run_tests() -> None:
    """
    Execute all Phase 3 validation assertions.

    Prints a structured pass/fail summary and raises AssertionError with a
    descriptive message on the first failing check.
    """
    print()
    print("=" * 60)
    print("  PHASE 3 — ML MODULE VALIDATION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    comments_path   = os.path.join(config.DATA_DIR, "comments.csv")
    engagement_path = os.path.join(config.DATA_DIR, "engagement.csv")

    for p in (comments_path, engagement_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Missing: {p}\nRun Phase 2 first: python core/simulator.py"
            )

    comments_df   = pd.read_csv(comments_path)
    engagement_df = pd.read_csv(engagement_path)

    # ==================================================================
    # SECTION A — SentimentAnalyzer
    # ==================================================================
    print()
    print("  [A] SentimentAnalyzer (Naive Bayes)")
    print("  " + "-" * 40)

    analyzer = SentimentAnalyzer()
    results  = analyzer.train(comments_df)

    # A1 — accuracy > 0.50
    acc = results["accuracy"]
    assert acc > 0.50, (
        f"FAIL [A1]: Test accuracy {acc:.4f} is not > 0.50. "
        "The model performs worse than chance; check training data quality."
    )
    print(f"  [A1] Accuracy = {acc:.4f} > 0.50  [OK]")

    # A2 — all three sentiment classes detected
    expected_classes = {"positive", "negative", "neutral"}
    actual_classes   = set(analyzer.classes_)
    assert expected_classes == actual_classes, (
        f"FAIL [A2]: Expected classes {expected_classes}, "
        f"got {actual_classes}."
    )
    print(f"  [A2] All 3 classes detected: {sorted(actual_classes)}  [OK]")

    # A3 — single inference result structure
    sample_text = "This content is absolutely amazing!"
    pred = analyzer.predict_sentiment(sample_text)
    required_keys = {"text", "predicted_label", "confidence", "class_probabilities"}
    assert required_keys == set(pred.keys()), (
        f"FAIL [A3]: predict_sentiment() keys {set(pred.keys())} "
        f"do not match expected {required_keys}."
    )
    assert pred["predicted_label"] in expected_classes, (
        f"FAIL [A3]: predicted_label '{pred['predicted_label']}' not in "
        f"{expected_classes}."
    )
    assert 0.0 <= pred["confidence"] <= 1.0, (
        f"FAIL [A3]: confidence {pred['confidence']} not in [0, 1]."
    )
    print(f"  [A3] Single inference structure valid  [OK]  "
          f"(label='{pred['predicted_label']}', "
          f"confidence={pred['confidence']:.3f})")

    # A4 — score_post_sentiment ratios sum to 1.0
    post_ids = comments_df["post_id"].unique()
    test_post_id = int(post_ids[0])
    score = analyzer.score_post_sentiment(
        post_id=test_post_id, comments_df=comments_df
    )
    ratio_sum = score["positive_ratio"] + score["negative_ratio"] + score["neutral_ratio"]
    assert abs(ratio_sum - 1.0) <= 0.01, (
        f"FAIL [A4]: Sentiment ratios sum to {ratio_sum:.6f}, expected 1.0 "
        f"(+/- 0.01) for post_id={test_post_id}."
    )
    required_score_keys = {
        "post_id", "avg_sentiment_score", "positive_ratio",
        "negative_ratio", "neutral_ratio", "comment_count",
    }
    assert required_score_keys == set(score.keys()), (
        f"FAIL [A4]: score_post_sentiment() keys {set(score.keys())} "
        f"do not match expected {required_score_keys}."
    )
    print(f"  [A4] score_post_sentiment() ratios sum to {ratio_sum:.4f}  [OK]  "
          f"(post_id={test_post_id}, comments={score['comment_count']})")

    # ==================================================================
    # SECTION B — PerformanceClusterer
    # ==================================================================
    print()
    print("  [B] PerformanceClusterer (K-Means)")
    print("  " + "-" * 40)

    clusterer, tiered_df = cluster_and_save(engagement_df)

    # B1 — tiered_df only contains valid tier values
    valid_tiers   = {"Flop", "Average", "Viral"}
    actual_tiers  = set(tiered_df["performance_tier"].unique())
    invalid_tiers = actual_tiers - valid_tiers
    assert not invalid_tiers, (
        f"FAIL [B1]: performance_tier contains invalid values: {invalid_tiers}."
    )
    assert actual_tiers == valid_tiers, (
        f"FAIL [B1]: Not all expected tiers present. "
        f"Expected {valid_tiers}, got {actual_tiers}."
    )
    print(f"  [B1] Tier values valid: {sorted(actual_tiers)}  [OK]")

    # B2 — predict_tier returns a valid tier
    sample_row = engagement_df.iloc[0].to_dict()
    predicted  = clusterer.predict_tier(sample_row)
    assert predicted in valid_tiers, (
        f"FAIL [B2]: predict_tier() returned '{predicted}', "
        f"expected one of {valid_tiers}."
    )
    print(f"  [B2] predict_tier() returned '{predicted}'  [OK]")

    # B3 — cluster ordering: Flop_mean < Average_mean < Viral_mean
    summary = clusterer.get_cluster_summary()
    flop_rate    = float(summary.loc["Flop",    "engagement_rate"])
    average_rate = float(summary.loc["Average", "engagement_rate"])
    viral_rate   = float(summary.loc["Viral",   "engagement_rate"])
    assert flop_rate < average_rate, (
        f"FAIL [B3]: Flop engagement_rate ({flop_rate:.4f}) is not < "
        f"Average ({average_rate:.4f})."
    )
    assert average_rate < viral_rate, (
        f"FAIL [B3]: Average engagement_rate ({average_rate:.4f}) is not < "
        f"Viral ({viral_rate:.4f})."
    )
    print(f"  [B3] Tier ordering Flop({flop_rate:.4f}) < "
          f"Average({average_rate:.4f}) < Viral({viral_rate:.4f})  [OK]")

    # ==================================================================
    # SECTION C — Physical file existence
    # ==================================================================
    print()
    print("  [C] Physical artefacts on disk")
    print("  " + "-" * 40)

    tiered_csv_path = os.path.join(config.DATA_DIR, "engagement_tiered.csv")
    assert os.path.isfile(tiered_csv_path), (
        f"FAIL [C1]: engagement_tiered.csv not found at {tiered_csv_path}."
    )
    print(f"  [C1] data/engagement_tiered.csv exists  [OK]")

    scatter_path = os.path.join(config.OUTPUT_DIR, "cluster_scatter.png")
    assert os.path.isfile(scatter_path), (
        f"FAIL [C2]: cluster_scatter.png not found at {scatter_path}."
    )
    print(f"  [C2] outputs/cluster_scatter.png exists  [OK]")

    # ==================================================================
    # All passed
    # ==================================================================
    print()
    print("=" * 60)
    print("  ALL PHASE 3 VALIDATION TESTS PASSED")
    print("=" * 60)
    print()


if __name__ == "__main__":
    run_tests()
