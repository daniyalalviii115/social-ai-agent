"""
core/test_simulator.py — Phase 2 Validation Tests
===================================================
Loads the three CSVs produced by core/simulator.py and asserts that:
  1. posts.csv has exactly config.NUM_POSTS rows.
  2. engagement.csv has exactly config.NUM_POSTS rows.
  3. comments.csv row count is within the valid range implied by
     config.COMMENTS_PER_POST_RANGE = (5, 40).
  4. No NaN values exist in any DataFrame.
  5. engagement_rate values are all in [0, 1].
  6. true_sentiment_label only contains "positive", "negative", "neutral".

Run from social-ai-agent/ with the venv activated:
    .\\venv\\Scripts\\python.exe core\\test_simulator.py
"""

import os
import sys

import pandas as pd

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402


def run_tests() -> None:
    """
    Execute all Phase 2 validation assertions.

    Raises
    ------
    AssertionError
        On the first failing check, with a descriptive message.
    FileNotFoundError
        If a CSV file is missing (simulation has not been run yet).
    """
    posts_path      = os.path.join(config.DATA_DIR, "posts.csv")
    engagement_path = os.path.join(config.DATA_DIR, "engagement.csv")
    comments_path   = os.path.join(config.DATA_DIR, "comments.csv")

    # ------------------------------------------------------------------
    # 0. File existence guard
    # ------------------------------------------------------------------
    for path in (posts_path, engagement_path, comments_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing CSV: {path}\n"
                "Run `python core/simulator.py` first to generate the data."
            )

    # ------------------------------------------------------------------
    # Load CSVs
    # ------------------------------------------------------------------
    posts_df      = pd.read_csv(posts_path)
    engagement_df = pd.read_csv(engagement_path)
    comments_df   = pd.read_csv(comments_path)

    # ------------------------------------------------------------------
    # 1. posts_df row count == NUM_POSTS
    # ------------------------------------------------------------------
    assert len(posts_df) == config.NUM_POSTS, (
        f"FAIL: posts.csv has {len(posts_df)} rows, "
        f"expected {config.NUM_POSTS} (config.NUM_POSTS)."
    )

    # ------------------------------------------------------------------
    # 2. engagement_df row count == NUM_POSTS
    # ------------------------------------------------------------------
    assert len(engagement_df) == config.NUM_POSTS, (
        f"FAIL: engagement.csv has {len(engagement_df)} rows, "
        f"expected {config.NUM_POSTS} (config.NUM_POSTS)."
    )

    # ------------------------------------------------------------------
    # 3. comments_df row count in [NUM_POSTS * min_comments,
    #                               NUM_POSTS * max_comments]
    # ------------------------------------------------------------------
    min_comments = config.NUM_POSTS * config.COMMENTS_PER_POST_RANGE[0]
    max_comments = config.NUM_POSTS * config.COMMENTS_PER_POST_RANGE[1]
    assert min_comments <= len(comments_df) <= max_comments, (
        f"FAIL: comments.csv has {len(comments_df)} rows, "
        f"expected between {min_comments} and {max_comments} "
        f"(NUM_POSTS × COMMENTS_PER_POST_RANGE bounds)."
    )

    # ------------------------------------------------------------------
    # 4. No NaN values in any DataFrame
    # ------------------------------------------------------------------
    for name, df in [("posts", posts_df),
                     ("engagement", engagement_df),
                     ("comments", comments_df)]:
        nan_counts = df.isnull().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        assert cols_with_nan.empty, (
            f"FAIL: {name}.csv contains NaN values in columns:\n"
            f"{cols_with_nan.to_dict()}"
        )

    # ------------------------------------------------------------------
    # 5. engagement_rate in [0, 1]
    # ------------------------------------------------------------------
    rates = engagement_df["engagement_rate"]
    assert (rates >= 0.0).all() and (rates <= 1.0).all(), (
        f"FAIL: engagement_rate values out of [0, 1] bounds.\n"
        f"  min={rates.min():.6f}, max={rates.max():.6f}"
    )

    # ------------------------------------------------------------------
    # 6. true_sentiment_label only contains valid labels
    # ------------------------------------------------------------------
    valid_labels   = {"positive", "negative", "neutral"}
    actual_labels  = set(comments_df["true_sentiment_label"].unique())
    invalid_labels = actual_labels - valid_labels
    assert not invalid_labels, (
        f"FAIL: true_sentiment_label contains unexpected values: "
        f"{invalid_labels}. "
        f"Only {valid_labels} are allowed."
    )

    # ------------------------------------------------------------------
    # All checks passed
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  ALL PHASE 2 VALIDATION TESTS PASSED")
    print("=" * 60)
    print(f"  posts.csv      : {len(posts_df):>6} rows  [OK]")
    print(f"  engagement.csv : {len(engagement_df):>6} rows  [OK]")
    print(f"  comments.csv   : {len(comments_df):>6} rows  [OK]  "
          f"(range [{min_comments}, {max_comments}])")
    print(f"  NaN values     : none in any DataFrame  [OK]")
    print(f"  engagement_rate: all in [0, 1]  [OK]  "
          f"(mean={rates.mean():.4f})")
    sentiment_dist = comments_df["true_sentiment_label"].value_counts().to_dict()
    print(f"  Sentiment dist : {sentiment_dist}  [OK]")
    print("=" * 60)
    print()


if __name__ == "__main__":
    run_tests()
