"""
decision/fuzzy_engine.py — Mamdani Fuzzy Inference System
===========================================================
Phase 4 of the Autonomous Social Media AI Agent project.

Implements a Mamdani-style fuzzy logic system using scikit-fuzzy that
takes two crisp inputs — avg_sentiment_score (from SentimentAnalyzer)
and performance_tier (from PerformanceClusterer) — and outputs a
strategy_shift_rate in [0, 1] indicating how aggressively the content
strategy should be pivoted.

Downstream consumers:
  - Phase 6 (decision/genetic_optimizer.py): uses shift_rate as part
    of the fitness weighting for GA chromosome evaluation.
  - Phase 7 (dashboard/app.py): embeds fuzzy_membership.png and uses
    compute_shift_rate() in the Strategy Advisor panel.

Usage:
    python decision/fuzzy_engine.py
    from decision.fuzzy_engine import StrategyFuzzyEngine, evaluate_all_posts
"""

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for servers/headless
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `import config` works regardless of
# the working directory the file is run from.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402


class StrategyFuzzyEngine:
    """
    Mamdani Fuzzy Inference System for content strategy pivot decisions.

    Inputs
    ------
    sentiment : float in [-1.0, +1.0]
        avg_sentiment_score returned by SentimentAnalyzer.score_post_sentiment().
        Exactly the -1 to +1 scale — NOT renormalised to 0-1.
    performance_tier : str
        One of "Flop", "Average", "Viral" from PerformanceClusterer.predict_tier().
        Internally mapped to ordinal integers 0, 1, 2 for the fuzzy universe.

    Output
    ------
    strategy_shift_rate : float in [0, 1]
        Defuzzified shift rate.  Values closer to 1 mean "pivot strategy
        significantly"; values closer to 0 mean "stay the course".

    Rules (Mamdani, 9 rules for full coverage + meaningful semantics)
    ---------
    1. negative sentiment  + flop    tier -> HIGH shift rate
    2. negative sentiment  + average tier -> MEDIUM-HIGH (medium) shift rate
    3. negative sentiment  + viral   tier -> MEDIUM shift rate  (content resonates
                                            despite complaints — moderate rethink)
    4. neutral  sentiment  + flop    tier -> MEDIUM shift rate
    5. neutral  sentiment  + average tier -> MEDIUM shift rate
    6. neutral  sentiment  + viral   tier -> LOW shift rate
    7. positive sentiment  + flop    tier -> MEDIUM shift rate  (good vibes,
                                            poor reach — distribution pivot)
    8. positive sentiment  + average tier -> LOW-MEDIUM (medium) shift rate
    9. positive sentiment  + viral   tier -> LOW shift rate     (it's working!)
    """

    # ------------------------------------------------------------------
    # Constructor — builds the full fuzzy system
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        # ----------------------------------------------------------------
        # Antecedent 1: sentiment   universe [-1.0, +1.0]
        # ----------------------------------------------------------------
        self._sentiment_universe = np.arange(-1.0, 1.01, 0.01)
        sentiment = ctrl.Antecedent(self._sentiment_universe, "sentiment")

        # Membership functions — overlapping trimf for smooth transitions
        sentiment["negative"] = fuzz.trimf(sentiment.universe, [-1.0, -1.0, -0.1])
        sentiment["neutral"]  = fuzz.trimf(sentiment.universe, [-0.5,  0.0,  0.5])
        sentiment["positive"] = fuzz.trimf(sentiment.universe, [ 0.1,  1.0,  1.0])

        # ----------------------------------------------------------------
        # Antecedent 2: performance_tier  universe [0, 2]
        # 0 = Flop, 1 = Average, 2 = Viral
        # ----------------------------------------------------------------
        self._tier_universe = np.arange(0.0, 2.01, 0.01)
        performance_tier = ctrl.Antecedent(self._tier_universe, "performance_tier")

        performance_tier["flop"]    = fuzz.trimf(performance_tier.universe, [0.0, 0.0, 1.0])
        performance_tier["average"] = fuzz.trimf(performance_tier.universe, [0.0, 1.0, 2.0])
        performance_tier["viral"]   = fuzz.trimf(performance_tier.universe, [1.0, 2.0, 2.0])

        # ----------------------------------------------------------------
        # Consequent: strategy_shift_rate  universe [0, 1]
        # ----------------------------------------------------------------
        self._shift_universe = np.arange(0.0, 1.01, 0.01)
        strategy_shift_rate = ctrl.Consequent(
            self._shift_universe, "strategy_shift_rate", defuzzify_method="centroid"
        )

        strategy_shift_rate["low"]    = fuzz.trimf(strategy_shift_rate.universe, [0.0, 0.0, 0.4])
        strategy_shift_rate["medium"] = fuzz.trimf(strategy_shift_rate.universe, [0.3, 0.5, 0.7])
        strategy_shift_rate["high"]   = fuzz.trimf(strategy_shift_rate.universe, [0.6, 1.0, 1.0])

        # ----------------------------------------------------------------
        # Store references for plotting / inspection
        # ----------------------------------------------------------------
        self._sentiment         = sentiment
        self._performance_tier  = performance_tier
        self._shift_rate        = strategy_shift_rate

        # ----------------------------------------------------------------
        # Fuzzy Rules  (9 rules — complete combinatorial coverage)
        # ----------------------------------------------------------------
        # Rule 1: negative + flop -> high   (worst case, needs complete overhaul)
        rule1 = ctrl.Rule(
            sentiment["negative"] & performance_tier["flop"],
            strategy_shift_rate["high"],
            label="neg_flop_high",
        )
        # Rule 2: negative + average -> medium  (struggling average, rethink tone)
        rule2 = ctrl.Rule(
            sentiment["negative"] & performance_tier["average"],
            strategy_shift_rate["medium"],
            label="neg_avg_medium",
        )
        # Rule 3: negative + viral -> medium   (viral but toxic — moderate adjustment)
        rule3 = ctrl.Rule(
            sentiment["negative"] & performance_tier["viral"],
            strategy_shift_rate["medium"],
            label="neg_viral_medium",
        )
        # Rule 4: neutral + flop -> medium     (low engagement, unclear signal)
        rule4 = ctrl.Rule(
            sentiment["neutral"] & performance_tier["flop"],
            strategy_shift_rate["medium"],
            label="neu_flop_medium",
        )
        # Rule 5: neutral + average -> medium  (baseline performance, moderate tweak)
        rule5 = ctrl.Rule(
            sentiment["neutral"] & performance_tier["average"],
            strategy_shift_rate["medium"],
            label="neu_avg_medium",
        )
        # Rule 6: neutral + viral -> low       (viral with neutral comments — don't fix it)
        rule6 = ctrl.Rule(
            sentiment["neutral"] & performance_tier["viral"],
            strategy_shift_rate["low"],
            label="neu_viral_low",
        )
        # Rule 7: positive + flop -> medium    (great vibes, poor distribution — pivot reach)
        rule7 = ctrl.Rule(
            sentiment["positive"] & performance_tier["flop"],
            strategy_shift_rate["medium"],
            label="pos_flop_medium",
        )
        # Rule 8: positive + average -> medium  (promising, small optimisation needed)
        rule8 = ctrl.Rule(
            sentiment["positive"] & performance_tier["average"],
            strategy_shift_rate["medium"],
            label="pos_avg_medium",
        )
        # Rule 9: positive + viral -> low      (best case — leave strategy alone)
        rule9 = ctrl.Rule(
            sentiment["positive"] & performance_tier["viral"],
            strategy_shift_rate["low"],
            label="pos_viral_low",
        )

        # ----------------------------------------------------------------
        # Build the control system and simulation
        # ----------------------------------------------------------------
        self._rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]
        self._ctrl_system = ctrl.ControlSystem(self._rules)
        self._simulation  = ctrl.ControlSystemSimulation(self._ctrl_system)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_shift_rate(
        self,
        avg_sentiment_score: float,
        performance_tier: str,
    ) -> dict:
        """
        Run the Mamdani FIS for one post and return the defuzzified shift rate.

        Parameters
        ----------
        avg_sentiment_score : float
            Crisp sentiment value in [-1.0, +1.0] from SentimentAnalyzer.
        performance_tier : str
            One of "Flop", "Average", "Viral" from PerformanceClusterer.

        Returns
        -------
        dict with keys:
            shift_rate       : float  — defuzzified output in [0, 1]
            trigger_pivot    : bool   — True if shift_rate > 0.5
            sentiment_input  : float  — the clipped sentiment passed in
            tier_input       : float  — ordinal tier value (0/1/2)
        """
        tier_map = {"Flop": 0.0, "Average": 1.0, "Viral": 2.0}

        # Clip inputs to their valid universes to avoid FIS boundary errors
        sentiment_clipped = float(np.clip(avg_sentiment_score, -1.0, 1.0))
        tier_value        = float(tier_map.get(performance_tier, 1.0))

        self._simulation.input["sentiment"]        = sentiment_clipped
        self._simulation.input["performance_tier"] = tier_value

        try:
            self._simulation.compute()
            shift_rate = float(self._simulation.output["strategy_shift_rate"])
        except Exception:
            # Fallback to midpoint if FIS fails (should not happen with valid inputs)
            shift_rate = 0.5

        # Clip to [0, 1] for safety
        shift_rate = float(np.clip(shift_rate, 0.0, 1.0))

        return {
            "shift_rate":      shift_rate,
            "trigger_pivot":   bool(shift_rate > 0.5),
            "sentiment_input": sentiment_clipped,
            "tier_input":      tier_value,
        }

    def plot_membership_functions(self, save_path: Optional[str] = None) -> None:
        """
        Plot membership functions for all three fuzzy variables (3 subplots).

        Parameters
        ----------
        save_path : str, optional
            File path to save the figure.  Defaults to
            config.OUTPUT_DIR/fuzzy_membership.png.
        """
        if save_path is None:
            save_path = os.path.join(config.OUTPUT_DIR, "fuzzy_membership.png")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 11))
        fig.patch.set_facecolor("#0F1117")

        # Colour palette
        colors = {
            "negative": "#E74C3C",
            "neutral":  "#F39C12",
            "positive": "#27AE60",
            "flop":     "#E74C3C",
            "average":  "#F39C12",
            "viral":    "#27AE60",
            "low":      "#27AE60",
            "medium":   "#F39C12",
            "high":     "#E74C3C",
        }

        # --- Subplot 1: Sentiment -------------------------------------------
        ax0 = axes[0]
        ax0.set_facecolor("#1A1D27")
        for label in ["negative", "neutral", "positive"]:
            mf = fuzz.trimf(
                self._sentiment_universe,
                {
                    "negative": [-1.0, -1.0, -0.1],
                    "neutral":  [-0.5,  0.0,  0.5],
                    "positive": [ 0.1,  1.0,  1.0],
                }[label],
            )
            ax0.plot(self._sentiment_universe, mf, lw=2.0,
                     color=colors[label], label=label.capitalize())
            ax0.fill_between(self._sentiment_universe, mf, alpha=0.15,
                             color=colors[label])
        ax0.set_title("Antecedent 1 — Sentiment Score [-1, +1]",
                      color="white", fontsize=12, fontweight="bold", pad=8)
        ax0.set_xlabel("avg_sentiment_score", color="#AAAAAA", fontsize=10)
        ax0.set_ylabel("Membership Degree", color="#AAAAAA", fontsize=10)
        ax0.tick_params(colors="#AAAAAA")
        ax0.spines["bottom"].set_color("#444444")
        ax0.spines["left"].set_color("#444444")
        ax0.spines["top"].set_visible(False)
        ax0.spines["right"].set_visible(False)
        ax0.legend(facecolor="#22252F", edgecolor="#444444", labelcolor="white",
                   fontsize=9)
        ax0.set_xlim(-1.0, 1.0)
        ax0.set_ylim(-0.05, 1.1)
        ax0.grid(alpha=0.15, color="#AAAAAA")

        # --- Subplot 2: Performance Tier ------------------------------------
        ax1 = axes[1]
        ax1.set_facecolor("#1A1D27")
        tier_mf_params = {
            "flop":    [0.0, 0.0, 1.0],
            "average": [0.0, 1.0, 2.0],
            "viral":   [1.0, 2.0, 2.0],
        }
        for label, params in tier_mf_params.items():
            mf = fuzz.trimf(self._tier_universe, params)
            ax1.plot(self._tier_universe, mf, lw=2.0,
                     color=colors[label], label=label.capitalize())
            ax1.fill_between(self._tier_universe, mf, alpha=0.15,
                             color=colors[label])
        ax1.set_title("Antecedent 2 — Performance Tier [0=Flop, 1=Average, 2=Viral]",
                      color="white", fontsize=12, fontweight="bold", pad=8)
        ax1.set_xlabel("Tier (ordinal)", color="#AAAAAA", fontsize=10)
        ax1.set_ylabel("Membership Degree", color="#AAAAAA", fontsize=10)
        ax1.set_xticks([0, 1, 2])
        ax1.set_xticklabels(["Flop (0)", "Average (1)", "Viral (2)"])
        ax1.tick_params(colors="#AAAAAA")
        ax1.spines["bottom"].set_color("#444444")
        ax1.spines["left"].set_color("#444444")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.legend(facecolor="#22252F", edgecolor="#444444", labelcolor="white",
                   fontsize=9)
        ax1.set_xlim(0.0, 2.0)
        ax1.set_ylim(-0.05, 1.1)
        ax1.grid(alpha=0.15, color="#AAAAAA")

        # --- Subplot 3: Strategy Shift Rate ---------------------------------
        ax2 = axes[2]
        ax2.set_facecolor("#1A1D27")
        shift_mf_params = {
            "low":    [0.0, 0.0, 0.4],
            "medium": [0.3, 0.5, 0.7],
            "high":   [0.6, 1.0, 1.0],
        }
        for label, params in shift_mf_params.items():
            mf = fuzz.trimf(self._shift_universe, params)
            ax2.plot(self._shift_universe, mf, lw=2.0,
                     color=colors[label], label=label.capitalize())
            ax2.fill_between(self._shift_universe, mf, alpha=0.15,
                             color=colors[label])
        ax2.set_title("Consequent — Strategy Shift Rate [0, 1]",
                      color="white", fontsize=12, fontweight="bold", pad=8)
        ax2.set_xlabel("shift_rate", color="#AAAAAA", fontsize=10)
        ax2.set_ylabel("Membership Degree", color="#AAAAAA", fontsize=10)
        ax2.tick_params(colors="#AAAAAA")
        ax2.spines["bottom"].set_color("#444444")
        ax2.spines["left"].set_color("#444444")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.legend(facecolor="#22252F", edgecolor="#444444", labelcolor="white",
                   fontsize=9)
        ax2.set_xlim(0.0, 1.0)
        ax2.set_ylim(-0.05, 1.1)
        ax2.grid(alpha=0.15, color="#AAAAAA")

        # Add 0.5 decision boundary line
        ax2.axvline(x=0.5, color="#8888FF", lw=1.5, linestyle="--", alpha=0.6,
                    label="Pivot threshold (0.5)")
        ax2.legend(facecolor="#22252F", edgecolor="#444444", labelcolor="white",
                   fontsize=9)

        # Overall figure title
        fig.suptitle(
            "Mamdani Fuzzy Inference System — Membership Functions\n"
            "Social Media Strategy Pivot Engine",
            color="white",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )

        plt.tight_layout(pad=2.0)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Fuzzy membership functions plot saved to: {save_path}")


# ===========================================================================
# Module-level convenience function
# ===========================================================================

def evaluate_all_posts(
    sentiment_analyzer,
    clusterer,
    posts_df: pd.DataFrame,
    engagement_tiered_df: pd.DataFrame,
    comments_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the Mamdani FIS over every post in posts_df and return results.

    For each post_id:
      1. Computes avg_sentiment_score via sentiment_analyzer.score_post_sentiment()
      2. Looks up performance_tier from engagement_tiered_df
      3. Runs StrategyFuzzyEngine.compute_shift_rate()

    Parameters
    ----------
    sentiment_analyzer   : SentimentAnalyzer — must be fitted
    clusterer            : PerformanceClusterer — must be fitted (used for reference)
    posts_df             : pd.DataFrame — must contain 'post_id' column
    engagement_tiered_df : pd.DataFrame — must contain 'post_id' and 'performance_tier'
    comments_df          : pd.DataFrame — must contain 'post_id' and 'comment_text'

    Returns
    -------
    pd.DataFrame with columns:
        post_id, avg_sentiment_score, performance_tier, shift_rate, trigger_pivot

    Also saves the DataFrame to config.DATA_DIR/fuzzy_results.csv.
    """
    engine = StrategyFuzzyEngine()

    # Build fast tier lookup dict: post_id -> tier string
    tier_lookup: dict[int, str] = dict(
        zip(
            engagement_tiered_df["post_id"].astype(int),
            engagement_tiered_df["performance_tier"].astype(str),
        )
    )

    records = []
    post_ids = posts_df["post_id"].astype(int).tolist()

    for post_id in post_ids:
        # --- Sentiment score -----------------------------------------------
        sentiment_result = sentiment_analyzer.score_post_sentiment(
            post_id=post_id, comments_df=comments_df
        )
        avg_score = float(sentiment_result["avg_sentiment_score"])

        # --- Performance tier -----------------------------------------------
        tier = tier_lookup.get(post_id, "Average")   # fallback to Average

        # --- Fuzzy computation ---------------------------------------------
        fuzzy_result = engine.compute_shift_rate(
            avg_sentiment_score=avg_score,
            performance_tier=tier,
        )

        records.append(
            {
                "post_id":             post_id,
                "avg_sentiment_score": round(avg_score, 6),
                "performance_tier":    tier,
                "shift_rate":          round(fuzzy_result["shift_rate"], 6),
                "trigger_pivot":       bool(fuzzy_result["trigger_pivot"]),
            }
        )

    results_df = pd.DataFrame(records)

    # Save to DATA_DIR
    save_path = os.path.join(config.DATA_DIR, "fuzzy_results.csv")
    results_df.to_csv(save_path, index=False)
    print(f"  Fuzzy results saved to: {save_path}")

    return results_df


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import ml.naive_bayes as nb_module
    import ml.kmeans_cluster as km_module

    print("=" * 65)
    print("  PHASE 4a — FUZZY ENGINE EXECUTION")
    print("=" * 65)

    # --- Load CSVs -----------------------------------------------------------
    posts_path          = os.path.join(config.DATA_DIR, "posts.csv")
    engagement_path     = os.path.join(config.DATA_DIR, "engagement.csv")
    tiered_path         = os.path.join(config.DATA_DIR, "engagement_tiered.csv")
    comments_path       = os.path.join(config.DATA_DIR, "comments.csv")

    print(f"  Loading posts           : {posts_path}")
    posts_df             = pd.read_csv(posts_path)

    print(f"  Loading engagement      : {engagement_path}")
    engagement_df        = pd.read_csv(engagement_path)

    print(f"  Loading tiered data     : {tiered_path}")
    engagement_tiered_df = pd.read_csv(tiered_path)

    print(f"  Loading comments        : {comments_path}")
    comments_df          = pd.read_csv(comments_path)

    # --- Retrain models fresh ------------------------------------------------
    print()
    print("  Retraining SentimentAnalyzer...")
    sentiment_analyzer = nb_module.train_and_save_model(comments_df)

    print()
    print("  Refitting PerformanceClusterer...")
    clusterer, _ = km_module.cluster_and_save(engagement_df)

    # --- Run evaluate_all_posts ----------------------------------------------
    print()
    print("  Running evaluate_all_posts()...")
    results_df = evaluate_all_posts(
        sentiment_analyzer=sentiment_analyzer,
        clusterer=clusterer,
        posts_df=posts_df,
        engagement_tiered_df=engagement_tiered_df,
        comments_df=comments_df,
    )

    # --- Summary stats -------------------------------------------------------
    mean_shift   = results_df["shift_rate"].mean()
    pivot_count  = int(results_df["trigger_pivot"].sum())
    total_posts  = len(results_df)

    print()
    print("=" * 65)
    print("  FUZZY ENGINE SUMMARY")
    print("=" * 65)
    print(f"  Total posts evaluated   : {total_posts}")
    print(f"  Mean shift_rate         : {mean_shift:.4f}")
    print(f"  Posts where trigger_pivot=True : {pivot_count} / {total_posts} "
          f"({pivot_count / total_posts * 100:.1f}%)")
    print()

    # Sample breakdown by tier
    tier_stats = (
        results_df.groupby("performance_tier")["shift_rate"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_shift_rate", "count": "posts"})
    )
    print("  Shift rate breakdown by tier:")
    print(tier_stats.to_string())
    print()

    # --- Plot membership functions -------------------------------------------
    mf_path = os.path.join(config.OUTPUT_DIR, "fuzzy_membership.png")
    engine_for_plot = StrategyFuzzyEngine()
    engine_for_plot.plot_membership_functions(save_path=mf_path)

    print("=" * 65)
    print("  PHASE 4a COMPLETE")
    print("=" * 65)
