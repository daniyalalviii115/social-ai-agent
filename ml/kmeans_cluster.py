"""
ml/kmeans_cluster.py — K-Means Performance Tier Clusterer
===========================================================
Phase 3 of the Autonomous Social Media AI Agent project.

Applies K-Means (k = config.KMEANS_N_CLUSTERS = 3) to the engagement metrics
produced by Phase 2, discovering three performance tiers:

    "Flop"    — lowest mean engagement_rate cluster
    "Average" — middle cluster
    "Viral"   — highest mean engagement_rate cluster

Downstream consumers:
  - Phase 5 (ml/fuzzy_engine.py)       : performance_tier used as one fuzzy
                                          output category for rule evaluation.
  - Phase 6 (ml/genetic_algorithm.py)  : GA fitness function uses tier labels
                                          to reward strategies that push posts
                                          toward the "Viral" cluster.
  - Phase 7 (dashboard/app.py)         : cluster scatter plot embedded in the
                                          Performance Analysis tab.

Usage:
    python ml/kmeans_cluster.py
    from ml.kmeans_cluster import PerformanceClusterer, cluster_and_save
"""

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402


class PerformanceClusterer:
    """
    K-Means engagement performance tier classifier.

    Clusters Instagram post engagement records into three human-readable
    tiers — "Flop", "Average", "Viral" — ordered by mean engagement_rate
    ascending.  The mapping from raw K-Means cluster IDs to tier names is
    learned during :meth:`fit` so it is always consistent regardless of the
    random cluster initialisation.

    Attributes
    ----------
    FEATURE_COLS : list[str]
        Engagement columns used as K-Means input features.
    TIER_NAMES : list[str]
        Human-readable tier labels, ascending by engagement quality.
    scaler : StandardScaler
        Fitted scaler (set after :meth:`fit`).
    kmeans : KMeans
        Fitted KMeans instance (set after :meth:`fit`).
    cluster_to_tier : dict[int, str]
        Mapping from raw cluster ID -> tier name (set after :meth:`fit`).
    is_fitted : bool
        True after a successful call to :meth:`fit`.
    """

    FEATURE_COLS: list[str] = [
        "impressions",
        "likes",
        "comments_count",
        "shares",
        "saves",
        "engagement_rate",
    ]
    TIER_NAMES: list[str] = ["Flop", "Average", "Viral"]

    def __init__(self) -> None:
        self.scaler: StandardScaler = StandardScaler()
        self.kmeans: KMeans = KMeans(
            n_clusters=config.KMEANS_N_CLUSTERS,
            random_state=config.KMEANS_RANDOM_STATE,
            n_init=10,
        )
        self.cluster_to_tier: dict[int, str] = {}
        self.is_fitted: bool = False
        self._summary: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, engagement_df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale engagement features and fit K-Means; assign tier labels.

        Cluster ordering is determined by mean ``engagement_rate`` ascending:
        the cluster with the lowest mean rate becomes "Flop", the middle
        becomes "Average", and the highest becomes "Viral".

        Parameters
        ----------
        engagement_df : pd.DataFrame
            Must contain all columns in ``FEATURE_COLS``.

        Returns
        -------
        pd.DataFrame
            Copy of ``engagement_df`` with an additional ``performance_tier``
            column containing "Flop" | "Average" | "Viral" strings.

        Raises
        ------
        ValueError
            If any required feature column is missing.
        """
        missing = set(self.FEATURE_COLS) - set(engagement_df.columns)
        if missing:
            raise ValueError(
                f"engagement_df is missing required columns: {missing}"
            )

        df = engagement_df.copy()
        X  = df[self.FEATURE_COLS].values.astype(float)

        X_scaled = self.scaler.fit_transform(X)
        labels   = self.kmeans.fit_predict(X_scaled)
        df["_cluster_id"] = labels

        # Rank cluster IDs by their mean engagement_rate (ascending)
        mean_rates = (
            df.groupby("_cluster_id")["engagement_rate"]
            .mean()
            .sort_values(ascending=True)
        )
        # Map rank 0 -> Flop, 1 -> Average, 2 -> Viral
        self.cluster_to_tier = {
            int(cluster_id): self.TIER_NAMES[rank]
            for rank, cluster_id in enumerate(mean_rates.index)
        }

        df["performance_tier"] = df["_cluster_id"].map(self.cluster_to_tier)
        df = df.drop(columns=["_cluster_id"])

        self.is_fitted = True

        # Build and cache summary table
        self._summary = self._build_summary(df)

        return df

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "PerformanceClusterer has not been fitted. "
                "Call .fit(engagement_df) first."
            )

    def predict_tier(self, new_engagement_row: dict) -> str:
        """
        Predict the performance tier for a single engagement record.

        Parameters
        ----------
        new_engagement_row : dict
            Must contain all keys listed in ``FEATURE_COLS``.

        Returns
        -------
        str
            One of "Flop", "Average", or "Viral".
        """
        self._check_fitted()
        row_df   = pd.DataFrame([new_engagement_row])[self.FEATURE_COLS]
        X_scaled = self.scaler.transform(row_df.values.astype(float))
        cluster  = int(self.kmeans.predict(X_scaled)[0])
        return self.cluster_to_tier[cluster]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _build_summary(self, tiered_df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-tier feature means and post counts."""
        agg = tiered_df.groupby("performance_tier")[self.FEATURE_COLS].mean()
        counts = tiered_df["performance_tier"].value_counts().rename("post_count")
        summary = agg.join(counts)
        # Reorder rows: Flop → Average → Viral
        summary = summary.reindex(
            [t for t in self.TIER_NAMES if t in summary.index]
        )
        return summary.round(4)

    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Return summary DataFrame of feature means and post counts per tier.

        Returns
        -------
        pd.DataFrame
            Index = tier names ("Flop", "Average", "Viral");
            Columns = FEATURE_COLS + ["post_count"].

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        self._check_fitted()
        return self._summary.copy()

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_clusters(
        self,
        engagement_df_with_tiers: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Scatter plot of ``impressions`` vs ``engagement_rate`` coloured by tier.

        Parameters
        ----------
        engagement_df_with_tiers : pd.DataFrame
            Output of :meth:`fit`; must contain ``impressions``,
            ``engagement_rate``, and ``performance_tier`` columns.
        save_path : str, optional
            File path to save the PNG.  If None, defaults to
            ``config.OUTPUT_DIR / 'cluster_scatter.png'``.
        """
        self._check_fitted()

        if save_path is None:
            save_path = os.path.join(config.OUTPUT_DIR, "cluster_scatter.png")

        palette = {
            "Flop":    "#E74C3C",   # Red
            "Average": "#F39C12",   # Amber
            "Viral":   "#27AE60",   # Green
        }
        tier_order = ["Flop", "Average", "Viral"]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=engagement_df_with_tiers,
            x="impressions",
            y="engagement_rate",
            hue="performance_tier",
            hue_order=tier_order,
            palette=palette,
            s=80,
            alpha=0.80,
            edgecolor="white",
            linewidth=0.4,
            ax=ax,
        )

        ax.set_title(
            "K-Means Performance Tiers: Impressions vs Engagement Rate",
            fontsize=13,
            fontweight="bold",
            pad=14,
        )
        ax.set_xlabel("Impressions", fontsize=11)
        ax.set_ylabel("Engagement Rate", fontsize=11)
        ax.legend(title="Performance Tier", fontsize=10, title_fontsize=10)
        sns.despine()
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Cluster scatter plot saved to: {save_path}")


# ===========================================================================
# Module-level convenience function
# ===========================================================================

def cluster_and_save(
    engagement_df: Optional[pd.DataFrame] = None,
) -> tuple["PerformanceClusterer", pd.DataFrame]:
    """
    Load engagement data, fit clusterer, save tiered CSV and scatter plot.

    Parameters
    ----------
    engagement_df : pd.DataFrame, optional
        If None, loads ``engagement.csv`` from ``config.DATA_DIR``.

    Returns
    -------
    tuple[PerformanceClusterer, pd.DataFrame]
        ``(clusterer, tiered_df)`` where ``tiered_df`` includes the
        ``performance_tier`` column and has been saved to
        ``data/engagement_tiered.csv``.
    """
    if engagement_df is None:
        path = os.path.join(config.DATA_DIR, "engagement.csv")
        print(f"Loading engagement data from: {path}")
        engagement_df = pd.read_csv(path)

    clusterer  = PerformanceClusterer()
    tiered_df  = clusterer.fit(engagement_df)

    # Save tiered CSV
    tiered_path = os.path.join(config.DATA_DIR, "engagement_tiered.csv")
    tiered_df.to_csv(tiered_path, index=False)
    print(f"  Tiered engagement CSV saved to: {tiered_path}")

    # Save cluster scatter plot
    plot_path = os.path.join(config.OUTPUT_DIR, "cluster_scatter.png")
    clusterer.plot_clusters(tiered_df, save_path=plot_path)

    # Print summary
    summary = clusterer.get_cluster_summary()
    print()
    print("=" * 70)
    print("  K-MEANS CLUSTER SUMMARY  (k=3, ordered Flop < Average < Viral)")
    print("=" * 70)
    print(summary.to_string())
    print()

    # Verify tier ordering by engagement_rate
    flop_rate    = summary.loc["Flop",    "engagement_rate"]
    average_rate = summary.loc["Average", "engagement_rate"]
    viral_rate   = summary.loc["Viral",   "engagement_rate"]
    print(f"  Tier ordering by engagement_rate:")
    print(f"    Flop    mean = {flop_rate:.4f}")
    print(f"    Average mean = {average_rate:.4f}")
    print(f"    Viral   mean = {viral_rate:.4f}")
    assert flop_rate < average_rate < viral_rate, (
        "UNEXPECTED: tier engagement_rate ordering is not Flop < Average < Viral!"
    )
    print("  Ordering verified: Flop < Average < Viral [OK]")
    print("=" * 70)

    # Tier distribution
    dist = tiered_df["performance_tier"].value_counts()
    print()
    print("  Post distribution across tiers:")
    for tier in PerformanceClusterer.TIER_NAMES:
        cnt = int(dist.get(tier, 0))
        print(f"    {tier:<10}: {cnt} posts")
    print()

    return clusterer, tiered_df


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    cluster_and_save()
