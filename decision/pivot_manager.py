"""
decision/pivot_manager.py — Two-Tier Adaptive Pivot Manager
===============================================================
Phase 2 of the Autonomous Social Media AI Agent project (Instagram
Autonomous AI Manager upgrade).

Coordinates two escalating levels of autonomous strategy adaptation:

  Level 1 — Strategy Mutation:
      Triggered whenever the Mamdani fuzzy engine's shift_rate exceeds
      0.55 for a single post. Runs the existing Genetic Algorithm to
      evolve a new hook_type / content_tone / visual_style / posting_hour
      combination, while KEEPING the current topic unchanged.

  Level 2 — Topic Exhaustion Shift:
      Triggered when a *topic* (not just a single post) shows sustained
      failure — landing in the "Flop" tier OR generating negative
      sentiment across 2 consecutive iterations on that same topic. When
      this happens, the manager abandons the topic entirely and pulls the
      next hot topic for the niche from core.trend_radar.

This module is stateful across calls for a given niche (it needs to
remember the last 2 iterations' tier/sentiment per topic to detect
"2 consecutive" failures), so it is used as a class instance that the
caller (main.py / dashboard) keeps alive across the session.

Downstream consumers:
  - main.py (--full-run / --generate-reel orchestration)
  - dashboard/app.py (Instagram Manager / Content Studio autonomous loop)

Usage:
    python decision/pivot_manager.py
    from decision.pivot_manager import PivotManager
"""

import os
import sys
from collections import defaultdict, deque
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from decision.fuzzy_engine import StrategyFuzzyEngine  # noqa: E402
from decision.genetic_optimizer import optimize_strategy  # noqa: E402
from core.trend_radar import TrendRadar  # noqa: E402

# Sentiment score below this is treated as "negative" for Level-2 tracking.
_NEGATIVE_SENTIMENT_THRESHOLD: float = -0.15

# Fuzzy shift_rate above this triggers a Level-1 mutation.
_LEVEL1_SHIFT_THRESHOLD: float = 0.55

# Consecutive Flop/negative-sentiment iterations on the SAME topic that
# trigger a Level-2 full topic switch.
_LEVEL2_CONSECUTIVE_FAILURES: int = 2


class PivotManager:
    """
    Stateful two-tier adaptive pivot coordinator.

    Call `evaluate()` once per new post/iteration with that post's
    (niche, topic, performance_tier, avg_sentiment_score, current gene
    dict). It returns a decision dict describing what — if anything — the
    autonomous agent should change before generating the next post.
    """

    def __init__(self, fitness_lookup_df=None, posts_df=None, engagement_df=None) -> None:
        self.fuzzy_engine = StrategyFuzzyEngine()
        self.trend_radar = TrendRadar()

        # posts_df / engagement_df are passed straight through to
        # optimize_strategy() for Level-1 GA runs (same interface main.py
        # and dashboard/app.py already use).
        self._posts_df = posts_df
        self._engagement_df = engagement_df

        # Per-(niche, topic) rolling history of the last N iteration
        # outcomes, used to detect "2 consecutive failures" for Level 2.
        # Each entry: {"tier": str, "sentiment": float}
        self._topic_history: dict[tuple, deque] = defaultdict(
            lambda: deque(maxlen=_LEVEL2_CONSECUTIVE_FAILURES)
        )

        # Topics already abandoned per niche, so Level 2 never re-suggests
        # a topic that already failed in this session.
        self._exhausted_topics: dict[str, list[str]] = defaultdict(list)

        # Currently active topic per niche (None until first evaluate() call).
        self._active_topic: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_outcome(self, niche: str, topic: str, tier: str, sentiment: float) -> None:
        key = (niche, topic)
        self._topic_history[key].append({"tier": tier, "sentiment": sentiment})

    def _is_topic_exhausted(self, niche: str, topic: str) -> bool:
        """
        True if the last `_LEVEL2_CONSECUTIVE_FAILURES` iterations on this
        (niche, topic) were ALL failures — Flop tier OR negative sentiment.
        """
        key = (niche, topic)
        history = self._topic_history[key]

        if len(history) < _LEVEL2_CONSECUTIVE_FAILURES:
            return False

        def _is_failure(entry: dict) -> bool:
            return entry["tier"] == "Flop" or entry["sentiment"] < _NEGATIVE_SENTIMENT_THRESHOLD

        return all(_is_failure(entry) for entry in history)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_active_topic(self, niche: str) -> str:
        """
        Return the currently active topic for a niche, picking a fresh one
        from the trend radar if none is active yet.
        """
        if niche not in self._active_topic:
            self._active_topic[niche] = self.trend_radar.get_next_topic(
                niche, exclude=self._exhausted_topics[niche]
            )
        return self._active_topic[niche]

    def evaluate(
        self,
        niche: str,
        performance_tier: str,
        avg_sentiment_score: float,
        current_individual: dict,
        topic: Optional[str] = None,
    ) -> dict:
        """
        Evaluate one post's outcome and decide what (if anything) the
        autonomous agent should change for the next post.

        Parameters
        ----------
        niche                : str  — post's niche
        performance_tier     : str  — "Flop" / "Average" / "Viral"
        avg_sentiment_score  : float — Naive Bayes sentiment score for the post
        current_individual   : dict — current gene dict (hook_type,
                                posting_hour, visual_style, content_tone)
        topic                : str, optional — topic this post was about.
                                If omitted, uses/creates the niche's active topic.

        Returns
        -------
        dict with keys:
            level            : int  — 0 (no pivot), 1 (strategy mutation),
                               or 2 (topic exhaustion shift)
            shift_rate       : float — fuzzy engine's raw output
            new_individual   : dict — gene combo to use next (unchanged if level 0)
            topic            : str  — topic to use next (unchanged unless level 2)
            reason           : str  — human-readable explanation
        """
        topic = topic or self.get_active_topic(niche)
        self._record_outcome(niche, topic, performance_tier, avg_sentiment_score)

        # --- Level 2 check first: topic exhaustion is the more severe pivot ---
        if self._is_topic_exhausted(niche, topic):
            self._exhausted_topics[niche].append(topic)
            new_topic = self.trend_radar.get_next_topic(
                niche, exclude=self._exhausted_topics[niche]
            )
            self._active_topic[niche] = new_topic
            self._topic_history[(niche, topic)].clear()

            return {
                "level": 2,
                "shift_rate": None,
                "new_individual": current_individual,
                "topic": new_topic,
                "reason": (
                    f"Topic '{topic}' failed {_LEVEL2_CONSECUTIVE_FAILURES} consecutive "
                    f"iterations (Flop tier or negative sentiment). Switching to '{new_topic}'."
                ),
            }

        # --- Level 1 check: fuzzy shift_rate on this single post ------------
        fuzzy_result = self.fuzzy_engine.compute_shift_rate(avg_sentiment_score, performance_tier)
        shift_rate = fuzzy_result["shift_rate"]

        if fuzzy_result["trigger_pivot"] or shift_rate > _LEVEL1_SHIFT_THRESHOLD:
            ga_result = optimize_strategy(self._posts_df, self._engagement_df)
            return {
                "level": 1,
                "shift_rate": shift_rate,
                "new_individual": ga_result["best_individual"],
                "topic": topic,
                "reason": (
                    f"Shift rate {shift_rate:.3f} exceeded the {_LEVEL1_SHIFT_THRESHOLD} "
                    "threshold. Evolved a new hook/tone via GA, topic unchanged."
                ),
            }

        # --- No pivot needed --------------------------------------------------
        return {
            "level": 0,
            "shift_rate": shift_rate,
            "new_individual": current_individual,
            "topic": topic,
            "reason": "Strategy stable — no pivot triggered.",
        }


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import pandas as pd

    print("=" * 65)
    print("  PHASE 2b — TWO-TIER ADAPTIVE PIVOT MANAGER DEMO")
    print("=" * 65)

    posts_df = pd.read_csv(os.path.join(config.DATA_DIR, "posts.csv"))
    engagement_df = pd.read_csv(os.path.join(config.DATA_DIR, "engagement.csv"))

    manager = PivotManager(posts_df=posts_df, engagement_df=engagement_df)

    demo_individual = {
        "hook_type": "Question",
        "posting_hour": 20,
        "visual_style": "Bold Colorful",
        "content_tone": "Inspirational",
    }
    niche = "Tech Reviews"
    topic = manager.get_active_topic(niche)
    print(f"\n  Active topic for '{niche}': {topic}")

    # Simulate 3 consecutive Flop-tier, negative-sentiment posts on the
    # same topic to demonstrate the Level-2 topic-exhaustion shift.
    for i in range(1, 4):
        decision = manager.evaluate(
            niche=niche,
            performance_tier="Flop",
            avg_sentiment_score=-0.4,
            current_individual=demo_individual,
            topic=topic,
        )
        print(f"\n  Iteration {i}:")
        print(f"    Level  : {decision['level']}")
        print(f"    Reason : {decision['reason']}")
        if decision["level"] == 2:
            topic = decision["topic"]

    print()
    print("=" * 65)
    print("  PHASE 2b COMPLETE")
    print("=" * 65)