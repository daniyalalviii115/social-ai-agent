"""
core/trend_radar.py — Real-Time Trend Radar
=============================================
Phase 2 of the Autonomous Social Media AI Agent project (Instagram
Autonomous AI Manager upgrade).

Fetches live trending search topics per niche using Google Trends' public
RSS feed (no API key required, zero-cost). Falls back to a deterministic,
curated offline topic list whenever the network is unavailable, the feed
is empty, or the request fails for any reason — this project is designed
to always run 100% offline-capable, so a network hiccup must never halt
the pipeline.

Downstream consumers:
  - decision/pivot_manager.py (Level 2 — Topic Exhaustion Shift): pulls the
    next hot topic for a niche when the current topic goes stale.
  - dashboard/app.py (optional): can display live trends per niche.

Usage:
    python core/trend_radar.py
    from core.trend_radar import TrendRadar, get_trending_topics
"""

import os
import sys
import time
import random
import xml.etree.ElementTree as ET
from typing import Optional

import requests

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.simulator import NICHES  # noqa: E402

# ---------------------------------------------------------------------------
# Google Trends "daily trends" RSS feed (public, no key required).
# geo=US is used as a broad, high-volume default; this endpoint returns
# general trending searches, not niche-specific ones — we filter/match them
# against our niches heuristically, and always have the offline list ready
# as the primary source of niche-specific relevance.
# ---------------------------------------------------------------------------
_TRENDS_RSS_URL = "https://trends.google.com/trends/trendingsearches/daily/rss?geo=US"
_REQUEST_TIMEOUT = 6  # seconds — fail fast, this must never block the pipeline

# ---------------------------------------------------------------------------
# Deterministic offline fallback — curated, evergreen-ish hot topics per
# niche. Used whenever the live fetch fails, returns nothing usable, or the
# machine is offline. This keeps the project's "100% offline-capable"
# guarantee intact even for this new module.
# ---------------------------------------------------------------------------
_OFFLINE_TOPIC_CACHE: dict[str, list[str]] = {
    "Fitness": [
        "12-3-30 treadmill workout", "zone 2 cardio", "hybrid training",
        "creatine loading myths", "cold plunge recovery", "rucking workouts",
    ],
    "Personal Finance": [
        "high-yield savings rates", "index fund investing", "side hustle taxes",
        "debt snowball vs avalanche", "emergency fund rule", "Roth IRA basics",
    ],
    "Tech Reviews": [
        "on-device AI models", "foldable phone durability", "budget flagship killers",
        "AI coding assistants", "USB-C ecosystem changes", "battery health tips",
    ],
    "Food & Recipes": [
        "high-protein snacks", "one-pan dinners", "air fryer hacks",
        "viral pasta recipes", "meal prep for beginners", "fermented foods trend",
    ],
    "Travel": [
        "shoulder season destinations", "digital nomad visas", "budget solo travel",
        "hidden gem beaches", "slow travel itineraries", "carry-on only packing",
    ],
}

_OFFLINE_TOPIC_CACHE["_default"] = [
    "trending challenge", "day in the life", "behind the scenes",
    "top 3 mistakes beginners make", "this changed everything for me",
]


class TrendRadar:
    """
    Fetches and caches trending topics, structured per niche.

    Public interface
    -----------------
    get_trends(niche: str, force_refresh: bool = False) -> list[str]
        Returns a list of hot topics for the given niche.

    get_all_trends(force_refresh: bool = False) -> dict[str, list[str]]
        Returns {niche: [list_of_hot_topics]} for every configured niche.
    """

    def __init__(self) -> None:
        self._cache: dict[str, list[str]] = {}
        self._last_fetch_ts: float = 0.0
        self._cache_ttl_seconds: int = 3600  # re-fetch at most once per hour

    # ------------------------------------------------------------------
    # Live fetch (Google Trends RSS)
    # ------------------------------------------------------------------

    def _fetch_live_trending_terms(self) -> list[str]:
        """
        Fetch raw trending search terms from the Google Trends RSS feed.
        Returns an empty list on ANY failure (network, parsing, timeout) —
        callers must always be ready to fall back to the offline cache.
        """
        try:
            resp = requests.get(_TRENDS_RSS_URL, timeout=_REQUEST_TIMEOUT)
            if resp.status_code != 200:
                return []

            root = ET.fromstring(resp.content)
            terms: list[str] = []
            for item in root.iter("item"):
                title_el = item.find("title")
                if title_el is not None and title_el.text:
                    terms.append(title_el.text.strip())
            return terms
        except Exception as exc:
            print(f"  [TrendRadar] Live fetch failed ({type(exc).__name__}: {exc}). "
                  "Using offline cache.")
            return []

    def _match_terms_to_niche(self, niche: str, raw_terms: list[str]) -> list[str]:
        """
        Very lightweight keyword-overlap matching between raw trending
        search terms and a niche. Google Trends' daily feed is general
        (not niche-scoped), so this is a best-effort heuristic filter —
        the offline cache remains the authoritative, always-relevant
        source of niche-specific topics.
        """
        niche_keywords = {
            "Fitness": {"gym", "workout", "fitness", "health", "training", "diet"},
            "Personal Finance": {"stock", "market", "finance", "money", "invest", "economy", "bank"},
            "Tech Reviews": {"phone", "app", "tech", "ai", "software", "gadget", "launch"},
            "Food & Recipes": {"recipe", "food", "restaurant", "cook", "meal"},
            "Travel": {"travel", "flight", "airport", "destination", "vacation", "tourism"},
        }.get(niche, set())

        matched = [
            term for term in raw_terms
            if any(kw in term.lower() for kw in niche_keywords)
        ]
        return matched

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_trends(self, niche: str, force_refresh: bool = False) -> list[str]:
        """
        Return a list of hot topics for `niche`. Tries the live feed first
        (cached for up to 1 hour), matches terms heuristically to the
        niche, and always tops up with the offline curated list so the
        result is never empty or purely generic.
        """
        now = time.time()
        cache_stale = (now - self._last_fetch_ts) > self._cache_ttl_seconds

        if force_refresh or cache_stale or not self._cache:
            raw_terms = self._fetch_live_trending_terms()
            self._cache["_raw"] = raw_terms
            self._last_fetch_ts = now

        raw_terms = self._cache.get("_raw", [])
        live_matches = self._match_terms_to_niche(niche, raw_terms) if raw_terms else []

        offline_topics = _OFFLINE_TOPIC_CACHE.get(niche, _OFFLINE_TOPIC_CACHE["_default"])

        # Live matches (if any) are surfaced first, then topped up with the
        # curated offline list, de-duplicated, capped at 6 topics.
        combined = list(dict.fromkeys(live_matches + offline_topics))
        return combined[:6]

    def get_all_trends(self, force_refresh: bool = False) -> dict[str, list[str]]:
        """Return {niche: [list_of_hot_topics]} for every configured niche."""
        return {niche: self.get_trends(niche, force_refresh=force_refresh) for niche in NICHES}

    def get_next_topic(self, niche: str, exclude: Optional[list[str]] = None) -> str:
        """
        Return a single 'next' hot topic for a niche, optionally excluding
        topics already tried (used by pivot_manager's Level-2 topic-shift).
        Falls back to a random offline topic if every candidate is excluded.
        """
        exclude = exclude or []
        topics = self.get_trends(niche)
        remaining = [t for t in topics if t not in exclude]

        if remaining:
            return remaining[0]

        offline_topics = _OFFLINE_TOPIC_CACHE.get(niche, _OFFLINE_TOPIC_CACHE["_default"])
        return random.choice(offline_topics)


# ===========================================================================
# Module-level convenience function
# ===========================================================================

def get_trending_topics(force_refresh: bool = False) -> dict[str, list[str]]:
    """Convenience wrapper: {niche: [list_of_hot_topics]} for all niches."""
    radar = TrendRadar()
    return radar.get_all_trends(force_refresh=force_refresh)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  PHASE 2a — REAL-TIME TREND RADAR DEMO")
    print("=" * 65)

    radar = TrendRadar()
    all_trends = radar.get_all_trends()

    for niche, topics in all_trends.items():
        print(f"\n  {niche}:")
        for topic in topics:
            print(f"    - {topic}")

    print()
    print("=" * 65)
    print("  PHASE 2a COMPLETE")
    print("=" * 65)