"""
core/simulator.py — Synthetic Instagram Data Simulator
=======================================================
Phase 2 of the Autonomous Social Media AI Agent project.

This module generates a fully synthetic dataset of Instagram posts,
engagement metrics, and user comments that feeds every downstream phase:

  - Phase 3 (core/naive_bayes.py)   : comments_df with true_sentiment_label
                                       used as labelled training data.
  - Phase 4 (ml/clustering.py)      : posts_df + engagement_df merged for
                                       K-Means audience-segment discovery.
  - Phase 5 (ml/fuzzy_engine.py)    : engagement_rate column used as fuzzy
                                       input universe variable.
  - Phase 6 (ml/genetic_algorithm.py): HOOK_TYPES, POSTING_HOURS, VISUAL_STYLES,
                                       CONTENT_TONES, NICHES imported directly
                                       as the GA chromosome gene search space.
  - Phase 7 (dashboard/app.py)      : all three CSVs loaded for visualisation.

Usage:
    python core/simulator.py           # runs run_simulation(), saves CSVs
    from core.simulator import run_simulation, HOOK_TYPES, NICHES, ...
"""

import os
import sys
import random
import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make sure the project root (social-ai-agent/) is on sys.path so that
# `import config` works regardless of the working directory from which this
# file is executed (e.g. `python core/simulator.py` from project root, or
# imported as a module from another package).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402  (after sys.path manipulation)

# ---------------------------------------------------------------------------
# Reproducibility — seed both Python stdlib random and NumPy.
# Using config.RANDOM_SEED (= 42) guarantees identical output on every run.
# ---------------------------------------------------------------------------
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)


# ===========================================================================
# Module-level gene search-space constants
# These lists define every valid gene value the Phase 6 Genetic Algorithm
# may place into a chromosome.  Import them directly:
#   from core.simulator import HOOK_TYPES, POSTING_HOURS, VISUAL_STYLES, ...
# ===========================================================================

HOOK_TYPES: list[str] = [
    "Question",
    "Bold Statement",
    "Relatable Struggle",
    "Listicle Tease",
    "Before/After",
    "Controversial Take",
    "Story Opener",
    "Stat Shock",
]
"""8 hook archetypes.  The GA evolves toward hook types that maximise
engagement_rate for a given audience segment."""

POSTING_HOURS: list[int] = [6, 7, 8, 9, 11, 12, 13, 17, 18, 19, 20, 21]
"""12 plausible posting hours weighted toward morning commute (7-9),
lunch (12-13), and evening scroll windows (18-21)."""

VISUAL_STYLES: list[str] = [
    "Midnight Violet",
    "Cyberpunk Cyan",
    "Emerald Luxe",
    "Sunset Amber / Gold",
    "Crimson Noir",
]
"""7 visual style categories.  Phase 6 GA optimises visual style per niche."""

CONTENT_TONES: list[str] = [
    "Humorous",
    "Inspirational",
    "Educational",
    "Controversial",
    "Empathetic",
    "Urgent/FOMO",
]
"""6 content tone options.  Tone interacts with hook type to produce
correlated engagement bonuses in simulate_engagement()."""

NICHES: list[str] = [
    "Fitness",
    "Personal Finance",
    "Tech Reviews",
    "Food & Recipes",
    "Travel",
]
"""5 content niches.  Used to label posts and to condition engagement
distribution (some niches have structurally higher engagement ceilings)."""


# ---------------------------------------------------------------------------
# Internal scoring tables  (not exported, used only by simulate_engagement)
# ---------------------------------------------------------------------------

# Additive base impressions per niche (realistic Instagram scale, ~1K-50K)
_NICHE_BASE_IMPRESSIONS: dict[str, int] = {
    "Fitness":          18_000,
    "Personal Finance": 14_000,
    "Tech Reviews":     12_000,
    "Food & Recipes":   22_000,
    "Travel":           20_000,
}

# Hour bonus: peak hours earn extra impressions
_HOUR_BONUS: dict[int, int] = {
    6: 500, 7: 2_000, 8: 2_500, 9: 2_000,
    11: 1_000, 12: 2_500, 13: 2_000,
    17: 1_000, 18: 3_000, 19: 3_500, 20: 3_000, 21: 2_000,
}

# Hook type multiplier on base engagement rate
_HOOK_RATE_BONUS: dict[str, float] = {
    "Question":           0.012,
    "Bold Statement":     0.014,
    "Relatable Struggle": 0.018,
    "Listicle Tease":     0.010,
    "Before/After":       0.016,
    "Controversial Take": 0.020,
    "Story Opener":       0.013,
    "Stat Shock":         0.015,
}

# Tone multiplier on base engagement rate
_TONE_RATE_BONUS: dict[str, float] = {
    "Humorous":       0.010,
    "Inspirational":  0.014,
    "Educational":    0.008,
    "Controversial":  0.018,
    "Empathetic":     0.012,
    "Urgent/FOMO":    0.016,
}

# Comment sentiment pools — 15 templates each, labelled as ground truth
_POSITIVE_COMMENTS: list[str] = [
    "This is exactly what I needed today 🙌",
    "Absolutely love this, saving for later! ❤️",
    "You always deliver such quality content 🔥",
    "This changed my perspective completely, thank you!",
    "Best post I've seen all week no cap 💯",
    "OK this is genuinely helpful, sharing with my friends!",
    "Wow I never thought about it this way before 😮",
    "More of this please! The algorithm needs to push this 🚀",
    "This is so underrated, everyone needs to see it",
    "Literally bookmarked. Coming back to this tomorrow ⭐",
    "You read my mind! I was just thinking about this topic",
    "The production quality on this is incredible 👏",
    "I showed this to my whole family lol, so relatable",
    "This deserves way more engagement fr 💪",
    "New favourite account, instant follow 🙏",
]

_NEGATIVE_COMMENTS: list[str] = [
    "Not really feeling this one tbh",
    "This feels a bit oversimplified to me 🤔",
    "I disagree with pretty much all of this tbh",
    "Did anyone fact-check this? Seems off",
    "Overhyped, this isn't as groundbreaking as people say",
    "This is just recycled content from last year",
    "The advice here could actually be harmful, be careful",
    "Unfollow. This used to be better quality",
    "Way too long for what it's saying",
    "I've seen this exact post 10 times this week 🙄",
    "The editing is really distracting here",
    "This doesn't apply to most people at all",
    "Misinformation alert ⚠️ please do your research",
    "Super clickbaity title, the content doesn't deliver",
    "Hard pass. Not what this account used to be about",
]

_NEUTRAL_COMMENTS: list[str] = [
    "Can you do a follow-up on this?",
    "Interesting, what's the source for that stat?",
    "I've heard both sides on this, not sure what to think",
    "What tool do you use for these graphics?",
    "Tagging a friend who needs to see this",
    "First! 🙋",
    "At what point does this apply to beginners?",
    "Is this sponsored? Genuine question",
    "How long did this take to put together?",
    "What's your take on the other side of this argument?",
    "I tried this last month — mixed results honestly",
    "Have you covered the counter-argument too?",
    "Saving this for when I need motivation",
    "Would this still apply in Europe? Asking for a friend",
    "Just found this account, catching up on the feed now",
]


# ===========================================================================
# Public API
# ===========================================================================

def generate_post(post_id: int) -> dict:
    """
    Create a single synthetic Instagram post record.

    Randomly samples content attributes from the module-level gene lists
    (HOOK_TYPES, POSTING_HOURS, VISUAL_STYLES, CONTENT_TONES, NICHES).
    These same lists are used by the Phase 6 GA as its chromosome gene space,
    so every valid post is also a valid GA chromosome.

    Parameters
    ----------
    post_id : int
        Unique integer identifier for this post (1-based, set by run_simulation).

    Returns
    -------
    dict
        Keys: post_id, niche, hook_type, posting_hour, visual_style,
              content_tone, caption_length, hashtag_count, posted_at.

    Notes
    -----
    - caption_length  : sampled uniformly from [30, 300] words, reflecting
                        real Instagram caption lengths.
    - hashtag_count   : sampled uniformly from [5, 30] hashtags.
    - posted_at       : ISO-8601 timestamp string; the date is spread across
                        the last 365 days from a fixed reference point so the
                        dataset spans a realistic calendar year.
    """
    niche        = str(np.random.choice(NICHES))
    hook_type    = str(np.random.choice(HOOK_TYPES))
    posting_hour = int(np.random.choice(POSTING_HOURS))
    visual_style = str(np.random.choice(VISUAL_STYLES))
    content_tone = str(np.random.choice(CONTENT_TONES))

    caption_length = int(np.random.randint(30, 301))
    hashtag_count  = int(np.random.randint(5, 31))

    # Spread posts across the last 365 days for temporal realism
    reference_date = datetime.date(2025, 8, 18)
    days_offset    = int(np.random.randint(0, 365))
    post_date      = reference_date - datetime.timedelta(days=days_offset)
    posted_at      = datetime.datetime(
        post_date.year, post_date.month, post_date.day,
        posting_hour, int(np.random.randint(0, 60)), 0
    ).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "post_id":        post_id,
        "niche":          niche,
        "hook_type":      hook_type,
        "posting_hour":   posting_hour,
        "visual_style":   visual_style,
        "content_tone":   content_tone,
        "caption_length": caption_length,
        "hashtag_count":  hashtag_count,
        "posted_at":      posted_at,
    }


def simulate_engagement(post: dict) -> dict:
    """
    Compute realistic, correlated engagement metrics for a single post.

    Uses a structured additive model:
        impressions = niche_base + hour_bonus + gaussian_noise
        base_rate   = 0.03 + hook_bonus + tone_bonus + gaussian_noise
        engagement  = clip(base_rate * impressions * metric_fractions)

    The correlations (hook type → rate, posting hour → impressions, tone →
    rate) are intentional so that Phase 4's K-Means clustering discovers
    meaningful, separable audience segments rather than random blobs, and
    so Phase 5's Fuzzy Engine has a signal-rich engagement_rate to reason over.

    Parameters
    ----------
    post : dict
        A record returned by generate_post().

    Returns
    -------
    dict
        Keys: post_id, impressions, likes, comments_count, shares, saves,
              engagement_rate.
        - All count metrics are non-negative integers.
        - engagement_rate is a float in [0, 1], capped at 0.95 for realism.

    Notes
    -----
    Metric breakdown (realistic Instagram ratios):
        likes         ≈ 70-80 % of total engagement actions
        comments      ≈  8-12 %
        shares        ≈  5-10 %
        saves         ≈ 10-15 %
    """
    niche        = post["niche"]
    hook_type    = post["hook_type"]
    posting_hour = post["posting_hour"]
    content_tone = post["content_tone"]

    # --- Impressions ----------------------------------------------------------
    base_impressions = _NICHE_BASE_IMPRESSIONS.get(niche, 15_000)
    hour_bonus       = _HOUR_BONUS.get(posting_hour, 0)
    impression_noise = int(np.random.normal(0, 2_000))
    impressions      = max(500, base_impressions + hour_bonus + impression_noise)

    # --- Base engagement rate -------------------------------------------------
    hook_bonus  = _HOOK_RATE_BONUS.get(hook_type, 0.010)
    tone_bonus  = _TONE_RATE_BONUS.get(content_tone, 0.010)
    rate_noise  = float(np.random.normal(0, 0.005))
    base_rate   = float(np.clip(0.03 + hook_bonus + tone_bonus + rate_noise,
                                0.005, 0.95))

    # Total engagement actions implied by the rate
    total_actions = int(round(base_rate * impressions))
    total_actions = max(0, total_actions)

    # --- Distribute actions across metric types (Dirichlet split) -------------
    # Weights: likes 75%, comments 10%, shares 7%, saves 8%
    fractions = np.random.dirichlet([7.5, 1.0, 0.7, 0.8])
    likes          = max(0, int(round(fractions[0] * total_actions)))
    comments_count = max(0, int(round(fractions[1] * total_actions)))
    shares         = max(0, int(round(fractions[2] * total_actions)))
    saves          = max(0, int(round(fractions[3] * total_actions)))

    # Recompute engagement_rate from actual counts for internal consistency
    actual_actions  = likes + comments_count + shares + saves
    engagement_rate = float(np.clip(actual_actions / impressions, 0.0, 0.95)) \
                      if impressions > 0 else 0.0

    return {
        "post_id":         post["post_id"],
        "impressions":     impressions,
        "likes":           likes,
        "comments_count":  comments_count,
        "shares":          shares,
        "saves":           saves,
        "engagement_rate": round(engagement_rate, 6),
    }


def generate_synthetic_comments(post_id: int, engagement: dict) -> list[dict]:
    """
    Generate a variable number of labelled synthetic comments for one post.

    The number of comments is drawn uniformly from
    config.COMMENTS_PER_POST_RANGE = (5, 40).

    Sentiment distribution is **correlated with engagement_rate** so the
    dataset is internally consistent:
        - High engagement_rate  → higher share of positive comments
        - Low engagement_rate   → more negative comments creep in

    This ground-truth label (true_sentiment_label) becomes the training
    signal for Phase 3's Naive Bayes sentiment classifier.

    Parameters
    ----------
    post_id    : int   — post identifier, forwarded into each comment record.
    engagement : dict  — record from simulate_engagement(); uses engagement_rate
                         to skew sentiment proportions.

    Returns
    -------
    list[dict]
        Each element has keys:
            comment_id         : str  — unique, e.g. "42_3"
            post_id            : int
            comment_text       : str  — realistic Instagram-style sentence
            true_sentiment_label : str — "positive" | "negative" | "neutral"

    Notes
    -----
    The comment_text strings are drawn from three distinct pools
    (_POSITIVE_COMMENTS, _NEGATIVE_COMMENTS, _NEUTRAL_COMMENTS), each
    with 15 templates, giving 45 unique templates total.  The 'true_'
    prefix on the label column signals to Phase 3 that this is ground truth
    (not a model prediction).
    """
    low, high = config.COMMENTS_PER_POST_RANGE          # (5, 40)
    n_comments = int(np.random.randint(low, high + 1))  # inclusive upper bound

    engagement_rate = engagement.get("engagement_rate", 0.05)

    # Skew sentiment proportions based on engagement_rate.
    # engagement_rate is typically 0.03–0.20; normalise into [0,1].
    # At rate=0.03 → pos=0.40, neu=0.35, neg=0.25
    # At rate=0.20 → pos=0.65, neu=0.25, neg=0.10
    normalised = float(np.clip((engagement_rate - 0.03) / 0.20, 0.0, 1.0))
    p_positive  = 0.40 + 0.25 * normalised   # 0.40 → 0.65
    p_negative  = 0.25 - 0.15 * normalised   # 0.25 → 0.10
    p_neutral   = max(0.05, 1.0 - p_positive - p_negative)

    # Renormalise so probabilities sum to exactly 1
    total_p = p_positive + p_negative + p_neutral
    probs = [p_positive / total_p, p_negative / total_p, p_neutral / total_p]

    sentiment_choices = np.random.choice(
        ["positive", "negative", "neutral"],
        size=n_comments,
        p=probs,
    )

    comments = []
    for i, sentiment in enumerate(sentiment_choices):
        if sentiment == "positive":
            text = str(np.random.choice(_POSITIVE_COMMENTS))
        elif sentiment == "negative":
            text = str(np.random.choice(_NEGATIVE_COMMENTS))
        else:
            text = str(np.random.choice(_NEUTRAL_COMMENTS))

        comments.append({
            "comment_id":           f"{post_id}_{i}",
            "post_id":              post_id,
            "comment_text":         text,
            "true_sentiment_label": sentiment,
        })

    return comments


def run_simulation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full simulation for config.NUM_POSTS posts.

    For each post the function:
      1. Calls generate_post()       → one post record
      2. Calls simulate_engagement() → one engagement record
      3. Calls generate_synthetic_comments() → list of comment records

    Then assembles three DataFrames, saves them to config.DATA_DIR as:
      - posts.csv
      - engagement.csv
      - comments.csv

    A summary is printed to stdout covering:
      - Total posts generated
      - Total comments generated
      - Mean engagement rate (mean of engagement_df['engagement_rate'])
      - Sentiment class balance: count of positive/negative/neutral labels

    Parameters
    ----------
    None  — all parameters are read from config.py.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (posts_df, engagement_df, comments_df)

    Notes
    -----
    This function is idempotent — re-running it with the same RANDOM_SEED
    (set at module level) produces identical CSVs.  Data is saved to
    config.DATA_DIR (auto-created by config.py on import).
    """
    all_posts      : list[dict] = []
    all_engagement : list[dict] = []
    all_comments   : list[dict] = []

    for post_id in range(1, config.NUM_POSTS + 1):
        post       = generate_post(post_id)
        engagement = simulate_engagement(post)
        comments   = generate_synthetic_comments(post_id, engagement)

        all_posts.append(post)
        all_engagement.append(engagement)
        all_comments.extend(comments)

    # --- Build DataFrames -----------------------------------------------------
    posts_df      = pd.DataFrame(all_posts)
    engagement_df = pd.DataFrame(all_engagement)
    comments_df   = pd.DataFrame(all_comments)

    # --- Save to CSV ----------------------------------------------------------
    posts_path      = os.path.join(config.DATA_DIR, "posts.csv")
    engagement_path = os.path.join(config.DATA_DIR, "engagement.csv")
    comments_path   = os.path.join(config.DATA_DIR, "comments.csv")

    posts_df.to_csv(posts_path,      index=False)
    engagement_df.to_csv(engagement_path, index=False)
    comments_df.to_csv(comments_path,    index=False)

    # --- Console summary ------------------------------------------------------
    sentiment_counts = comments_df["true_sentiment_label"].value_counts()
    mean_eng_rate    = engagement_df["engagement_rate"].mean()

    print("=" * 60)
    print("  PHASE 2 — SIMULATION COMPLETE")
    print("=" * 60)
    print(f"  Total posts generated      : {len(posts_df)}")
    print(f"  Total comments generated   : {len(comments_df)}")
    print(f"  Mean engagement rate       : {mean_eng_rate:.4f}  "
          f"({mean_eng_rate * 100:.2f}%)")
    print()
    print("  Sentiment class balance (comments):")
    for label in ["positive", "negative", "neutral"]:
        count = int(sentiment_counts.get(label, 0))
        pct   = count / len(comments_df) * 100 if len(comments_df) > 0 else 0
        print(f"    {label:<10}: {count:>5}  ({pct:.1f}%)")
    print()
    print(f"  CSVs saved to: {config.DATA_DIR}")
    print(f"    posts.csv      -> {len(posts_df)} rows")
    print(f"    engagement.csv -> {len(engagement_df)} rows")
    print(f"    comments.csv   -> {len(comments_df)} rows")
    print("=" * 60)

    return posts_df, engagement_df, comments_df


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    run_simulation()
