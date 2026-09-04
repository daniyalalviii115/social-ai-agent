"""
core/test_upgrade_modules.py — Phase 8/9 Validation Tests
===================================================
Tests TrendRadar, PivotManager, and ReelGenerator initialization.
"""

import os
import sys

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import config
from core.trend_radar import TrendRadar
from decision.pivot_manager import PivotManager
from content.video_generator import ReelGenerator

def run_tests() -> None:
    # 1. Test TrendRadar
    radar = TrendRadar()
    trends = radar.get_trends("Fitness")
    assert isinstance(trends, list), "TrendRadar output must be a list"
    assert len(trends) > 0, "TrendRadar must return at least one topic (fallback or live)"
    
    # 2. Test PivotManager
    posts_df = pd.read_csv(os.path.join(config.DATA_DIR, "posts.csv"))
    engagement_df = pd.read_csv(os.path.join(config.DATA_DIR, "engagement.csv"))
    manager = PivotManager(posts_df=posts_df, engagement_df=engagement_df)
    
    demo_individual = {
        "hook_type": "Question",
        "posting_hour": 20,
        "visual_style": "Bold Colorful",
        "content_tone": "Inspirational",
    }
    
    decision = manager.evaluate(
        niche="Tech Reviews",
        performance_tier="Flop",
        avg_sentiment_score=-0.4,
        current_individual=demo_individual,
    )
    assert isinstance(decision, dict), "PivotManager evaluate must return a dict"
    assert "level" in decision, "PivotManager decision must have 'level'"
    assert "new_individual" in decision, "PivotManager decision must have 'new_individual'"
    
    # 3. ReelGenerator instantiation
    generator = ReelGenerator()
    assert generator.width == 1080 and generator.height == 1920, "ReelGenerator must default to 1080x1920"

    print("=" * 60)
    print("  ALL UPGRADE MODULES VALIDATION TESTS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
