"""
Autonomous Social Media AI Agent & Strategy Optimization Dashboard
Master CLI Orchestrator (Phase 7, updated in Phase 3 of the Instagram
Autonomous AI Manager upgrade to add --generate-reel).

Runs the full 6-stage pipeline end-to-end, launches the dashboard,
runs the master test suite, or generates a single autonomous 9:16
video Reel around a live trending topic — depending on the CLI flag passed.
"""

import os
import sys
import argparse
import random
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import core.simulator as sim
from ml.naive_bayes import train_and_save_model
from ml.kmeans_cluster import cluster_and_save
from decision.fuzzy_engine import StrategyFuzzyEngine, evaluate_all_posts
from decision.genetic_optimizer import optimize_strategy
from content.llm_generator import ContentGenerator
from content.banner_renderer import BannerRenderer
from core.trend_radar import TrendRadar
from content.video_generator import ReelGenerator


def run_full_pipeline():
    """
    Executes the complete autonomous pipeline in sequence:
    simulate -> sentiment -> cluster -> fuzzy -> GA -> content+banner.
    Mirrors exactly what the Streamlit dashboard's "Generate New Post"
    button does, but for the full historical dataset in one CLI run.
    """
    print("=" * 70)
    print("🚀 RUNNING AUTONOMOUS SOCIAL MEDIA AI AGENT — FULL LIFECYCLE")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Simulator (Phase 2) — run_simulation() takes no args,
    #    reads config.NUM_POSTS internally, returns 3 DataFrames.
    # ---------------------------------------------------------------
    print("\n[Stage 1/6] Running Data Simulation Engine...")
    t0 = time.time()
    posts_df, engagement_df, comments_df = sim.run_simulation()
    print(
        f"  ✓ Generated {len(posts_df)} posts, {len(engagement_df)} "
        f"engagement records, {len(comments_df)} comments "
        f"in {time.time() - t0:.2f}s"
    )

    # ---------------------------------------------------------------
    # 2. Sentiment Analyzer (Phase 3) — train_and_save_model returns
    #    a single fitted SentimentAnalyzer instance, not a tuple.
    # ---------------------------------------------------------------
    print("\n[Stage 2/6] Training Sentiment Analyzer (Naive Bayes)...")
    t0 = time.time()
    sentiment_analyzer = train_and_save_model(comments_df)
    print(
        f"  ✓ Model trained on {len(comments_df)} comments "
        f"in {time.time() - t0:.2f}s"
    )

    # ---------------------------------------------------------------
    # 3. K-Means Clustering (Phase 3) — cluster_and_save returns
    #    (clusterer, tiered_df).
    # ---------------------------------------------------------------
    print("\n[Stage 3/6] Fitting K-Means Performance Clusterer (k=3)...")
    t0 = time.time()
    clusterer, tiered_df = cluster_and_save(engagement_df)
    tier_counts = dict(tiered_df["performance_tier"].value_counts())
    print(
        f"  ✓ Fitted clusters. Tier counts: {tier_counts} "
        f"in {time.time() - t0:.2f}s"
    )

    # ---------------------------------------------------------------
    # 4. Fuzzy Logic Decision Engine (Phase 4) — evaluate_all_posts
    #    signature: (sentiment_analyzer, clusterer, posts_df,
    #    engagement_tiered_df, comments_df).
    # ---------------------------------------------------------------
    print("\n[Stage 4/6] Evaluating Posts via Mamdani Fuzzy Inference Engine...")
    t0 = time.time()
    fuzzy_results_df = evaluate_all_posts(
        sentiment_analyzer, clusterer, posts_df, tiered_df, comments_df
    )
    pivots_count = int(fuzzy_results_df["trigger_pivot"].sum())
    print(
        f"  ✓ Fuzzy evaluation complete. "
        f"Mean Shift Rate: {fuzzy_results_df['shift_rate'].mean():.4f} | "
        f"Pivot Triggers: {pivots_count}/{len(fuzzy_results_df)} "
        f"in {time.time() - t0:.2f}s"
    )

    # ---------------------------------------------------------------
    # 5. Genetic Algorithm Optimization (Phase 4)
    # ---------------------------------------------------------------
    print("\n[Stage 5/6] Running Genetic Algorithm Content Parameter Optimization...")
    t0 = time.time()
    ga_results = optimize_strategy(posts_df=posts_df, engagement_df=engagement_df)
    best_ind = ga_results["best_individual"]
    print(
        f"  ✓ GA Evolution Complete. "
        f"Best Fitness: {ga_results['best_fitness']:.6f} "
        f"in {time.time() - t0:.2f}s"
    )
    print(f"    Optimal Chromosome: {best_ind}")

    # ---------------------------------------------------------------
    # 6. Content Generation & Banner Rendering (Phase 5)
    # ---------------------------------------------------------------
    print("\n[Stage 6/6] Generating Autonomous Content & Rendering Banner...")
    t0 = time.time()
    generator = ContentGenerator()
    content = generator.generate_from_ga_result(best_ind)

    renderer = BannerRenderer()
    banner_save_path = os.path.join(config.OUTPUT_DIR, "final_optimized_banner.png")
    final_img_path = renderer.render_from_content_dict(content, save_path=banner_save_path)

    print(f"  ✓ Content generated (Source: {content.get('source', 'unknown')})")
    print(f"    Hook: {content.get('hook')}")
    print(
        f"    Hashtags ({len(content.get('hashtags', []))}): "
        f"{' '.join(content.get('hashtags', []))}"
    )
    print(f"  ✓ Rendered 1080x1080 graphic banner to: {final_img_path} in {time.time() - t0:.2f}s")

    print("\n" + "=" * 70)
    print("✅ FULL AUTONOMOUS PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    return {
        "posts_df": posts_df,
        "engagement_df": engagement_df,
        "comments_df": comments_df,
        "tiered_df": tiered_df,
        "fuzzy_results_df": fuzzy_results_df,
        "ga_results": ga_results,
        "content": content,
        "banner_path": final_img_path,
    }


def run_generate_reel(niche: str = None) -> dict:
    """
    Generates a single autonomous 9:16 video Reel end-to-end:
    pick a niche -> fetch a live trending topic (TrendRadar) ->
    evolve optimal content parameters (GA) -> write a topic-aware
    Reel script (ContentGenerator.generate_script) -> render the
    final MP4 (ReelGenerator).

    Parameters
    ----------
    niche : str, optional — one of core.simulator.NICHES. Random if omitted.

    Returns
    -------
    dict with keys: niche, live_topic, best_individual, script, reel_path
    """
    print("=" * 70)
    print("🎬 GENERATING AUTONOMOUS VIDEO REEL")
    print("=" * 70)

    if niche is None:
        niche = random.choice(sim.NICHES)
    print(f"\n[Step 1/4] Niche selected: {niche}")

    print("\n[Step 2/4] Fetching live trending topic (Trend Radar)...")
    t0 = time.time()
    trend_radar = TrendRadar()
    live_topic = trend_radar.get_next_topic(niche)
    print(f"  ✓ Live topic: '{live_topic}' in {time.time() - t0:.2f}s")

    print("\n[Step 3/4] Evolving optimal content parameters (Genetic Algorithm)...")
    t0 = time.time()
    ga_results = optimize_strategy()
    best_ind = ga_results["best_individual"]
    print(f"  ✓ Best chromosome: {best_ind} in {time.time() - t0:.2f}s")

    print("\n[Step 4/4] Writing Reel script & rendering 9:16 MP4...")
    t0 = time.time()
    generator = ContentGenerator()
    script = generator.generate_script(
        best_individual=best_ind,
        live_topic=live_topic,
        script_type="reel",
        niche=niche,
    )
    print(f"  ✓ Script generated (Source: {script.get('source', 'unknown')})")
    print(f"    Hook: {script.get('hook_headline')}")

    reel_generator = ReelGenerator()
    reel_path = reel_generator.render_reel(script)
    print(f"  ✓ Reel rendered to: {reel_path} in {time.time() - t0:.2f}s")

    print("\n" + "=" * 70)
    print("✅ AUTONOMOUS VIDEO REEL GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    return {
        "niche": niche,
        "live_topic": live_topic,
        "best_individual": best_ind,
        "script": script,
        "reel_path": reel_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Social Media AI Agent Master Orchestrator"
    )
    parser.add_argument(
        "--full-run", action="store_true",
        help="Execute the complete 6-stage pipeline end-to-end"
    )
    parser.add_argument(
        "--launch-dashboard", action="store_true",
        help="Launch the Streamlit dashboard"
    )
    parser.add_argument(
        "--generate-reel", action="store_true",
        help="Generate a single autonomous 9:16 video Reel around a live trending topic"
    )
    parser.add_argument(
        "--niche", type=str, default=None,
        help="Niche to use with --generate-reel (random from core.simulator.NICHES if omitted)"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run the master test suite across all phases"
    )
    args = parser.parse_args()

    if args.full_run:
        run_full_pipeline()
    elif args.launch_dashboard:
        os.system(f'"{sys.executable}" -m streamlit run dashboard/app.py')
    elif args.generate_reel:
        run_generate_reel(niche=args.niche)
    elif args.test:
        os.system(f'"{sys.executable}" run_all_tests.py')
    else:
        # Default behavior: run the full pipeline
        run_full_pipeline()


if __name__ == "__main__":
    main()