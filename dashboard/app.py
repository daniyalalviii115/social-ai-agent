import sys
import os
import time
import random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageDraw

# Ensure project root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.simulator import (
    HOOK_TYPES, POSTING_HOURS, VISUAL_STYLES, CONTENT_TONES, NICHES,
    generate_post, simulate_engagement, generate_synthetic_comments
)
from ml.naive_bayes import train_and_save_model
from ml.kmeans_cluster import cluster_and_save
from decision.fuzzy_engine import StrategyFuzzyEngine
from decision.genetic_optimizer import optimize_strategy
from content.llm_generator import ContentGenerator
from content.banner_renderer import BannerRenderer, _load_font
from content.video_generator import ReelGenerator
from core.trend_radar import TrendRadar

st.set_page_config(page_title="Autonomous Social Media AI Agent", layout="wide", page_icon="📊")


# ===========================================================================
# Cached data / model loading
# ===========================================================================
@st.cache_resource
def load_models_and_data():
    posts_df = pd.read_csv(os.path.join(config.DATA_DIR, "posts.csv"))
    engagement_df = pd.read_csv(os.path.join(config.DATA_DIR, "engagement.csv"))
    engagement_tiered_df = pd.read_csv(os.path.join(config.DATA_DIR, "engagement_tiered.csv"))
    comments_df = pd.read_csv(os.path.join(config.DATA_DIR, "comments.csv"))
    fuzzy_results_df = pd.read_csv(os.path.join(config.DATA_DIR, "fuzzy_results.csv"))

    sentiment_res = train_and_save_model(comments_df)
    sentiment_analyzer = sentiment_res[0] if isinstance(sentiment_res, tuple) else sentiment_res

    cluster_res = cluster_and_save(engagement_df)
    clusterer = cluster_res[0] if isinstance(cluster_res, tuple) else clusterer

    fuzzy_engine = StrategyFuzzyEngine()

    return (posts_df, engagement_df, engagement_tiered_df, comments_df, fuzzy_results_df,
            sentiment_analyzer, clusterer, fuzzy_engine)


with st.spinner("Loading models and data..."):
    (posts_df, engagement_df, engagement_tiered_df, comments_df, fuzzy_results_df,
     sentiment_analyzer, clusterer, fuzzy_engine) = load_models_and_data()


# ===========================================================================
# Shared helper functions
# ===========================================================================

_TIER_ORDER = ["Flop", "Average", "Viral"]

_AUDIO_POOL: dict[str, list[dict]] = {
    "Fitness": [
        {"track": "High-energy trap workout beat", "vibe": "Aggressive / motivating", "tempo": "140-150 BPM"},
        {"track": "Uplifting EDM gym anthem", "vibe": "Euphoric / driving", "tempo": "128 BPM"},
        {"track": "Bass-heavy hip-hop instrumental", "vibe": "Confident / gritty", "tempo": "95 BPM"},
    ],
    "Personal Finance": [
        {"track": "Minimal corporate piano loop", "vibe": "Trustworthy / calm", "tempo": "90 BPM"},
        {"track": "Lo-fi focus beat", "vibe": "Thoughtful / productive", "tempo": "80 BPM"},
        {"track": "Subtle synth build-up", "vibe": "Aspirational / clean", "tempo": "100 BPM"},
    ],
    "Tech Reviews": [
        {"track": "Futuristic synthwave loop", "vibe": "Sleek / modern", "tempo": "110 BPM"},
        {"track": "Clean electronic pulse beat", "vibe": "Precise / crisp", "tempo": "120 BPM"},
        {"track": "Ambient tech background pad", "vibe": "Calm / premium", "tempo": "85 BPM"},
    ],
    "Food & Recipes": [
        {"track": "Warm acoustic guitar loop", "vibe": "Cozy / inviting", "tempo": "95 BPM"},
        {"track": "Upbeat indie-pop snippet", "vibe": "Playful / fresh", "tempo": "115 BPM"},
        {"track": "Jazzy lo-fi kitchen beat", "vibe": "Relaxed / homely", "tempo": "88 BPM"},
    ],
    "Travel": [
        {"track": "Tropical house summer loop", "vibe": "Free / adventurous", "tempo": "118 BPM"},
        {"track": "Cinematic acoustic build", "vibe": "Epic / wanderlust", "tempo": "100 BPM"},
        {"track": "Chill world-fusion beat", "vibe": "Dreamy / exploratory", "tempo": "92 BPM"},
    ],
    "_default": [
        {"track": "Trending upbeat pop loop", "vibe": "Broadly energetic", "tempo": "112 BPM"},
        {"track": "Feel-good acoustic snippet", "vibe": "Warm / relatable", "tempo": "96 BPM"},
        {"track": "Modern minimal beat", "vibe": "Clean / versatile", "tempo": "104 BPM"},
    ],
}


def get_audio_suggestions(niche: str) -> list[dict]:
    return _AUDIO_POOL.get(niche, _AUDIO_POOL["_default"])


def compute_tier_summary(tiered_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        tiered_df.groupby("performance_tier")["engagement_rate"]
        .agg(post_count="count", mean_engagement_rate="mean")
        .reindex(_TIER_ORDER)
        .fillna(0)
    )
    summary["mean_engagement_rate"] = (summary["mean_engagement_rate"] * 100).round(2)
    summary["post_count"] = summary["post_count"].astype(int)
    summary = summary.rename(columns={
        "post_count": "Post Count",
        "mean_engagement_rate": "Mean Engagement Rate (%)",
    })
    summary.index.name = "Performance Tier"
    return summary.reset_index()


def render_threshold_gauge(shift_rate: float, threshold: float = 0.55) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 1.6))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D27")

    bar_color = "#E74C3C" if shift_rate >= threshold else "#7C83FD"
    ax.barh([0], [shift_rate], color=bar_color, height=0.5, zorder=3)
    ax.barh([0], [1.0], color="#22252F", height=0.5, zorder=1)

    ax.axvline(x=threshold, color="#F39C12", lw=2, linestyle="--", zorder=4,
               label=f"Pivot threshold: {threshold}")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, threshold, 0.75, 1.0])
    ax.tick_params(colors="#AAAAAA", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(shift_rate, 0.42, f"{shift_rate:.3f}", color="white",
            fontsize=10, ha="center", fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.35), frameon=False,
              labelcolor="#AAAAAA", fontsize=8, ncol=1)

    plt.tight_layout()
    return fig


def generate_weekly_schedule(peak_hour: int = 21) -> pd.DataFrame:
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    rows = []
    for day in days:
        rows.append({"Day": day, "Time": f"{peak_hour:02d}:00", "Type": "📸 Feed Post",
                     "Suggestion": "GA-optimized peak-hour feed post"})
        rows.append({"Day": day, "Time": "13:00", "Type": "📊 Story — Poll",
                     "Suggestion": "Quick engagement poll tied to today's topic"})
        rows.append({"Day": day, "Time": "18:00", "Type": "🎬 Story — BTS",
                     "Suggestion": "Behind-the-scenes / process clip"})
    return pd.DataFrame(rows)

def schedule_approved_content(format_type: str, content_hook: str, file_path: str, niche: str, default_peak_hour: int = 21):
    if "weekly_schedule" not in st.session_state:
        st.session_state["weekly_schedule"] = generate_weekly_schedule(default_peak_hour)
        
    df = st.session_state["weekly_schedule"]
    
    # Determine which type to look for in the schedule
    type_str = "📸 Feed Post" if "Post" in format_type else "Story"
    
    # Find next available slot that isn't already a Priority
    mask = (df["Type"].str.contains(type_str)) & (~df["Suggestion"].str.contains("⭐ Priority"))
    available_indices = df[mask].index
    
    if len(available_indices) > 0:
        idx = available_indices[0]
        day = df.at[idx, "Day"]
        time_slot = df.at[idx, "Time"]
        target_type = df.at[idx, "Type"]
        
        # Update the schedule dataframe
        df.at[idx, "Suggestion"] = f"{content_hook} (⭐ Priority)"
        st.session_state["weekly_schedule"] = df
        
        # Add to approved queue
        if "cs_approved" not in st.session_state:
            st.session_state["cs_approved"] = []
            
        st.session_state["cs_approved"].append({
            "Type": format_type,
            "Niche": niche,
            "Topic / Hook": content_hook,
            "Scheduled Slot": f"{day} at {time_slot}",
            "File Path": file_path
        })
        
        return day, time_slot, target_type
    else:
        return None, None, None


def overlay_hook_on_image(pil_img: Image.Image, hook_text: str) -> Image.Image:
    img = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    font_size = max(28, int(w * 0.06))
    font = _load_font(font_size, bold=True)

    strip_h = int(font_size * 2.4)
    overlay = Image.new("RGBA", (w, strip_h), (0, 0, 0, 160))
    img.paste(Image.alpha_composite(
        Image.new("RGBA", (w, strip_h), (0, 0, 0, 0)), overlay
    ).convert("RGB"), (0, h - strip_h))

    display_text = hook_text if len(hook_text) <= 90 else hook_text[:87] + "..."
    try:
        text_w = draw.textlength(display_text, font=font)
    except Exception:
        bbox = draw.textbbox((0, 0), display_text, font=font)
        text_w = bbox[2] - bbox[0]

    x = max(20, (w - text_w) / 2)
    y = h - strip_h + (strip_h - font_size) / 2
    draw.text((x, y), display_text, font=font, fill=(255, 255, 255))
    return img


def analyze_uploaded_image(pil_img: Image.Image) -> str:
    small = pil_img.convert("RGB").resize((60, 60))
    pixels = np.array(small).reshape(-1, 3)
    avg_r, avg_g, avg_b = pixels.mean(axis=0)
    brightness = (avg_r + avg_g + avg_b) / 3
    warmth = avg_r - avg_b

    brightness_desc = "bright and airy" if brightness > 150 else (
        "moody and dim" if brightness < 90 else "balanced in lighting")
    warmth_desc = "warm-toned (reds/oranges)" if warmth > 15 else (
        "cool-toned (blues)" if warmth < -15 else "neutral-toned")

    return (
        f"The uploaded image is {brightness_desc} and {warmth_desc}. "
        f"Write the hook and caption so the tone visually matches this image."
    )


# ===========================================================================
# Sidebar navigation
# ===========================================================================
st.sidebar.title("Agent Dashboard")
st.sidebar.write("Monitor and control the Autonomous Social Media AI Agent's decisions, clusters, and content pipeline.")
page = st.sidebar.radio("Navigation", [
    "📊 Overview",
    "🎯 Cluster Analysis",
    "🧠 Fuzzy & GA Engine",
    "🎨 Content Studio",
    "📈 Instagram Manager",
    "📱 Live Feed Simulation",
])


# ===========================================================================
# PAGE: Overview
# ===========================================================================
if page == "📊 Overview":
    st.header("📊 System Overview")

    merged_df = posts_df.merge(engagement_tiered_df, on="post_id").merge(
        fuzzy_results_df.drop(columns=["performance_tier"], errors="ignore"), on="post_id"
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    total_posts = len(merged_df)
    mean_eng_rate = merged_df["engagement_rate"].mean() * 100
    mean_sentiment = merged_df["avg_sentiment_score"].mean()
    pivot_count = merged_df["trigger_pivot"].sum()
    pct_viral = (merged_df["performance_tier"] == "Viral").mean() * 100

    col1.metric("Total Posts", total_posts)
    col2.metric("Mean Engagement", f"{mean_eng_rate:.2f}%")
    col3.metric("Mean Sentiment", f"{mean_sentiment:.2f}")
    col4.metric("Pivot Triggers", int(pivot_count))
    col5.metric("Viral %", f"{pct_viral:.1f}%")

    st.subheader("Historical Data")
    display_cols = ["post_id", "niche", "hook_type", "performance_tier", "engagement_rate",
                    "avg_sentiment_score", "shift_rate", "trigger_pivot"]
    st.dataframe(merged_df[display_cols], use_container_width=True)


# ===========================================================================
# PAGE: Cluster Analysis
# ===========================================================================
elif page == "🎯 Cluster Analysis":
    st.header("🎯 Cluster Analysis")
    if os.path.exists(os.path.join(config.OUTPUT_DIR, "cluster_scatter.png")):
        st.image(os.path.join(config.OUTPUT_DIR, "cluster_scatter.png"), use_column_width=True)

    st.subheader("Tier Summary — Post Count & Mean Engagement Rate")
    tier_summary = compute_tier_summary(engagement_tiered_df)
    t1, t2, t3 = st.columns(3)
    tier_cols = {"Flop": t1, "Average": t2, "Viral": t3}
    tier_icons = {"Flop": "📉", "Average": "➖", "Viral": "🚀"}
    for _, row in tier_summary.iterrows():
        tier = row["Performance Tier"]
        col = tier_cols.get(tier)
        if col is not None:
            col.metric(f"{tier_icons.get(tier, '')} {tier}",
                       f"{int(row['Post Count'])} posts",
                       f"{row['Mean Engagement Rate (%)']:.2f}% avg")
    st.dataframe(tier_summary, use_container_width=True, hide_index=True)

    st.subheader("Cluster Summary (Model)")
    st.dataframe(clusterer.get_cluster_summary(), use_container_width=True)

    st.subheader("Predict Post Tier")
    c1, c2, c3, c4, c5 = st.columns(5)
    impressions = c1.number_input("Impressions", 1000, 100000, 10000)
    likes = c2.number_input("Likes", 0, 10000, 500)
    comments_count = c3.number_input("Comments", 0, 1000, 50)
    shares = c4.number_input("Shares", 0, 1000, 20)
    saves = c5.number_input("Saves", 0, 1000, 30)

    if st.button("Predict Tier"):
        test_row = {
            "impressions": impressions,
            "likes": likes,
            "comments_count": comments_count,
            "shares": shares,
            "saves": saves,
            "engagement_rate": (likes + comments_count + shares + saves) / impressions
        }
        tier = clusterer.predict_tier(test_row)
        if tier == "Viral":
            st.success(f"Predicted Tier: {tier}")
        elif tier == "Average":
            st.info(f"Predicted Tier: {tier}")
        else:
            st.warning(f"Predicted Tier: {tier}")


# ===========================================================================
# PAGE: Fuzzy & GA Engine
# ===========================================================================
elif page == "🧠 Fuzzy & GA Engine":
    st.header("🧠 Fuzzy & GA Engine")

    st.subheader("Fuzzy Logic Shift Rate")
    if os.path.exists(os.path.join(config.OUTPUT_DIR, "fuzzy_membership.png")):
        st.image(os.path.join(config.OUTPUT_DIR, "fuzzy_membership.png"), use_column_width=True)
    c1, c2 = st.columns(2)
    sentiment_val = c1.slider("Sentiment Score", -1.0, 1.0, 0.0, 0.01)
    tier_val = c2.selectbox("Performance Tier", ["Flop", "Average", "Viral"], index=1)

    if st.button("Compute Shift Rate"):
        res = fuzzy_engine.compute_shift_rate(sentiment_val, tier_val)
        st.session_state["last_shift_result"] = res

    if "last_shift_result" in st.session_state:
        res = st.session_state["last_shift_result"]
        st.metric("Strategy Shift Rate", f"{res['shift_rate']:.3f}")
        st.pyplot(render_threshold_gauge(res["shift_rate"], threshold=0.55))
        if res["trigger_pivot"]:
            st.error("PIVOT TRIGGERED — shift rate crossed the 0.55 threshold")
        else:
            st.success("STRATEGY STABLE — shift rate below the 0.55 threshold")

    st.divider()
    st.subheader("Genetic Algorithm Optimization")
    if os.path.exists(os.path.join(config.OUTPUT_DIR, "ga_convergence.png")):
        st.image(os.path.join(config.OUTPUT_DIR, "ga_convergence.png"), use_column_width=True)

    if st.button("Run Genetic Algorithm Now"):
        with st.spinner("Evolving best strategy over 40 generations..."):
            ga_res = optimize_strategy(posts_df, engagement_df)
        st.session_state["last_ga_result"] = ga_res
        st.session_state["cs_chromosome"] = ga_res["best_individual"]

    if "last_ga_result" in st.session_state:
        ga_res = st.session_state["last_ga_result"]
        st.success("Optimization Complete!")

        st.markdown("**Interactive Convergence Chart (all generations)**")
        st.line_chart(pd.DataFrame({"Best Fitness": ga_res["fitness_history"]}))

        baseline = float(engagement_df["engagement_rate"].mean())
        optimized = float(ga_res["best_fitness"])
        lift_pct = ((optimized - baseline) / baseline * 100) if baseline > 0 else 0.0

        st.markdown("**Baseline Performance vs. GA-Optimized Expected Lift**")
        b1, b2, b3 = st.columns(3)
        b1.metric("Baseline Engagement Rate", f"{baseline * 100:.2f}%")
        b2.metric("GA-Optimized Fitness", f"{optimized * 100:.2f}%")
        b3.metric("Expected Lift", f"{lift_pct:+.1f}%")

        st.metric("Best Fitness (raw)", f"{ga_res['best_fitness']:.4f}")
        st.json(ga_res["best_individual"])


# ===========================================================================
# PAGE: Content Studio
# ===========================================================================
elif page == "🎨 Content Studio":
    st.header("🎨 Content Studio")
    st.caption("Review and fine-tune the AI agent's current autonomous content pick before scheduling it.")

    content_gen = ContentGenerator()
    banner_ren = BannerRenderer()

    if "cs_chromosome" not in st.session_state:
        st.session_state["cs_chromosome"] = None
    if "cs_content" not in st.session_state:
        st.session_state["cs_content"] = None
    if "cs_banner_path" not in st.session_state:
        st.session_state["cs_banner_path"] = None
    if "cs_niche" not in st.session_state:
        st.session_state["cs_niche"] = random.choice(NICHES)
    if "cs_approved" not in st.session_state:
        st.session_state["cs_approved"] = []

    def _run_generation(user_instruction: str = ""):
        chromosome = st.session_state["cs_chromosome"]
        niche = st.session_state["cs_niche"]
        content = content_gen.generate_from_ga_result(
            chromosome, niche=niche, user_instruction=user_instruction
        )
        banner_path = banner_ren.render_from_content_dict(
            content,
            save_path=os.path.join(config.OUTPUT_DIR, f"studio_banner_{int(time.time())}.png"),
        )
        st.session_state["cs_content"] = content
        st.session_state["cs_banner_path"] = banner_path

    btn_col1, btn_col2 = st.columns(2)

    if btn_col1.button("🚀 Generate Autonomous", use_container_width=True):
        with st.spinner("Running GA + content generation..."):
            ga_res = optimize_strategy(posts_df, engagement_df)
            st.session_state["cs_chromosome"] = ga_res["best_individual"]
            st.session_state["cs_niche"] = random.choice(NICHES)
            _run_generation()

    if btn_col2.button("🔄 Regenerate", use_container_width=True,
                       disabled=st.session_state["cs_chromosome"] is None):
        with st.spinner("Regenerating with the same winning chromosome..."):
            _run_generation()

    if st.session_state["cs_chromosome"] is not None:
        st.subheader("Active Winning Chromosome")
        chrom = st.session_state["cs_chromosome"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Hook Type", chrom.get("hook_type", "—"))
        k2.metric("Posting Hour", f"{chrom.get('posting_hour', '—')}:00")
        k3.metric("Visual Style", chrom.get("visual_style", "—"))
        k4.metric("Tone", chrom.get("content_tone", "—"))

    st.divider()
    st.subheader("✏️ Revise / Fine-Tune")
    revision_text = st.text_input(
        "Optional instruction (e.g. 'make it more casual', 'shorten caption', 'make it educational')",
        key="cs_revision_box",
    )
    if st.button("Apply Revision", disabled=st.session_state["cs_chromosome"] is None):
        with st.spinner("Applying revision..."):
            _run_generation(user_instruction=revision_text)

    if st.session_state["cs_content"] is not None:
        content = st.session_state["cs_content"]
        st.divider()
        st.subheader("Generated Content Preview")
        left, right = st.columns([1.3, 1])
        with left:
            layout_type = content.get('layout_type', 'bold_banner')
            st.markdown(f"🎨 **Layout Style:** `{layout_type}`")
            st.markdown(f"**Hook:** {content.get('hook', '')}")
            st.markdown(f"**Caption:** {content.get('caption', '')}")
            st.markdown(f"**Hashtags:** {' '.join(content.get('hashtags', []))}")
            st.markdown(f"**CTA:** {content.get('cta', '')}")
            st.caption(f"Source: `{content.get('source', 'unknown')}`")
        with right:
            if st.session_state["cs_banner_path"] and os.path.exists(st.session_state["cs_banner_path"]):
                st.image(st.session_state["cs_banner_path"], use_column_width=True)

        st.divider()
        st.subheader("📅 Schedule Format")
        f_col1, f_col2 = st.columns([1, 1])
        with f_col1:
            selected_format = st.radio("Select Target Format:", ["📸 Post", "📊 Story"], index=0, horizontal=True)
        with f_col2:
            st.write("") # spacing
            if st.button("✅ Approve & Schedule", use_container_width=True):
                chrom = st.session_state["cs_chromosome"] or {}
                peak_hour = chrom.get("posting_hour", 21)
                content_hook = content.get("hook", "")
                file_path = st.session_state.get("cs_banner_path", "N/A")
                niche = st.session_state["cs_niche"]
                
                day, t_slot, t_type = schedule_approved_content(selected_format, content_hook, file_path, niche, peak_hour)
                
                if day:
                    # e.g. "🎉 Approved as Story — scheduled for Monday at 13:00 Poll slot (⭐ Priority)"
                    slot_desc = t_type.split("—")[-1].strip() if "—" in t_type else "slot"
                    st.success(f"🎉 Approved as {selected_format.split(' ')[1]} — scheduled for {day} at {t_slot} {slot_desc} (⭐ Priority)")
                else:
                    st.warning("No available slots for this format in the current week's schedule.")

    if st.session_state["cs_approved"]:
        st.divider()
        st.subheader("📋 Approved & Scheduled Queue")
        st.dataframe(pd.DataFrame(st.session_state["cs_approved"]), use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE: Instagram Manager (Upgraded with Video Reel Generator & Approval Queue)
# ===========================================================================
elif page == "📈 Instagram Manager":
    st.header("📈 Instagram Manager")

    tab_reel, tab_sched, tab_media, tab_audio = st.tabs([
        "🎬 Video Reel Generator", "📅 Smart Scheduler", "🖼️ Custom Media Upload", "🎵 Trending Audio Matcher"
    ])

    # ------------------------------------------------------------------
    # Video Reel Generator (Phase 2 Upgrade)
    # ------------------------------------------------------------------
    with tab_reel:
        st.subheader("🎬 Autonomous 9:16 Reel Studio")
        st.caption("Evolves optimal parameters, pulls live niche trends, scripts voiceover, and renders vertical MP4 video.")

        radar = TrendRadar()
        content_gen = ContentGenerator()
        reel_gen = ReelGenerator()

        r_col1, r_col2 = st.columns(2)
        reel_niche = r_col1.selectbox("Target Niche", NICHES, key="reel_target_niche")
        
        # Pull trend suggestions for the selected niche
        trends_available = radar.get_trends(reel_niche)
        trends_available.append("✍️ Custom Topic (Type your own)...")
        selected_trend_opt = r_col2.selectbox("Live Trending Topic", trends_available, key="reel_target_topic")

        reel_topic = selected_trend_opt
        if selected_trend_opt == "✍️ Custom Topic (Type your own)...":
            custom_input = st.text_input("Enter Custom Topic / Trend Idea:", key="reel_custom_topic")
            reel_topic = custom_input

        if st.button("🚀 Render Autonomous Reel Video (FFmpeg + TTS)", use_container_width=True):
            if not reel_topic.strip():
                st.warning("⚠️ Please select a topic or enter a custom topic to render the reel.")
            else:
                with st.status("Assembling Video Reel...", expanded=True) as status:
                    st.write("1. Evolving optimal strategy via Genetic Algorithm...")
                ga_res = optimize_strategy(posts_df, engagement_df)
                best_chrom = ga_res["best_individual"]

                st.write("2. Generating dynamic voiceover script & headline...")
                script_data = content_gen.generate_script(
                    best_individual=best_chrom,
                    live_topic=reel_topic,
                    script_type="reel",
                    niche=reel_niche
                )

                st.write("3. Rendering frames, synthesising TTS audio & encoding MP4...")
                video_out_path = reel_gen.render_reel(script_data)
                
                status.update(label="Reel Generation Complete!", state="complete", expanded=False)

            st.session_state["latest_reel_path"] = video_out_path
            st.session_state["latest_reel_script"] = script_data
            st.session_state["latest_reel_chromosome"] = best_chrom

        if "latest_reel_path" in st.session_state and os.path.exists(st.session_state["latest_reel_path"]):
            st.divider()
            st.subheader("🎥 Generated Reel Preview")
            v_left, v_right = st.columns([1, 1.2])

            with v_left:
                st.video(st.session_state["latest_reel_path"])
                st.caption(f"Path: `{st.session_state['latest_reel_path']}`")

            with v_right:
                script = st.session_state["latest_reel_script"]
                chrom = st.session_state.get("latest_reel_chromosome", {})
                peak_hour = chrom.get("posting_hour", 21)

                st.markdown(f"**Headline Hook:** {script.get('hook_headline', '')}")
                st.markdown("**Voiceover Script:**")
                for idx, line in enumerate(script.get("voiceover_script", []), 1):
                    st.write(f"• *Line {idx}:* {line}")
                st.markdown(f"**Caption:** {script.get('caption', '')}")
                st.markdown(f"**CTA:** {script.get('cta', '')}")
                st.markdown(f"**Hashtags:** {' '.join(script.get('hashtags', []))}")

                st.divider()
                st.subheader("📅 Schedule Format")
                rf_col1, rf_col2 = st.columns([1, 1])
                with rf_col1:
                    reel_format = st.radio("Select Target Format:", ["📸 Post", "📊 Story"], index=1, horizontal=True, key="reel_format_radio")
                with rf_col2:
                    st.write("") # spacing
                    if st.button("✅ Approve & Schedule Reel", use_container_width=True):
                        chrom = st.session_state.get("latest_reel_chromosome", {})
                        peak_hour = chrom.get("posting_hour", 21)
                        content_hook = script.get("hook_headline", "")
                        file_path = st.session_state["latest_reel_path"]
                        niche = script.get("niche", reel_niche)
                        
                        day, t_slot, t_type = schedule_approved_content(reel_format, content_hook, file_path, niche, peak_hour)
                        
                        if day:
                            slot_desc = t_type.split("—")[-1].strip() if "—" in t_type else "slot"
                            st.success(f"🎉 Approved as {reel_format.split(' ')[1]} — scheduled for {day} at {t_slot} {slot_desc} (⭐ Priority)")
                        else:
                            st.warning("No available slots for this format in the current week's schedule.")

            # --- Display Approved Queue ---
            if st.session_state.get("cs_approved"):
                st.divider()
                st.subheader("📋 Approved & Scheduled Queue")
                st.dataframe(pd.DataFrame(st.session_state["cs_approved"]), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Smart Feed & Story Scheduler
    # ------------------------------------------------------------------
    with tab_sched:
        st.subheader("Weekly Publishing Timetable")
        default_peak = 21
        if st.session_state.get("cs_chromosome"):
            default_peak = st.session_state["cs_chromosome"].get("posting_hour", 21)
        elif st.session_state.get("last_ga_result"):
            default_peak = st.session_state["last_ga_result"]["best_individual"].get("posting_hour", 21)

        peak_hour = st.slider("Optimal Feed peak hour (from latest GA run, adjustable)", 0, 23, int(default_peak))

        if "sched_peak_hour" not in st.session_state:
            st.session_state["sched_peak_hour"] = int(default_peak)
            
        if "weekly_schedule" not in st.session_state:
            st.session_state["weekly_schedule"] = generate_weekly_schedule(peak_hour=st.session_state["sched_peak_hour"])

        if st.button("📅 Generate Weekly Schedule (Resets Queue)"):
            st.session_state["sched_peak_hour"] = peak_hour
            st.session_state["weekly_schedule"] = generate_weekly_schedule(peak_hour=peak_hour)

        st.table(st.session_state["weekly_schedule"].reset_index(drop=True))

    # ------------------------------------------------------------------
    # Custom Media Upload
    # ------------------------------------------------------------------
    with tab_media:
        st.subheader("Upload Your Own Creative")
        st.caption("Lightweight brightness/warmth heuristic biases the generated caption toward your image's mood.")
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
            st.image(pil_img, caption="Uploaded creative", use_column_width=True)

            media_niche = st.selectbox("Niche for this post", NICHES, key="media_niche")
            media_tone = st.selectbox("Content tone", CONTENT_TONES, key="media_tone")
            media_hook_type = st.selectbox("Hook type", HOOK_TYPES, key="media_hook_type")

            if st.button("Generate Caption for This Image"):
                with st.spinner("Analyzing image and generating matching content..."):
                    context_instruction = analyze_uploaded_image(pil_img)
                    media_gen = ContentGenerator()
                    media_content = media_gen.generate_content(
                        niche=media_niche,
                        hook_type=media_hook_type,
                        content_tone=media_tone,
                        visual_style=random.choice(VISUAL_STYLES),
                        user_instruction=context_instruction,
                    )
                    overlaid = overlay_hook_on_image(pil_img, media_content.get("hook", ""))

                st.image(overlaid, caption="Preview with hook overlay", use_column_width=True)
                st.markdown(f"**Hook:** {media_content.get('hook', '')}")
                st.markdown(f"**Caption:** {media_content.get('caption', '')}")
                st.markdown(f"**Hashtags:** {' '.join(media_content.get('hashtags', []))}")
                st.markdown(f"**CTA:** {media_content.get('cta', '')}")
                st.caption(f"Source: `{media_content.get('source', 'unknown')}`")

    # ------------------------------------------------------------------
    # Trending Music & Audio Matcher
    # ------------------------------------------------------------------
    with tab_audio:
        st.subheader("Trending Audio Suggestions")
        st.caption("Rule-based curated suggestions (Phase 1).")
        audio_niche = st.selectbox("Niche", NICHES, key="audio_niche")
        suggestions = get_audio_suggestions(audio_niche)
        st.dataframe(pd.DataFrame(suggestions), use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE: Live Feed Simulation
# ===========================================================================
elif page == "📱 Live Feed Simulation":
    st.header("📱 Live Feed Simulation")
    mode = st.radio("Mode", ["Replay Existing Feed", "Generate New Post"], horizontal=True)

    if mode == "Replay Existing Feed":
        if "replay_idx" not in st.session_state:
            st.session_state.replay_idx = 1

        st.session_state.replay_idx = st.slider("Select Post ID", 1, config.NUM_POSTS, st.session_state.replay_idx)

        post_id = st.session_state.replay_idx
        row = fuzzy_results_df[fuzzy_results_df["post_id"] == post_id].iloc[0]
        p_row = posts_df[posts_df["post_id"] == post_id].iloc[0]
        e_row = engagement_tiered_df[engagement_tiered_df["post_id"] == post_id].iloc[0]

        st.info(f"Replaying Post #{post_id} - Niche: {p_row['niche']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hook", p_row["hook_type"])
        c2.metric("Visual Style", p_row["visual_style"])
        c3.metric("Tone", p_row["content_tone"])
        c4.metric("Tier", e_row["performance_tier"])

        c5, c6, c7 = st.columns(3)
        c5.metric("Engagement Rate", f"{e_row['engagement_rate']*100:.1f}%")
        c6.metric("Sentiment", f"{row['avg_sentiment_score']:.2f}")
        c7.metric("Shift Rate", f"{row['shift_rate']:.2f}")

        if row["trigger_pivot"]:
            st.error("Status: PIVOT TRIGGERED")
        else:
            st.success("Status: STRATEGY STABLE")

        if st.button("Next Post"):
            st.session_state.replay_idx = min(st.session_state.replay_idx + 1, config.NUM_POSTS)
            st.rerun()

    elif mode == "Generate New Post":
        if st.button("🚀 Simulate New Post & Run Full Pipeline"):
            new_post_id = config.NUM_POSTS + random.randint(1, 1000)
            pipeline_steps = []

            content_gen = ContentGenerator()
            banner_ren = BannerRenderer()

            with st.status("Running pipeline...", expanded=True) as status:
                pipeline_steps.append("Generating post & simulating engagement...")
                st.write(f"{len(pipeline_steps)}. {pipeline_steps[-1]}")
                new_post = generate_post(new_post_id)
                new_engagement = simulate_engagement(new_post)
                new_comments = generate_synthetic_comments(new_post_id, new_engagement)

                pipeline_steps.append("Scoring sentiment...")
                st.write(f"{len(pipeline_steps)}. {pipeline_steps[-1]}")
                comments_df_new = pd.DataFrame(new_comments)
                sentiment_result = sentiment_analyzer.score_post_sentiment(new_post_id, comments_df_new)

                pipeline_steps.append("Clustering performance...")
                st.write(f"{len(pipeline_steps)}. {pipeline_steps[-1]}")
                tier = clusterer.predict_tier(new_engagement)

                pipeline_steps.append("Evaluating fuzzy logic...")
                st.write(f"{len(pipeline_steps)}. {pipeline_steps[-1]}")
                fuzzy_result = fuzzy_engine.compute_shift_rate(sentiment_result["avg_sentiment_score"], tier)

                if fuzzy_result["trigger_pivot"]:
                    pipeline_steps.append("Pivot triggered. Running GA optimization...")
                    st.write(f"{len(pipeline_steps)}. {pipeline_steps[-1]}")
                    ga_res = optimize_strategy(posts_df, engagement_df)
                    pipeline_steps.append("Generating content from GA-optimized parameters...")
                    st.write(f"{len(pipeline_steps)}. {pipeline_steps[-1]}")
                    content_dict = content_gen.generate_from_ga_result(ga_res["best_individual"], niche=new_post["niche"])
                    path_taken = "GA-Optimized (strategy pivot triggered)"
                else:
                    pipeline_steps.append("Strategy stable. Generating content from direct parameters...")
                    st.write(f"{len(pipeline_steps)}. {pipeline_steps[-1]}")
                    content_dict = content_gen.generate_content(
                        new_post["niche"],
                        new_post["hook_type"],
                        new_post["content_tone"],
                        new_post["visual_style"]
                    )
                    path_taken = "Direct Generation (Previous strategy was stable)"

                pipeline_steps.append("Rendering banner graphic...")
                st.write(f"{len(pipeline_steps)}. {pipeline_steps[-1]}")
                banner_path = banner_ren.render_from_content_dict(
                    content_dict,
                    save_path=os.path.join(config.OUTPUT_DIR, f"live_banner_{new_post_id}.png")
                )

                status.update(label="Pipeline Complete!", state="complete", expanded=True)

            if "GA-Optimized" in path_taken:
                st.warning(f"Path Taken: {path_taken}")
            else:
                st.success(f"Path Taken: {path_taken}")

            st.markdown(f"**Hook:** {content_dict.get('hook', '')}")
            st.markdown(f"**Caption:** {content_dict.get('caption', '')}")
            st.markdown(f"**Hashtags:** {' '.join(content_dict.get('hashtags', []))}")
            st.markdown(f"**CTA:** {content_dict.get('cta', '')}")

            if banner_path and os.path.exists(banner_path):
                st.image(banner_path, use_column_width=True)