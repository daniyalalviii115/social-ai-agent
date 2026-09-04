# Autonomous Social Media AI Agent & Strategy Optimization Dashboard

An end-to-end simulation system that autonomously generates Instagram content, simulates audience engagement, and applies a set of AI/ML algorithms to evaluate content performance and trigger strategy pivots — all visualized through an interactive Streamlit dashboard.

---

## Overview

This project simulates the full lifecycle of managing an Instagram content strategy using AI:

1. **Content Generation** — Captions, hooks, and hashtags are generated using a free-tier LLM API (Gemini/Groq), with an automatic fallback to a template-based generator when no API key is configured.
2. **Engagement Simulation** — Synthetic audience metrics (impressions, likes, comments, shares, saves) are generated for each post.
3. **Performance Evaluation** — Four core AI/ML algorithms classify sentiment, cluster posts by performance tier, compute a strategy-shift rate, and evolve optimized content parameters when a pivot is triggered.
4. **Interactive Dashboard** — A four-page Streamlit app provides a live, clickable demo of the entire pipeline.

The system is designed to run entirely on free-tier tools at **$0 cost**.

---

## Algorithm Architecture

| Module | Algorithm | File | Purpose |
|---|---|---|---|
| ML | Naïve Bayes | `ml/naive_bayes.py` | Classifies comment sentiment (positive / negative / neutral) |
| ML | K-Means Clustering | `ml/kmeans_cluster.py` | Groups posts into **Flop**, **Average**, and **Viral** performance tiers |
| Decision | Mamdani Fuzzy Inference | `decision/fuzzy_engine.py` | Combines sentiment + tier to compute a defuzzified strategy-shift rate (0–1) |
| Decision | Genetic Algorithm | `decision/genetic_optimizer.py` | Evolves optimal content parameters when a strategy pivot is triggered |

**Pipeline flow:**

```
Post + Comments → Sentiment (Naïve Bayes) → Tier (K-Means)
     → Shift Rate (Fuzzy Engine) → [if pivot triggered] → GA Optimization
     → Content Generation (LLM/Fallback) → Banner Rendering
```

---

## Project Structure

```
social-ai-agent/
├── config.py
├── main.py                    # CLI orchestrator (--full-run for end-to-end pipeline)
├── run_all_tests.py           # Runs all phase test suites with a master summary
├── DEMO_SCRIPT.md             # Live demo walkthrough with discussion points
├── requirements.txt
├── .env / .env.example
├── core/
│   ├── simulator.py           # Synthetic post & comment generation
│   └── test_simulator.py
├── ml/
│   ├── naive_bayes.py         # Sentiment classification
│   ├── kmeans_cluster.py      # Performance tier clustering
│   └── test_ml_modules.py
├── decision/
│   ├── fuzzy_engine.py        # Mamdani fuzzy inference
│   ├── genetic_optimizer.py   # Genetic algorithm for parameter optimization
│   └── test_decision_modules.py
├── content/
│   ├── llm_generator.py       # LLM-based content generation with fallback
│   ├── banner_renderer.py     # Banner graphic rendering
│   └── test_content_modules.py
├── dashboard/
│   ├── app.py                 # Streamlit dashboard (4 pages)
│   ├── test_dashboard_imports.py
│   └── README_RUN.md
├── data/                      # Auto-generated CSVs (posts, engagement, comments, fuzzy results)
└── outputs/                   # Auto-generated visualizations and banner graphics
```

---

## Setup & Installation

### Requirements
- Python **3.12.10** (Python 3.14 is not currently supported due to missing NumPy wheels)

### Steps

```bash
# 1. Clone or navigate to the project directory
cd social-ai-agent

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Optionally add a Gemini or Groq API key to enable live LLM content generation.
# Without a key, the system automatically falls back to a template-based generator.
```

> **Note:** `scikit-fuzzy==0.5.0` is used instead of `0.4.2` for compatibility with Python 3.12 (the `distutils` module was removed in this version).

---

## Running the Project

### Full Pipeline (CLI)
```bash
python main.py --full-run
```
Runs the complete pipeline — simulation, sentiment analysis, clustering, fuzzy evaluation, optional GA optimization, content generation, and banner rendering — in a single command.

### Test Suite
```bash
python run_all_tests.py
```
Runs all module-level test suites and prints a consolidated summary.

### Dashboard
```bash
streamlit run dashboard/app.py
```
Opens the interactive dashboard in your browser (default: `http://localhost:8501`).

**Dashboard pages:**
| Page | Description |
|---|---|
| Overview | KPI summary cards and historical data table |
| Cluster Analysis | Scatter plot of post performance with live tier prediction |
| Fuzzy & GA Engine | Adjustable sentiment/tier sliders, live shift-rate computation, and GA runner |
| Live Feed Simulation | Replay historical posts or generate new posts through the full live pipeline |

---

## Verified Results

| Component | Result |
|---|---|
| Synthetic dataset | 150 posts, 3,348 comments |
| Naïve Bayes | 100% test accuracy (synthetic data is trivially separable; expected to be lower on real-world data) |
| K-Means tiers | Flop: 70 · Average: 62 · Viral: 18 (correctly ordered by engagement rate) |
| Fuzzy engine | Mean shift rate: 0.4607 · 22/150 posts trigger a pivot |
| Genetic Algorithm | Best fitness: 0.074235, converging within 5 generations (small dataset) |
| Full pipeline (`main.py --full-run`) | 6/6 stages complete, no errors |
| Test suite (`run_all_tests.py`) | 5/5 suites passed |
| Dashboard | All 4 pages manually verified end-to-end |

---

## Known Design Notes

- **Average-tier shift rate is always exactly 0.5** — this is an intentional artifact of the fuzzy rule design (sentiment has no effect within the Average tier), not a bug.
- **LLM fallback is expected behavior** — if no API key is set in `.env`, content generation automatically uses a template-based fallback system rather than failing. The content source is labeled `fallback_template` vs `llm` accordingly.
- **GA convergence in ~5 generations** is expected given the relatively small (150-post) dataset.

---

## Tech Stack

Python 3.12 · scikit-learn · scikit-fuzzy · Pillow · Streamlit · pandas · NumPy · Matplotlib · Seaborn · Gemini / Groq (optional, free-tier)
