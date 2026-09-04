"""
config.py — Central Configuration Module
=========================================
This is the single source of truth for all project-wide constants and
path definitions. Every subsequent phase (K-Means clustering, fuzzy logic,
genetic algorithm, LLM content generation, Streamlit dashboard) imports
its constants from this exact file using the exact names defined here.

Do NOT rename any constant — later phases import them by name directly:
    from config import KMEANS_N_CLUSTERS, GA_POPULATION_SIZE, ...

Usage:
    import config
    # or
    from config import NUM_POSTS, DATA_DIR, ...
"""

import os
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file from the project root directory (same folder as this file).
# This populates os.environ with values from .env before we call os.getenv().
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


# ===========================================================================
# LLM Provider Configuration
# ===========================================================================

# Which LLM backend to use: "gemini" (Google Generative AI) or "groq".
# Phase 6 (LLM Content Generation) reads this to decide which client to init.
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")

# API key for Google Gemini (google-generativeai SDK).
# Must be set in .env as: GEMINI_API_KEY=your_actual_key
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# API key for Groq (groq SDK, e.g. Llama-3 / Mixtral hosted on Groq cloud).
# Must be set in .env as: GROQ_API_KEY=your_actual_key
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")


# ===========================================================================
# Data Simulation Constants  (Phase 2 — core/data_simulator.py)
# ===========================================================================

# Total number of synthetic social-media posts to generate.
# Increasing this improves cluster quality but slows GA convergence.
NUM_POSTS: int = 150

# Inclusive range (min, max) for the number of comments on each post.
# Used by the simulator when sampling comment counts from a random distribution.
COMMENTS_PER_POST_RANGE: tuple[int, int] = (5, 40)

# Global random seed — set everywhere (numpy, pandas sample, sklearn) so that
# results are fully reproducible across runs.
RANDOM_SEED: int = 42


# ===========================================================================
# K-Means Clustering Constants  (Phase 3 — ml/clustering.py)
# ===========================================================================

# Number of audience segments / clusters to discover.
# Three clusters map intuitively to: Low-engagement, Mid-engagement, High-engagement.
KMEANS_N_CLUSTERS: int = 3

# Random state passed to sklearn.cluster.KMeans to ensure reproducible centroids.
KMEANS_RANDOM_STATE: int = 42


# ===========================================================================
# Fuzzy Logic Constants  (Phase 4 — ml/fuzzy_engine.py)
# ===========================================================================

# Universe of discourse for fuzzy output membership functions (e.g. post score).
# A (0, 1) range means all defuzzified outputs are normalised probabilities/scores.
FUZZY_OUTPUT_RANGE: tuple[float, float] = (0, 1)


# ===========================================================================
# Genetic Algorithm Constants  (Phase 5 — ml/genetic_algorithm.py)
# ===========================================================================

# Number of candidate strategies (chromosomes) in each generation.
# Larger populations explore the search space more broadly but run slower.
GA_POPULATION_SIZE: int = 30

# Number of evolutionary iterations (generations) to run before stopping.
# 40 generations × 30 individuals = 1 200 fitness evaluations total.
GA_GENERATIONS: int = 40

# Probability that a single gene mutates during reproduction.
# 0.15 provides enough diversity without destroying good solutions.
GA_MUTATION_RATE: float = 0.15

# Probability that two parents exchange genetic material at a crossover point.
# 0.7 is a standard value that balances exploitation and exploration.
GA_CROSSOVER_RATE: float = 0.7

# Number of top-performing individuals automatically carried to the next
# generation without modification (elitism). Prevents regression.
GA_ELITE_COUNT: int = 2


# ===========================================================================
# Directory Paths
# ===========================================================================

# Absolute path to the directory containing this config.py file.
# All other paths are derived from BASE_DIR so the project is relocatable.
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# Directory where generated CSV datasets are stored (e.g. simulated_posts.csv).
# Listed in .gitignore (data/*.csv) so large generated files are not committed.
DATA_DIR: str = os.path.join(BASE_DIR, "data")

# Directory for static assets — fonts and background images used by
# Phase 6 (content/banner_generator.py) when compositing post banners.
ASSETS_DIR: str = os.path.join(BASE_DIR, "assets")

# Directory for all generated artefacts: banner images, PDF reports,
# exported cluster plots, etc. Listed in .gitignore so they are not committed.
OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs")

# ---------------------------------------------------------------------------
# Ensure all output/data directories exist at import time.
# Using exist_ok=True makes this idempotent — safe to call on every import.
# ---------------------------------------------------------------------------
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
