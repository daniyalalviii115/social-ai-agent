"""
decision/genetic_optimizer.py — Genetic Algorithm Content Optimizer
====================================================================
Phase 4 of the Autonomous Social Media AI Agent project.

Implements a Genetic Algorithm (GA) that evolves optimal content
parameter combinations (hook_type, posting_hour, visual_style,
content_tone) to maximise engagement-rate-based fitness.

The gene search space is imported directly from core/simulator.py's
module-level lists so that valid GA chromosomes are always consistent
with the simulated dataset.

Downstream consumers:
  - Phase 7 (dashboard/app.py): embeds ga_convergence.png and displays
    the best_individual in the Strategy Recommendations panel.

Usage:
    python decision/genetic_optimizer.py
    from decision.genetic_optimizer import ContentGeneticOptimizer, optimize_strategy
"""

import os
import sys
import random
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for servers/headless
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `import config` and package imports
# work regardless of the working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402

# Import the gene search-space lists from core.simulator as specified
from core.simulator import (  # noqa: E402
    HOOK_TYPES,
    POSTING_HOURS,
    VISUAL_STYLES,
    CONTENT_TONES,
)

# Peak hours that earn impression bonuses (from simulator._HOUR_BONUS)
_PEAK_HOURS: frozenset[int] = frozenset([7, 8, 9, 12, 13, 18, 19, 20, 21])


class ContentGeneticOptimizer:
    """
    Genetic Algorithm for content parameter optimisation.

    Chromosome structure (gene dict)
    ---------------------------------
    {
        "hook_type":     str — one of HOOK_TYPES
        "posting_hour":  int — one of POSTING_HOURS
        "visual_style":  str — one of VISUAL_STYLES
        "content_tone":  str — one of CONTENT_TONES
    }

    Fitness function
    ----------------
    If fitness_lookup_df is provided (posts + engagement merged), the base
    fitness is the historical mean engagement_rate for posts matching the
    individual's hook_type AND content_tone.  Falls back to hook_type alone,
    then to the global mean if no match is found.  Additive bonuses are
    applied for peak posting hours and rarer (more diverse) visual styles.

    GA Parameters
    -------------
    All pulled from config.py at __init__ time:
        GA_POPULATION_SIZE, GA_GENERATIONS, GA_MUTATION_RATE,
        GA_CROSSOVER_RATE, GA_ELITE_COUNT
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, fitness_lookup_df: Optional[pd.DataFrame] = None) -> None:
        """
        Parameters
        ----------
        fitness_lookup_df : pd.DataFrame, optional
            Merged posts + engagement DataFrame with at minimum columns:
            hook_type, content_tone, visual_style, posting_hour,
            engagement_rate.  Used to build the historical fitness heuristic.
        """
        self.population_size : int   = config.GA_POPULATION_SIZE
        self.generations     : int   = config.GA_GENERATIONS
        self.mutation_rate   : float = config.GA_MUTATION_RATE
        self.crossover_rate  : float = config.GA_CROSSOVER_RATE
        self.elite_count     : int   = config.GA_ELITE_COUNT

        # Store lookup data and pre-compute heuristic tables if available
        self._lookup_df: Optional[pd.DataFrame] = fitness_lookup_df
        self._global_mean: float = 0.05        # reasonable engagement default

        # Pre-compute group means for fast fitness evaluation
        self._hook_tone_means: dict[tuple, float] = {}
        self._hook_means: dict[str, float] = {}
        self._visual_style_counts: dict[str, int] = {}

        if fitness_lookup_df is not None and not fitness_lookup_df.empty:
            required_cols = {"hook_type", "content_tone", "engagement_rate"}
            if required_cols.issubset(set(fitness_lookup_df.columns)):
                self._global_mean = float(
                    fitness_lookup_df["engagement_rate"].mean()
                )
                # Group by hook_type + content_tone
                grp1 = (
                    fitness_lookup_df
                    .groupby(["hook_type", "content_tone"])["engagement_rate"]
                    .mean()
                )
                self._hook_tone_means = {
                    k: float(v) for k, v in grp1.items()
                }
                # Group by hook_type alone
                grp2 = (
                    fitness_lookup_df
                    .groupby("hook_type")["engagement_rate"]
                    .mean()
                )
                self._hook_means = {
                    str(k): float(v) for k, v in grp2.items()
                }
                # Visual style frequency counts (for rarity bonus)
                if "visual_style" in fitness_lookup_df.columns:
                    vc = fitness_lookup_df["visual_style"].value_counts()
                    self._visual_style_counts = {
                        str(k): int(v) for k, v in vc.items()
                    }

        # Seed RNG for reproducibility
        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

    # ------------------------------------------------------------------
    # Individual creation
    # ------------------------------------------------------------------

    def _create_individual(self) -> dict:
        """
        Create a random chromosome dict by independently sampling each gene
        from its respective source list.

        Returns
        -------
        dict with keys: hook_type, posting_hour, visual_style, content_tone
        """
        return {
            "hook_type":    random.choice(HOOK_TYPES),
            "posting_hour": random.choice(POSTING_HOURS),
            "visual_style": random.choice(VISUAL_STYLES),
            "content_tone": random.choice(CONTENT_TONES),
        }

    # ------------------------------------------------------------------
    # Fitness function
    # ------------------------------------------------------------------

    def _fitness(self, individual: dict) -> float:
        """
        Compute a non-degenerate fitness score for a chromosome.

        Base fitness
        ------------
        1. Look up mean engagement_rate for rows matching
           (hook_type, content_tone).
        2. Fall back to hook_type-only mean if no combined match.
        3. Fall back to global mean if no hook match.

        Bonuses (additive, scaled to be small relative to base)
        -------
        - +0.005 if posting_hour is in _PEAK_HOURS
        - +rarity_bonus for under-represented visual styles
          (rare style → small bonus to encourage diversity without dominating)

        Returns
        -------
        float — fitness score (higher = better predicted engagement)
        """
        hook  = individual["hook_type"]
        tone  = individual["content_tone"]
        hour  = individual["posting_hour"]
        style = individual["visual_style"]

        # --- Base fitness from historical data or fallback ------------------
        key = (hook, tone)
        if key in self._hook_tone_means:
            base_fitness = self._hook_tone_means[key]
        elif hook in self._hook_means:
            base_fitness = self._hook_means[hook]
        else:
            base_fitness = self._global_mean

        # --- Peak hour bonus ------------------------------------------------
        peak_bonus = 0.005 if hour in _PEAK_HOURS else 0.0

        # --- Visual style rarity bonus (under-used styles get a tiny boost) -
        if self._visual_style_counts:
            max_count   = max(self._visual_style_counts.values()) or 1
            style_count = self._visual_style_counts.get(style, 0)
            # Rarity score in [0, 1]: 1 = rarest, 0 = most common
            rarity      = 1.0 - (style_count / max_count)
            rarity_bonus = 0.002 * rarity   # small so it doesn't dominate
        else:
            rarity_bonus = 0.001            # minimal default if no data

        return float(base_fitness + peak_bonus + rarity_bonus)

    # ------------------------------------------------------------------
    # Selection (tournament)
    # ------------------------------------------------------------------

    def _selection(
        self,
        population: list[dict],
        fitnesses: list[float],
    ) -> list[dict]:
        """
        Tournament selection (k=3) to build a new parent pool.

        For each slot in the new population, randomly pick 3 individuals
        and keep the one with the highest fitness.

        Parameters
        ----------
        population : list[dict] — current generation individuals
        fitnesses  : list[float] — corresponding fitness scores

        Returns
        -------
        list[dict] of the same length as population, filled by tournament winners
        """
        selected = []
        pop_size = len(population)
        tournament_size = 3

        for _ in range(pop_size):
            # Pick tournament_size unique random indices
            indices = random.sample(range(pop_size), min(tournament_size, pop_size))
            winner_idx = max(indices, key=lambda i: fitnesses[i])
            selected.append(population[winner_idx].copy())

        return selected

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def _crossover(
        self,
        parent1: dict,
        parent2: dict,
    ) -> tuple[dict, dict]:
        """
        Uniform crossover over the 4 gene keys.

        With probability config.GA_CROSSOVER_RATE, each gene is
        independently assigned to child1/child2 by a coin flip (uniform
        crossover).  If crossover does not happen, children are copies of
        the parents unchanged.

        Parameters
        ----------
        parent1, parent2 : dict — two parent chromosomes

        Returns
        -------
        tuple[dict, dict] — (child1, child2)
        """
        gene_keys = ["hook_type", "posting_hour", "visual_style", "content_tone"]

        if random.random() >= self.crossover_rate:
            # No crossover — return parent copies unchanged
            return parent1.copy(), parent2.copy()

        child1: dict = {}
        child2: dict = {}

        for key in gene_keys:
            if random.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]

        return child1, child2

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _mutate(self, individual: dict) -> dict:
        """
        Gene-wise mutation: each gene mutates with probability GA_MUTATION_RATE.

        Mutated gene value is sampled fresh from its source list (may be
        the same value — this is standard and correct behaviour).

        Parameters
        ----------
        individual : dict — chromosome to (potentially) mutate

        Returns
        -------
        dict — new mutated chromosome (original is not modified in-place)
        """
        child = individual.copy()

        gene_sources = {
            "hook_type":    HOOK_TYPES,
            "posting_hour": POSTING_HOURS,
            "visual_style": VISUAL_STYLES,
            "content_tone": CONTENT_TONES,
        }

        for gene, source_list in gene_sources.items():
            if random.random() < self.mutation_rate:
                child[gene] = random.choice(source_list)

        return child

    # ------------------------------------------------------------------
    # Main GA loop
    # ------------------------------------------------------------------

    def evolve(self, verbose: bool = True) -> dict:
        """
        Run the full Genetic Algorithm for config.GA_GENERATIONS generations.

        Algorithm outline
        -----------------
        1. Initialise random population of size GA_POPULATION_SIZE.
        2. Each generation:
            a. Compute fitness for all individuals.
            b. Carry forward top GA_ELITE_COUNT unchanged (elitism).
            c. Fill remaining slots via tournament selection → crossover → mutation.
            d. Track best fitness seen so far (monotonically non-decreasing).
        3. Return best individual, best fitness, fitness_history.

        Parameters
        ----------
        verbose : bool
            If True, print progress every 5 generations.

        Returns
        -------
        dict with keys:
            best_individual  : dict  — gene combo with highest fitness ever found
            best_fitness     : float
            fitness_history  : list[float] — best fitness per generation (length
                               == GA_GENERATIONS; strictly non-decreasing)
            final_population : list[dict]
        """
        # Initialise population
        population: list[dict] = [
            self._create_individual() for _ in range(self.population_size)
        ]

        best_individual: dict  = {}
        best_fitness: float    = float("-inf")
        fitness_history: list[float] = []

        for generation in range(1, self.generations + 1):
            # --- Evaluate fitnesses -----------------------------------------
            fitnesses = [self._fitness(ind) for ind in population]

            # --- Update global best ------------------------------------------
            gen_best_idx     = int(np.argmax(fitnesses))
            gen_best_fitness = float(fitnesses[gen_best_idx])

            if gen_best_fitness > best_fitness:
                best_fitness    = gen_best_fitness
                best_individual = population[gen_best_idx].copy()

            # Record the best fitness *ever seen* (non-decreasing history)
            fitness_history.append(best_fitness)

            # --- Elitism: preserve top GA_ELITE_COUNT individuals -----------
            sorted_indices   = sorted(
                range(len(population)), key=lambda i: fitnesses[i], reverse=True
            )
            elites = [population[i].copy() for i in sorted_indices[: self.elite_count]]

            # --- Build next generation via selection → crossover → mutation --
            selected    = self._selection(population, fitnesses)
            next_gen: list[dict] = list(elites)   # start with elites

            # Fill up to population_size
            idx = 0
            while len(next_gen) < self.population_size:
                parent1 = selected[idx % len(selected)]
                parent2 = selected[(idx + 1) % len(selected)]
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                next_gen.append(child1)
                if len(next_gen) < self.population_size:
                    next_gen.append(child2)
                idx += 2

            population = next_gen

            # --- Progress reporting -----------------------------------------
            if verbose and (generation % 5 == 0 or generation == 1 or
                            generation == self.generations):
                print(
                    f"  Gen {generation:>3}/{self.generations}  |  "
                    f"Best fitness: {best_fitness:.6f}  |  "
                    f"Best individual: {best_individual}"
                )

        return {
            "best_individual":  best_individual,
            "best_fitness":     best_fitness,
            "fitness_history":  fitness_history,
            "final_population": population,
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_convergence(
        self,
        fitness_history: list[float],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Line plot of best fitness per generation (GA convergence curve).

        Parameters
        ----------
        fitness_history : list[float]
            List of best fitness values, one per generation
            (monotonically non-decreasing).
        save_path : str, optional
            File path to save the figure.  Defaults to
            config.OUTPUT_DIR/ga_convergence.png.
        """
        if save_path is None:
            save_path = os.path.join(config.OUTPUT_DIR, "ga_convergence.png")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        generations = list(range(1, len(fitness_history) + 1))

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0F1117")
        ax.set_facecolor("#1A1D27")

        # Main convergence line
        ax.plot(
            generations,
            fitness_history,
            color="#7C83FD",
            lw=2.5,
            zorder=3,
            label="Best Fitness",
        )

        # Fill under curve for visual appeal
        ax.fill_between(
            generations,
            fitness_history,
            alpha=0.20,
            color="#7C83FD",
        )

        # Annotate final best
        ax.axhline(
            y=fitness_history[-1],
            color="#F39C12",
            lw=1.2,
            linestyle="--",
            alpha=0.7,
            label=f"Final best: {fitness_history[-1]:.6f}",
        )

        ax.set_title(
            "Genetic Algorithm Convergence\nContent Parameter Optimisation",
            color="white",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Generation", color="#AAAAAA", fontsize=11)
        ax.set_ylabel("Best Fitness (Engagement Rate Proxy)", color="#AAAAAA", fontsize=11)
        ax.tick_params(colors="#AAAAAA")
        ax.spines["bottom"].set_color("#444444")
        ax.spines["left"].set_color("#444444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.15, color="#AAAAAA", linestyle="--")
        ax.legend(
            facecolor="#22252F",
            edgecolor="#444444",
            labelcolor="white",
            fontsize=10,
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  GA convergence plot saved to: {save_path}")


# ===========================================================================
# Module-level convenience function
# ===========================================================================

def optimize_strategy(
    posts_df: Optional[pd.DataFrame] = None,
    engagement_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Load data, build optimizer, run GA, save convergence plot, print summary.

    Parameters
    ----------
    posts_df      : pd.DataFrame, optional — if None, loads posts.csv
    engagement_df : pd.DataFrame, optional — if None, loads engagement.csv

    Returns
    -------
    dict — the result dict returned by ContentGeneticOptimizer.evolve()
    """
    # --- Load data if not provided -------------------------------------------
    if posts_df is None:
        posts_path = os.path.join(config.DATA_DIR, "posts.csv")
        print(f"  Loading posts from: {posts_path}")
        posts_df = pd.read_csv(posts_path)

    if engagement_df is None:
        engagement_path = os.path.join(config.DATA_DIR, "engagement.csv")
        print(f"  Loading engagement from: {engagement_path}")
        engagement_df = pd.read_csv(engagement_path)

    # --- Merge on post_id to build fitness_lookup_df -------------------------
    fitness_lookup_df = pd.merge(posts_df, engagement_df, on="post_id", how="inner")
    print(f"  Merged fitness lookup rows: {len(fitness_lookup_df)}")

    # --- Instantiate and run -------------------------------------------------
    optimizer = ContentGeneticOptimizer(fitness_lookup_df=fitness_lookup_df)

    print()
    print("  Starting Genetic Algorithm evolution...")
    result = optimizer.evolve(verbose=True)

    # --- Save convergence plot -----------------------------------------------
    convergence_path = os.path.join(config.OUTPUT_DIR, "ga_convergence.png")
    optimizer.plot_convergence(
        fitness_history=result["fitness_history"],
        save_path=convergence_path,
    )

    # --- Console summary -----------------------------------------------------
    print()
    print("=" * 65)
    print("  GENETIC ALGORITHM OPTIMISATION RESULTS")
    print("=" * 65)
    print(f"  Generations run    : {config.GA_GENERATIONS}")
    print(f"  Population size    : {config.GA_POPULATION_SIZE}")
    print(f"  Best fitness       : {result['best_fitness']:.6f}")
    print()
    print("  Best individual (optimal content parameters):")
    for gene, value in result["best_individual"].items():
        print(f"    {gene:<18}: {value}")
    print()
    print(f"  Fitness history (first 5 gens): "
          f"{[round(f, 6) for f in result['fitness_history'][:5]]}")
    print(f"  Fitness history (last  5 gens): "
          f"{[round(f, 6) for f in result['fitness_history'][-5:]]}")
    print("=" * 65)

    return result


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  PHASE 4b — GENETIC ALGORITHM EXECUTION")
    print("=" * 65)
    optimize_strategy()
    print()
    print("  PHASE 4b COMPLETE")
