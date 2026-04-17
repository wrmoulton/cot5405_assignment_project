from __future__ import annotations

import numpy as np
import os
import statistics
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import pandas as pd

from src.algorithms.greedy import greedy_assignment
from src.algorithms.hungarian import hungarian_assignment
from src.utils.generator import generate_cost_matrix
from src.utils.timer import time_function


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Problem sizes to benchmark
SIZES = list(range(10, 210, 10))

# Use more trials for smaller sizes and fewer for larger sizes to manage total runtime while still getting good estimates.
TRIALS_SMALL = 20
TRIALS_LARGE = 10
LARGE_THRESHOLD = 120


def trials_for_size(n: int) -> int:
    """Return the number of trials to run for a given input size."""
    return TRIALS_SMALL if n <= LARGE_THRESHOLD else TRIALS_LARGE


def main() -> None:
    rows = []

    # Run experiments for each matrix size
    for n in SIZES:
        num_trials = trials_for_size(n)
        greedy_times = []
        hungarian_times = []
        greedy_costs = []
        hungarian_costs = []
        ratios = []

        # Repeat each input size multiple times for averaging
        for trial in range(num_trials):
             # Generate reproducible random cost matrix
            matrix = generate_cost_matrix(n=n, low=1, high=1000, seed=10_000 * n + trial)

            # Measure runtime and output cost for each algorithm
            (_, greedy_cost), greedy_time = time_function(greedy_assignment, matrix)
            (_, hungarian_cost), hungarian_time = time_function(hungarian_assignment, matrix)

            greedy_times.append(greedy_time)
            hungarian_times.append(hungarian_time)
            greedy_costs.append(greedy_cost)
            hungarian_costs.append(hungarian_cost)

            # Compare greedy cost against optimal Hungarian cost
            ratios.append(greedy_cost / hungarian_cost if hungarian_cost else 1.0)

        # Store summary statistics for this input size
        rows.append(
            {
                "n": n,
                "trials": num_trials,
                "greedy_avg_time": statistics.mean(greedy_times),
                "greedy_median_time": statistics.median(greedy_times),
                "greedy_std_time": statistics.pstdev(greedy_times),
                "hungarian_avg_time": statistics.mean(hungarian_times),
                "hungarian_median_time": statistics.median(hungarian_times),
                "hungarian_std_time": statistics.pstdev(hungarian_times),
                "greedy_avg_cost": statistics.mean(greedy_costs),
                "hungarian_avg_cost": statistics.mean(hungarian_costs),
                "avg_approx_ratio": statistics.mean(ratios),
                "std_approx_ratio": statistics.pstdev(ratios),
            }
        )
        print(f"Completed n={n} with {num_trials} trials")

    # Convert collected results into a DataFrame and save to CSV
    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "assignment_experiment_results.csv"
    df.to_csv(csv_path, index=False)

    plot_runtime(df)
    plot_normalized(df)
    plot_quality(df)

    print(f"Saved results to {csv_path}")


def plot_runtime(df: pd.DataFrame) -> None:
    """Plot average runtime with standard deviation error bars."""
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        df["n"],
        df["greedy_avg_time"],
        yerr=df["greedy_std_time"],
        marker="o",
        capsize=4,
        label="Greedy",
    )
    plt.errorbar(
        df["n"],
        df["hungarian_avg_time"],
        yerr=df["hungarian_std_time"],
        marker="o",
        capsize=4,
        label="Hungarian",
    )
    plt.xlabel("Problem size (n)")
    plt.ylabel("Average runtime (seconds)")
    plt.title("Runtime Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "runtime_comparison.png", dpi=200)
    plt.close()


def plot_normalized(df: pd.DataFrame) -> None:
    """Plot runtimes normalized by theoretical growth rates."""
    plt.figure(figsize=(8, 5))

    # Greedy is normalized by O(n^2 log n)
    plt.plot(df["n"], df["greedy_avg_time"] / (df["n"] ** 2 * np.log(df["n"])), marker="o", label="Greedy: T(n)/(n² log n)")

    # Hungarian is normalized by O(n^3)
    plt.plot(df["n"], df["hungarian_avg_time"] / (df["n"] ** 3), marker="o", label="Hungarian: T(n)/n^3")

    plt.xlabel("Problem size (n)")
    plt.ylabel("Normalized runtime")
    plt.title("Theoretical vs Empirical Growth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "normalized_runtime.png", dpi=200)
    plt.close()


def plot_quality(df: pd.DataFrame) -> None:
    """Plot greedy solution quality relative to the optimal solution."""
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        df["n"],
        df["avg_approx_ratio"],
        yerr=df["std_approx_ratio"],
        marker="o",
        capsize=4,
    )
    plt.xlabel("Problem size (n)")
    plt.ylabel("Average greedy/optimal cost ratio")
    plt.title("Solution Quality Comparison")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "solution_quality.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
