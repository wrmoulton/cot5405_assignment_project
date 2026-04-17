from __future__ import annotations

import itertools
import math
import os
import sys  

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.algorithms.greedy import greedy_assignment
from src.algorithms.hungarian import hungarian_assignment
from src.utils.generator import generate_cost_matrix


def brute_force_assignment(cost_matrix):
    """Check all n! permutations and return the guaranteed optimal assignment."""
    n = len(cost_matrix)
    best_cost = math.inf
    best_assignment = None

    for perm in itertools.permutations(range(n)):
        cost = sum(cost_matrix[i][perm[i]] for i in range(n))
        if cost < best_cost:
            best_cost = cost
            best_assignment = [(i, perm[i]) for i in range(n)]

    return best_assignment, best_cost


def is_valid_assignment(assignment, n):
    """Check that every row and column appears exactly once in the assignment."""
    rows = [r for r, _ in assignment]
    cols = [c for _, c in assignment]
    return sorted(rows) == list(range(n)) and sorted(cols) == list(range(n))


def test_hungarian_matches_bruteforce():
    """Hungarian must match brute force optimal cost on small inputs."""
    for n in range(2, 7):
        for seed in range(5):
            matrix = generate_cost_matrix(n=n, low=1, high=30, seed=100 * n + seed)
            _, optimal_cost = brute_force_assignment(matrix)
            assignment, hungarian_cost = hungarian_assignment(matrix)

            assert is_valid_assignment(assignment, n), \
                f"Invalid Hungarian assignment for n={n}, seed={seed}"

            assert math.isclose(hungarian_cost, optimal_cost, rel_tol=1e-9), (
                f"Hungarian mismatch for n={n}, seed={seed}: got {hungarian_cost}, expected {optimal_cost}"
            )

def test_empty_matrix():
    """Both algorithms should handle the empty matrix case."""
    assert greedy_assignment([]) == ([], 0.0)
    assert hungarian_assignment([]) == ([], 0.0)

def test_non_square_matrix_raises():
    """Both algorithms should raise ValueError for non-square matrices."""
    bad_matrix = [[1, 2], [3]]
    try:
        greedy_assignment(bad_matrix)
        assert False, "Expected ValueError for greedy"
    except ValueError:
        pass

    try:
        hungarian_assignment(bad_matrix)
        assert False, "Expected ValueError for Hungarian"
    except ValueError:
        pass

def test_greedy_is_feasible():
    """Greedy must produce a valid assignment with a correctly computed cost."""
    for n in range(2, 15):
        for seed in range(5):
            matrix = generate_cost_matrix(n=n, low=1, high=100, seed=100 * n + seed)
            assignment, greedy_cost = greedy_assignment(matrix)

            assert is_valid_assignment(assignment, n), \
                f"Invalid greedy assignment for n={n}, seed={seed}"

            expected_cost = sum(matrix[r][c] for r, c in assignment)
            assert math.isclose(greedy_cost, expected_cost), \
                f"Greedy cost mismatch for n={n}, seed={seed}: got {greedy_cost}, expected {expected_cost}"


def test_greedy_vs_hungarian():
    """Greedy cost must never be less than Hungarian (optimal) cost."""
    for n in range(2, 15):
        for seed in range(5):
            matrix = generate_cost_matrix(n=n, low=1, high=100, seed=100 * n + seed)
            _, greedy_cost = greedy_assignment(matrix)
            _, hungarian_cost = hungarian_assignment(matrix)

            assert greedy_cost >= hungarian_cost - 1e-9, (
                f"Greedy beat Hungarian for n={n}, seed={seed}: "
                f"greedy={greedy_cost}, hungarian={hungarian_cost}"
            )

def test_greedy_can_be_suboptimal():
    """Concrete example where greedy is strictly worse than optimal."""
    matrix = [
        [1, 2],
        [5, 8],
    ]
    _, greedy_cost = greedy_assignment(matrix)
    _, hungarian_cost = hungarian_assignment(matrix)

    assert greedy_cost == 9
    assert hungarian_cost == 7
    assert greedy_cost > hungarian_cost


if __name__ == "__main__":
    test_hungarian_matches_bruteforce()
    test_empty_matrix()
    test_non_square_matrix_raises()
    test_greedy_is_feasible()
    test_greedy_vs_hungarian()
    test_greedy_can_be_suboptimal()
    print("All tests passed.")
