"""Greedy heuristic for the Assignment Problem.

This implementation uses a global greedy rule:
repeatedly select the lowest-cost available (row, column) pair that does
not conflict with previously selected assignments. After selecting a pair,
the corresponding row and column are removed from further consideration.

Algorithm implemented with help from https://en.wikipedia.org/wiki/Assignment_problem#Greedy_heuristic
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


Assignment = List[Tuple[int, int]]


def greedy_assignment(cost_matrix: Sequence[Sequence[float]]) -> Tuple[Assignment, float]:
    """Return a feasible greedy assignment and its total cost.

    Args:
        cost_matrix: square n x n cost matrix

    Returns:
        (assignment, total_cost)
        assignment is a list of (row, col) pairs
    """
    n = len(cost_matrix)
    if n == 0:
        return [], 0.0

     # Ensure matrix is square
    if any(len(row) != n for row in cost_matrix):
        raise ValueError("cost_matrix must be square")

    # Store every possible assignment edge as (cost, row, column)
    edges: List[Tuple[float, int, int]] = []
    for row_idx in range(n):
        for col_idx in range(n):
            edges.append((cost_matrix[row_idx][col_idx], row_idx, col_idx))

    # Sort all edges so cheapest assignments are considered first
    edges.sort(key=lambda x: x[0])

    # Track which rows and columns have already been assigned
    used_rows = set()
    used_cols = set()

     # Store chosen assignments and total greedy cost
    assignment: Assignment = []
    total_cost = 0.0

    # Scan edges from lowest cost to highest cost
    for cost, row_idx, col_idx in edges:
        if row_idx in used_rows or col_idx in used_cols:
            continue

        # Accept the cheapest currently valid assignment    
        assignment.append((row_idx, col_idx))
        total_cost += cost
        used_rows.add(row_idx)
        used_cols.add(col_idx)

        # Stop once every row has been assigned
        if len(assignment) == n:
            break
    
    # complete assignment should have exactly n pairs
    if len(assignment) != n:
        raise RuntimeError("Failed to construct a complete assignment")

    assignment.sort()
    return assignment, float(total_cost)