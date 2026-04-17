"""Hungarian algorithm for the Assignment Problem.

This implementation uses the standard potential-based O(n^3) method for
minimum-cost perfect matching in a square cost matrix.

Algorithm implemented with help from https://doi.org/10.1002/nav.3800020109

"""

from __future__ import annotations

from typing import List, Sequence, Tuple


Assignment = List[Tuple[int, int]]


def hungarian_assignment(cost_matrix: Sequence[Sequence[float]]) -> Tuple[Assignment, float]:
    """Return the optimal assignment and its total cost.

    Args:
        cost_matrix: square n x n cost matrix

    Returns:
        (assignment, total_cost)
    """
    n = len(cost_matrix)
    if n == 0:
        return [], 0.0

    if any(len(row) != n for row in cost_matrix):
        raise ValueError("cost_matrix must be square")

    # 1-indexed internals for the classical algorithm.
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    # Iterate over each row to build matching
    for i in range(1, n + 1):
        p[0] = i
        minv = [float("inf")] * (n + 1)  # Minimum reduced cost to each column
        used = [False] * (n + 1) # Tracks visited columns
        j0 = 0 

        while True:
            used[j0] = True
            i0 = p[j0]  # Current row being matched
            delta = float("inf")
            j1 = 0
            # Try to relax edges to all unused columns
            for j in range(1, n + 1):
                if not used[j]:
                    # Reduced cost using potentials
                    cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]

                    # Update best edge to column j
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0

                    # Track smallest slack value
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            # Update potentials to maintain feasibility
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1

            # If we found an unmatched column, stop
            if p[j0] == 0:
                break

        # Reconstruct augmenting path and update matching        
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # Build final assignment from matching array            
    assignment: Assignment = []
    for j in range(1, n + 1):
        i = p[j]
        if i != 0:
            assignment.append((i - 1, j - 1))

    assignment.sort() # Sort for consistency
    
    # Compute total cost of assignment
    total_cost = sum(cost_matrix[i][j] for i, j in assignment)
    return assignment, float(total_cost)
