# COT 5405 Final Project

## Topic
Empirical comparison of two algorithms for the Assignment Problem (minimum-cost bipartite matching):
1. A greedy heuristic
2. The Hungarian algorithm

## Repo Layout
```text
cot5405_assignment_project/
├── README.md
├── requirements.txt
├── .gitignore
├── report_outline.md
├── src/
│   ├── algorithms/
│   │   ├── greedy.py
│   │   └── hungarian.py
│   └── utils/
│       ├── generator.py
│       └── timer.py
├── experiments/
│   └── run_experiments.py
└── tests/
    └── test_correctness.py
```

## Algorithm Definitions

### 1) Greedy Heuristic
This implementation uses a **global greedy heuristic** inspired by https://en.wikipedia.org/wiki/Assignment_problem :

- Consider all possible (agent, task) pairs
- Sort them by cost in increasing order
- Iteratively select the cheapest available pair that does not conflict with previously selected assignments
- Stop when all agents are assigned

This is a greedy algorithm because it makes locally optimal decisions without revisiting earlier choices.

**Time complexity:** `O(n^2 log n)`
- There are `n^2` possible assignments
- Sorting dominates the runtime

This approach is fast but does **not guarantee optimality**, unlike the Hungarian algorithm below.

### 2) Hungarian Algorithm
This implementation uses the standard primal-dual / potential-based Hungarian method for minimum-cost assignment from https://doi.org/10.1002/nav.3800020109.

**Time complexity:** `O(n^3)`

It produces an optimal assignment and is the benchmark algorithm for this project.

## Suggested Experimental Design

### Inputs
Generate random `n x n` integer cost matrices.
- Example cost range: 1 to 1000
- Use multiple trials per `n`
- Fix random seeds for reproducibility

### Recommended sizes
Start with:
- `n = 10, 20, 30, ..., 200`

If runtime is still reasonable, extend Hungarian to:
- `n = 250, 300, 350, 400`

### Trials
- 20 trials for small/medium sizes
- 10 trials for larger sizes

### Measurements
For each size `n` and each algorithm:
- average runtime
- median runtime
- standard deviation

Also record:
- total assignment cost from greedy
- total assignment cost from Hungarian
- approximation ratio = `greedy_cost / optimal_cost`

## Graphs
- Runtime and solution quality plots include **standard deviation error bars** to show variability across trials

### Graph 1: Runtime Comparison
- X-axis: problem size `n`
- Y-axis: average runtime (seconds)
- Curves: greedy vs Hungarian

### Graph 2: Theory vs Empirical Growth
- Plot `T(n)/(n^2 log n)` for greedy
- Plot `T(n)/n^3` for Hungarian

If the theoretical models, these normalized curves should become relatively stable for larger `n`.

### Graph 3: Solution Quality Comparison
- X-axis: problem size `n`
- Y-axis: average assignment cost or approximation ratio

This graph helps justify why Hungarian is worth the extra runtime.

## Key Findings

From empirical experiments:

- The greedy algorithm is significantly faster than the Hungarian algorithm, especially as problem size increases
- The Hungarian algorithm exhibits higher runtime growth consistent with its `O(n^3)` complexity
- Normalized runtime plots align well with theoretical complexity predictions
- The greedy algorithm produces increasingly suboptimal solutions as `n` grows, with approximation ratios rising above 2.5 for large inputs

This demonstrates a clear tradeoff:
- Greedy is fast but approximate
- Hungarian is optimal but slower

## Reproducibility

- Random seeds are fixed for each `(n, trial)` pair
- This ensures consistent and reproducible results
- Minor runtime variation may still occur due to system-level differences
- Use instructions below to run the tests and experiments provided with the algorithms

## How to Run

> Requires Python 3.9+

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run correctness tests
```bash
python tests/test_correctness.py
```

### 3. Run experiments
```bash
python experiments/run_experiments.py
```

This generates:
- a CSV of results
- runtime plot
- normalized plot
- solution quality plot
