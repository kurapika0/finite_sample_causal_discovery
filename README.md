# Finite-Sample Causal Discovery Benchmark

This project contains two causal discovery workflows on finite-sample linear Gaussian DAGs:

- `Direct Discovery Benchmark`: run `PC`, `GES`, and `BOSS` directly on samples and compare their recovered graphs against the true DAG.
- `Greedy {epsilon, delta} Top-K Recovery`: start from the `BOSS` output graph and run the greedy Top-K search over skeleton and permutation neighborhoods.

## Setup

Create a virtual environment and install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Workflow 1: Direct Discovery Benchmark

This is the original benchmark workflow. It runs `PC`, `GES`, and `BOSS` directly on the sampled data and reports `ds`, `dp`, and runtime.

### Run a single benchmark

```bash
python -m fscd.run \
  --algorithms pc ges boss \
  --nodes 4 \
  --densities 0.5 \
  --sample-sizes 20 \
  --runs 2 \
  --output results/smoke
```

### Run the default full-configuration benchmark

```bash
python -m fscd.run \
  --algorithms pc ges boss \
  --nodes 4 6 8 10 \
  --densities 0.2 0.5 0.8 \
  --sample-sizes 20 50 100 500 1000 5000 10000 \
  --runs 50 \
  --output results/default
```

  <!-- --nodes 5 10 15 \ -->

## Workflow 2: Greedy {epsilon, delta} Top-K Recovery

This second workflow starts from the `BOSS` output graph, extracts its skeleton and
lexicographically smallest topological order, and then runs the greedy Top-K search over
`(epsilon, delta, K)`.

The default command reproduces the new experiment implemented in this repo:

```bash
python -m fscd.run_greedy
```

Its built-in defaults are:

- `K_max = 10`
- `d = 5`
- `density = 0.4`
- `n_samples in {20, 500, 10000}`
- `epsilon in {1, 2}`
- `delta in {1, 2}`
- `runs = 50`

The default output directory is `results/greedy_default`, which contains:

- `raw_topk_results.csv`: one row per `(run_id, n_samples, epsilon, delta, k)`
- `summary_topk_results.csv`: aggregated `ds/dp/BIC/found_true_by_k` statistics
- `recovery_summary.csv`: Top-`K_max` recovery summary for each parameter setting
- `plots/topk_distances_*.png`: representative `ds` and `dp` vs `K` figures for `n=20`, `500`, and `10000`

Example with explicit arguments:

```bash
python -m fscd.run_greedy \
  --nodes 5 \
  --densities 0.4 \
  --sample-sizes 50 500 10000 \
  --epsilons 1 2 \
  --deltas 1 2 \
  --k-max 10 \
  --runs 50 \
  --output results/greedy_custom \
  --plot-sample-sizes 20 500 10000 \
  --plot-epsilon 1 \
  --plot-delta 1
```


## Test

```bash
pytest
```
