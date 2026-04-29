# Finite-Sample Causal Discovery Benchmark

This project benchmarks `PC`, `GES`, and `BOSS` on finite-sample linear Gaussian DAG discovery.

## Setup

Create a virtual environment and install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run a single benchmark

```bash
python -m fscd.run \
  --algorithms pc ges boss \
  --nodes 4 \
  --densities 0.5 \
  --sample-sizes 20 \
  --runs 2 \
  --output results/smoke
```

## Run the default full-configuration benchmark

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


## Test

```bash
pytest
```
