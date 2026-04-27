# Finite-Sample Causal Discovery Benchmark

This project benchmarks `PC`, `GES`, and `BOSS` on finite-sample linear Gaussian DAG discovery.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the default PDF-style benchmark

```bash
python -m fscd.run \
  --algorithms pc ges boss \
  --nodes 5 10 15 \
  --densities 0.2 0.5 0.8 \
  --sample-sizes 20 50 100 200 300 1000 5000 10000 \
  --runs 100 \
  --output results/pdf_default
```

## Run a smoke benchmark

```bash
python -m fscd.run \
  --algorithms pc ges boss \
  --nodes 4 \
  --densities 0.5 \
  --sample-sizes 20 \
  --runs 2 \
  --output results/smoke
```

## Test

```bash
pytest
```

