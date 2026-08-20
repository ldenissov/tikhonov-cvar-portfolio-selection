# Numerical Experiments for the Master's Thesis Tikhonov-regularized equilibrium selection for CVaR-based multi-portfolio games by Lucas Denissov. 

## Setup

Python 3.12 with numpy, scipy, matplotlib, tqdm and the package
`Regularization-Methods-for-HVIs`.

## Running

```
python experiments/standard_example.py     # Tables 3-5, Figure 2
python experiments/structured_example.py   # Tables 6-8, Figure 3
python smoothing/chks_smoothing.py         # Figure 1
```

Tables are written to `tables/`, figures to `figures/`. The seed is fixed, so
the results are reproducible.

## Layout

- `model/` - model, operators, projections, sampling
- `experiments/` - one script per example
- `smoothing/` - plot of the smoothing function
