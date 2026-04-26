usage

`simulate.py` is the single source of truth for environment parsing and simulation logic.
`plot_leaf_prob.py` is a thin wrapper that calls reusable APIs from `simulate.py`.
All plots/tables/summaries are logged to WandB.

```bash
python simulate.py --input testcases/demo.json
python plot_leaf_prob.py --input testcases/demo.json
python plot_summary_bars.py --input ref/caterpillarS2K5/summary.json
```

Optional: add `--wandb-mode offline` for local/offline logging.

## Tree Shape Sweep (K/S fixed, ratio sweep)

You can now sweep tree shape by fixing `K`, `S`, and `rounds`, then scanning `ratio`.
Each ratio value is logged as an independent WandB run, so algorithm comparisons are easy to filter.

### 1) WandB sweep agent

Sweep config file:

```text
sweeps/tree_shape_ratio_s4k4.yaml
```

Create sweep:

```bash
wandb sweep sweeps/tree_shape_ratio_s4k4.yaml
```

Then start one or more agents (replace with returned sweep id):

```bash
wandb agent kaito15-sun-yat-sen-university/bandit/<sweep_id>
```

### 2) Sweep metadata options

`simulate.py` keeps only lightweight sweep metadata options:

- `--sweep-group`: optional WandB group override.
- `--testcase-path`: logs the generated testcase path in run config.
