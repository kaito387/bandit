usage

`simulate.py` is the single source of truth for environment parsing and simulation logic.
`plot_leaf_prob.py` is a thin wrapper that calls reusable APIs from `simulate.py`.
All plots/tables/summaries are logged to WandB.

```bash
python simulate.py --input testcases/demo.json --wandb-project bandit --wandb-entity kaito15-sun-yat-sen-university
python plot_leaf_prob.py --input testcases/demo.json --wandb-project bandit --wandb-entity kaito15-sun-yat-sen-university
python plot_summary_bars.py --input ref/caterpillarS2K5/summary.json --wandb-project bandit --wandb-entity kaito15-sun-yat-sen-university
```

Optional: add `--wandb-mode offline` for local/offline logging.
