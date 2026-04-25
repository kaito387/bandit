usage

`simulate.py` is the single source of truth for environment parsing and simulation logic.
`plot_leaf_prob.py` is a thin wrapper that calls reusable APIs from `simulate.py`.

```bash
python simulate.py --input testcases/demo.json
python plot_leaf_prob.py --input testcases/demo.json
```
