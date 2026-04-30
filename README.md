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

## Tree Shape Generation

Two tree shapes are supported: **full-binary** and **caterpillar**.

### Full Binary Tree

A full S-ary tree with depth K (root at depth 0). Properties:
- Total nodes: $(S^{K+1} - 1) / (S - 1)$
- Leaves: $S^K$
- Structure: All internal nodes have exactly S children.

**Parametrization:**
- `--ratio`: Fraction of leaves (from left in DFS order) marked as `g=1` (0–1).
- Leaves with `g=1` and their ancestors are marked `g=1`; others are `g=0` (except root, which is always `g=1`).
- All leaves sample $p \in [0.2, 0.8]$ uniformly.
- One random leaf is marked $p=0.05$ and `distribution=TIMEVARIANT`.

**Example: generate a binary tree (S=2, K=3, ratio=0.5)**

```bash
python generate_full_binary_case.py \
  --tree-shape full-binary \
  --K 3 --S 2 --ratio 0.5 \
  --algo PS --rounds 100000 --seed 42 \
  --output testcases/generated/fullBinaryTreeS2K3R0.5.json
```

### Caterpillar Tree

A "backbone" tree where:
- Root has left child (next backbone node) and right child (leaf).
- Each backbone node at depth $d < K$ has a left child that continues the backbone and a right child (leaf).
- The deepest backbone node at depth $K$ is a leaf.
- Total nodes: $2K + 1$
- Total leaves: $K + 1$

**Parametrization:**
- `--R`: First R layers (depth 0 to R inclusive) have `g=0`; others have `g=1`.
- Leaves sample $p$ from depth-dependent intervals: deeper leaves sample from higher values in [0.2, 0.8].
- The deepest leaf (last in the leaves list) is marked $p=0.05$ and `distribution=TIMEVARIANT`.

**Example: generate a caterpillar tree (K=2, R=0)**

```bash
python generate_full_binary_case.py \
  --tree-shape caterpillar \
  --K 2 --R 0 \
  --algo PS --rounds 100000 --seed 42 \
  --output testcases/generated/caterpillarS2K2R0.json
```

## Tree Shape Sweep (K/S fixed, ratio sweep)

You can now sweep tree shape by fixing `K`, `S`, and `rounds`, then scanning `ratio`.
Each ratio value is logged as an independent WandB run, so algorithm comparisons are easy to filter.

### 1) WandB sweep agent (full-binary example)

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

### 2) Manual sweep (multi-shape)

To sweep multiple shapes or customize parameters, use `run_tree_shape_sweep.py`:

**Full-binary sweep:**
```bash
python run_tree_shape_sweep.py \
  --tree-shape full-binary \
  --K 3 --S 2 --ratio 0.5 \
  --algo PS --rounds 100000
```

**Caterpillar sweep:**
```bash
python run_tree_shape_sweep.py \
  --tree-shape caterpillar \
  --K 3 --R 1 \
  --algo PS --rounds 100000
```

### 3) Sweep metadata options

`simulate.py` keeps only lightweight sweep metadata options:

- `--sweep-group`: optional WandB group override.
- `--testcase-path`: logs the generated testcase path in run config.
