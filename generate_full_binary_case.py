#!/usr/bin/env python3
"""Generate tree case testcase JSON for simulate.py (full binary tree or caterpillar)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import tree_builders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a tree testcase JSON (full binary tree or caterpillar)."
    )
    parser.add_argument(
        "--tree-shape",
        type=str,
        choices=["full-binary", "caterpillar"],
        default="full-binary",
        help="Tree shape: 'full-binary' or 'caterpillar' (default: full-binary)",
    )

    # Common parameters
    parser.add_argument("--K", type=int, required=True, help="Tree depth (root depth is 0)")
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        help="Single algorithm code for this generated case (e.g. PS, E3Q, EE3)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5_000_000,
        help="Simulation rounds (default: 5000000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default=None,
        help="Environment name (default: auto-generated based on tree shape and parameters)",
    )

    # Full-binary specific
    parser.add_argument("--S", type=int, default=None, help="[full-binary] Branching factor")
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="[full-binary] Fraction of leaves marked as g=1 (in [0, 1])",
    )

    # Caterpillar specific
    parser.add_argument(
        "--R",
        type=int,
        default=0,
        help="[caterpillar] First R layers have g=0 (default: 0)",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.K < 1:
        raise ValueError("K must be >= 1")
    if args.rounds <= 0:
        raise ValueError("rounds must be positive")
    if not str(args.algo).strip():
        raise ValueError("algo must not be empty")

    if args.tree_shape == "full-binary":
        if args.S is None:
            raise ValueError("--S is required for full-binary tree")
        if args.ratio is None:
            raise ValueError("--ratio is required for full-binary tree")
        if args.S < 2:
            raise ValueError("S must be >= 2 for a non-trivial full tree")
        if not (0.0 <= args.ratio <= 1.0):
            raise ValueError("ratio must be in [0, 1]")
    elif args.tree_shape == "caterpillar":
        if args.R < 0 or args.R > args.K:
            raise ValueError(f"R must be in [0, K] (K={args.K})")


def generate_case_full_binary(
    k: int,
    s: int,
    ratio: float,
    algo: str,
    rounds: int,
    seed: int,
    env_name: str,
) -> list[dict]:
    """Generate full binary tree case (backward-compatible signature)."""
    rng = random.Random(seed)

    # Build tree using new builder
    builder = tree_builders.FullBinaryTreeBuilder(s=s, k=k)
    tree = builder.build()

    # Assign properties
    g = tree_builders.assign_g_values_full_binary(tree, ratio=ratio, rng=rng)
    p = tree_builders.assign_p_values_full_binary(tree, rng=rng)

    # Pick a random special leaf and mark it as TimeVariant with p=0.05
    rng_for_special = random.Random(seed + 1)
    special_leaf = tree_builders.select_random_leaf(tree.leaves, rng_for_special)
    special_leaf_idx = tree.leaves.index(special_leaf)
    p[special_leaf - 1] = 0.05
    distribution = tree_builders.assign_distribution(tree, special_leaf_idx, rng=rng)

    env = {
        "env_name": env_name,
        "algo": [algo],
        "seed": seed,
        "node_counts": tree.node_counts,
        "rounds": rounds,
        "parents": tree.parents,
        "g": g,
        "p": p,
        "distribution": distribution,
    }
    return [env]


def generate_case_caterpillar(
    k: int,
    r: int,
    algo: str,
    rounds: int,
    seed: int,
    env_name: str,
) -> list[dict]:
    """Generate caterpillar tree case."""
    rng = random.Random(seed)

    # Build tree using new builder
    builder = tree_builders.CaterpillarTreeBuilder(k=k)
    tree = builder.build()

    # Assign properties
    g = tree_builders.assign_g_values_caterpillar(tree, r=r)
    p = tree_builders.assign_p_values_caterpillar(tree, rng=rng)

    # Pick the last (deepest, rightmost) leaf for special treatment
    rng_for_special = random.Random(seed + 1)
    special_leaf_idx = len(tree.leaves) - 1  # Last leaf in the list
    special_leaf = tree.leaves[special_leaf_idx]
    p[special_leaf - 1] = 0.05
    distribution = tree_builders.assign_distribution(tree, special_leaf_idx, rng=rng)

    env = {
        "env_name": env_name,
        "algo": [algo],
        "seed": seed,
        "node_counts": tree.node_counts,
        "rounds": rounds,
        "parents": tree.parents,
        "g": g,
        "p": p,
        "distribution": distribution,
    }
    return [env]


def generate_case(
    k: int,
    s: int,
    ratio: float,
    algo: str,
    rounds: int,
    seed: int,
    env_name: str,
) -> list[dict]:
    """Legacy backward-compatible function for full binary tree generation."""
    return generate_case_full_binary(
        k=k,
        s=s,
        ratio=ratio,
        algo=algo,
        rounds=rounds,
        seed=seed,
        env_name=env_name,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    algo_code = str(args.algo).strip().upper()

    if args.tree_shape == "full-binary":
        env_name = args.env_name or f"fullBinaryTreeS{args.S}K{args.K}R{args.ratio}"
        payload = generate_case_full_binary(
            k=args.K,
            s=args.S,
            ratio=args.ratio,
            algo=algo_code,
            rounds=args.rounds,
            seed=args.seed,
            env_name=env_name,
        )
    elif args.tree_shape == "caterpillar":
        env_name = args.env_name or f"caterpillarS2K{args.K}R{args.R}"
        payload = generate_case_caterpillar(
            k=args.K,
            r=args.R,
            algo=algo_code,
            rounds=args.rounds,
            seed=args.seed,
            env_name=env_name,
        )
    else:
        raise ValueError(f"Unknown tree shape: {args.tree_shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
