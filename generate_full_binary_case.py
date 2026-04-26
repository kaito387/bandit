#!/usr/bin/env python3
"""Generate a full binary tree testcase JSON for simulate.py."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a full binary tree testcase JSON (single-env array)."
    )
    parser.add_argument("--K", type=int, required=True, help="Tree depth (root depth is 0)")
    parser.add_argument("--S", type=int, required=True, help="Branching factor")
    parser.add_argument(
        "--ratio",
        type=float,
        required=True,
        help="Probability each non-root node has g=1",
    )
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
        help="Environment name (default: fullBinaryTreeS{S}K{K}R{ratio})",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.K < 1:
        raise ValueError("K must be >= 1")
    if args.S < 2:
        raise ValueError("S must be >= 2 for a non-trivial full tree")
    if args.rounds <= 0:
        raise ValueError("rounds must be positive")
    if not (0.0 <= args.ratio <= 1.0):
        raise ValueError("ratio must be in [0, 1]")
    if not str(args.algo).strip():
        raise ValueError("algo must not be empty")


def full_tree_node_count(s: int, k: int) -> int:
    # Total nodes of an S-ary full tree with depth K: sum_{i=0..K} S^i
    return (s ** (k + 1) - 1) // (s - 1)


def build_parents(s: int, n: int) -> list[int]:
    parents: list[int] = []
    for node in range(1, n):
        parent = (node - 1) // s
        parents.append(parent)
    return parents


def leaf_start_index(s: int, k: int) -> int:
    # Nodes before the last level: sum_{i=0..K-1} S^i
    return (s**k - 1) // (s - 1)


def generate_case(
    k: int,
    s: int,
    ratio: float,
    algo: str,
    rounds: int,
    seed: int,
    env_name: str,
) -> list[dict]:
    rng = random.Random(seed)

    node_counts = full_tree_node_count(s, k)
    parents = build_parents(s, node_counts)

    g = [1 if rng.random() < ratio else 0 for _ in range(node_counts - 1)]

    p = [0.0] * (node_counts - 1)
    first_leaf = leaf_start_index(s, k)
    for node in range(first_leaf, node_counts):
        idx = node - 1
        p[idx] = round(rng.uniform(0.05, 0.95), 6)

    env = {
        "env_name": env_name,
        "algo": [algo],
        "seed": seed,
        "node_counts": node_counts,
        "rounds": rounds,
        "parents": parents,
        "g": g,
        "p": p,
        "distribution": {},
    }
    return [env]


def main() -> None:
    args = parse_args()
    validate_args(args)

    env_name = args.env_name or f"fullBinaryTreeS{args.S}K{args.K}R{args.ratio}"

    payload = generate_case(
        k=args.K,
        s=args.S,
        ratio=args.ratio,
        algo=str(args.algo).strip().upper(),
        rounds=args.rounds,
        seed=args.seed,
        env_name=env_name,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
