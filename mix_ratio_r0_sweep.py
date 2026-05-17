#!/usr/bin/env python3
"""Sweep ratio -> mean root R0 for mix tree shapes.

This script repeatedly generates trees for mixcaterpillar and mix-full-binary
cases, preprocesses each generated environment, and averages the root R value
(R0) over many trials.

The result is meant to be exported and plotted later.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import generate_full_binary_case
from simulate import _prepare_environment, _validate_env_obj


def _ratio_tag(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _normalize_tree_shape(value: str) -> str:
    shape = value.strip().lower()
    aliases = {
        "mixfullbinary": "mix-full-binary",
        "mix-full-binary": "mix-full-binary",
        "mixcaterpillar": "mixcaterpillar",
        "mix-caterpillar": "mixcaterpillar",
        "both": "both",
        "all": "both",
    }
    if shape not in aliases:
        raise ValueError(f"unknown tree shape: {value}")
    return aliases[shape]


def _build_case(
    tree_shape: str,
    k: int,
    s: int | None,
    ratio: float,
    seed: int,
    algo: str,
) -> list[dict[str, Any]]:
    if tree_shape == "mix-full-binary":
        if s is None:
            raise ValueError("--S is required for mix-full-binary")
        env_name = f"mixFullBinaryTreeS{s}K{k}R{_ratio_tag(ratio)}"
        return generate_full_binary_case.generate_case_mix_full_binary(
            k=k,
            s=s,
            ratio=ratio,
            algo=algo,
            rounds=1,
            seed=seed,
            env_name=env_name,
        )

    if tree_shape == "mixcaterpillar":
        env_name = f"mixcaterpillarS2K{k}Ratio{_ratio_tag(ratio)}"
        return generate_full_binary_case.generate_case_mix_caterpillar(
            k=k,
            mix_ratio=ratio,
            algo=algo,
            rounds=1,
            seed=seed,
            env_name=env_name,
        )

    raise ValueError(f"unsupported tree shape: {tree_shape}")


def _estimate_mean_r0(
    tree_shape: str,
    k: int,
    s: int | None,
    ratio: float,
    trials: int,
    base_seed: int,
    algo: str,
) -> dict[str, float | int | str]:
    mean = 0.0
    m2 = 0.0
    min_r0 = None
    max_r0 = None

    for trial_idx in range(trials):
        seed = base_seed + trial_idx
        payload = _build_case(tree_shape=tree_shape, k=k, s=s, ratio=ratio, seed=seed, algo=algo)
        prepared = _prepare_environment(_validate_env_obj(payload[0], 0))
        r0 = float(prepared.risk[0])

        delta = r0 - mean
        mean += delta / float(trial_idx + 1)
        m2 += delta * (r0 - mean)

        if min_r0 is None or r0 < min_r0:
            min_r0 = r0
        if max_r0 is None or r0 > max_r0:
            max_r0 = r0

    std = (m2 / float(max(trials - 1, 1))) ** 0.5 if trials > 1 else 0.0
    return {
        "tree_shape": tree_shape,
        "ratio": float(ratio),
        "trials": int(trials),
        "mean_R0": float(mean),
        "std_R0": float(std),
        "min_R0": float(min_r0 if min_r0 is not None else 0.0),
        "max_R0": float(max_r0 if max_r0 is not None else 0.0),
    }


def _write_output(output_path: Path, rows: list[dict[str, float | int | str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        fieldnames = ["tree_shape", "ratio", "trials", "mean_R0", "std_R0", "min_R0", "max_R0"]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return

    payload = {
        "results": rows,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep ratio values and estimate the mean root R0 for mix tree shapes."
    )
    parser.add_argument(
        "--tree-shapes",
        nargs="+",
        default=["mixcaterpillar", "mix-full-binary"],
        help="Tree shapes to sweep: mixcaterpillar, mix-full-binary, or both",
    )
    parser.add_argument("--ratios", nargs="+", type=float, required=True, help="ratio values to evaluate")
    parser.add_argument("--K", type=int, required=True, help="tree depth K")
    parser.add_argument("--S", type=int, default=None, help="branching factor for mix-full-binary")
    parser.add_argument("--trials", type=int, default=100000, help="number of trees to average per ratio")
    parser.add_argument("--seed", type=int, default=42, help="base seed for tree generation")
    parser.add_argument("--algo", type=str, default="PS", help="algorithm code passed to the generator")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_json/mix_ratio_r0_sweep.json"),
        help="output file (.json or .csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.K < 1:
        raise ValueError("K must be >= 1")
    if args.trials < 1:
        raise ValueError("trials must be positive")

    tree_shapes = []
    for shape in args.tree_shapes:
        normalized = _normalize_tree_shape(shape)
        if normalized == "both":
            tree_shapes.extend(["mixcaterpillar", "mix-full-binary"])
        else:
            tree_shapes.append(normalized)

    # Preserve input order while dropping duplicates.
    seen = set()
    tree_shapes = [shape for shape in tree_shapes if not (shape in seen or seen.add(shape))]

    if "mix-full-binary" in tree_shapes and args.S is None:
        raise ValueError("--S is required when sweeping mix-full-binary")

    rows: list[dict[str, float | int | str]] = []
    for tree_shape in tree_shapes:
        for ratio in args.ratios:
            print(f"Sweeping {tree_shape} ratio={ratio} over {args.trials} trials")
            row = _estimate_mean_r0(
                tree_shape=tree_shape,
                k=args.K,
                s=args.S,
                ratio=ratio,
                trials=args.trials,
                base_seed=args.seed,
                algo=str(args.algo).strip().upper(),
            )
            rows.append(row)
            print(
                f"  mean_R0={row['mean_R0']:.6f}, std_R0={row['std_R0']:.6f}, "
                f"min_R0={row['min_R0']:.6f}, max_R0={row['max_R0']:.6f}"
            )

    _write_output(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
