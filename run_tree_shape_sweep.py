#!/usr/bin/env python3
"""Run one tree-shape sweep trial from WandB agent parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import generate_full_binary_case
from simulate import ALGO_CODE_TO_ID, load_environments, simulate_single_env


def _parse_algo_code(code: str) -> str:
    algo = code.strip().upper()
    if not algo:
        raise ValueError("--algo must not be empty")
    if algo not in ALGO_CODE_TO_ID:
        raise ValueError(f"unknown algo code: {algo}")
    return algo


def _ratio_tag(ratio: float) -> str:
    text = f"{ratio:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _write_generated_case(
    out_dir: Path,
    tree_shape: str,
    k: int,
    s: int | None,
    ratio: float | None,
    r: int | None,
    algo: str,
    rounds: int,
    seed: int,
) -> Path:
    if tree_shape == "full-binary":
        if s is None or ratio is None:
            raise ValueError("--S and --ratio required for full-binary")
        ratio_tag = _ratio_tag(ratio)
        env_name = f"fullBinaryTreeS{s}K{k}R{ratio_tag}"
        payload = generate_full_binary_case.generate_case_full_binary(
            k=k,
            s=s,
            ratio=ratio,
            algo=algo,
            rounds=rounds,
            seed=seed,
            env_name=env_name,
        )
    elif tree_shape == "caterpillar":
        if r is None:
            r = 0
        env_name = f"caterpillarS2K{k}R{r}"
        payload = generate_full_binary_case.generate_case_caterpillar(
            k=k,
            r=r,
            algo=algo,
            rounds=rounds,
            seed=seed,
            env_name=env_name,
        )
    else:
        raise ValueError(f"unknown tree-shape: {tree_shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{env_name}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path


def _build_group_name(k: int, s: int | None, rounds: int, group: str | None, tree_shape: str) -> str:
    if s is not None:
        return group or f"tree-shape-sweep-{tree_shape}-S{s}K{k}-T{rounds}"
    else:
        return group or f"tree-shape-sweep-{tree_shape}-K{k}-T{rounds}"


def _run_one_trial(args: argparse.Namespace) -> None:
    testcase = _write_generated_case(
        out_dir=Path(args.generated_dir),
        tree_shape=args.tree_shape,
        k=args.K,
        s=args.S,
        ratio=args.ratio,
        r=args.R,
        algo=args.algo,
        rounds=args.rounds,
        seed=args.seed,
    )

    envs = load_environments(testcase)
    if len(envs) != 1:
        raise ValueError(f"expected one environment in generated testcase, got {len(envs)}")

    run_name = envs[0].env_name
    run_args = argparse.Namespace(
        wandb_group=args.wandb_group,
        sweep_group=_build_group_name(args.K, args.S, args.rounds, args.wandb_group, args.tree_shape),
        wandb_name=run_name,
        wandb_mode=args.wandb_mode,
        testcase_path=str(testcase),
    )

    simulate_single_env(envs[0], run_args, job_type="avg_regret")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one WandB sweep trial for tree-shape experiments")

    parser.add_argument(
        "--tree-shape",
        type=str,
        choices=["full-binary", "caterpillar"],
        default="full-binary",
        help="Tree shape (default: full-binary)",
    )
    parser.add_argument("--K", type=int, default=3, help="tree depth K")
    parser.add_argument("--rounds", type=int, default=2_000_000, help="simulation rounds")
    parser.add_argument("--seed", type=int, default=42, help="random seed for testcase generation")

    # Full-binary specific
    parser.add_argument("--S", type=int, default=None, help="[full-binary] branching factor")
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="[full-binary] fraction of leaves with g=1 (in [0, 1])",
    )

    # Caterpillar specific
    parser.add_argument("--R", type=int, default=0, help="[caterpillar] first R layers have g=0")

    parser.add_argument("--algo", type=str, required=True, help="single algo code from sweep config (e.g. PS, E3Q, EE3)")
    parser.add_argument("--generated-dir", default="testcases/generated", help="directory for generated testcase JSON")

    parser.add_argument("--leaf-prob", action="store_true", help="also log leaf probability plots")
    parser.add_argument("--wandb-group", default=None, help="optional WandB group override")
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="WandB mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.K < 1:
        raise ValueError("K must be >= 1")
    if args.rounds <= 0:
        raise ValueError("rounds must be positive")
    args.algo = _parse_algo_code(args.algo)

    if args.tree_shape == "full-binary":
        if args.S is None:
            raise ValueError("--S is required for full-binary")
        if args.ratio is None:
            raise ValueError("--ratio is required for full-binary")
        if args.S < 2:
            raise ValueError("S must be >= 2")
        if args.ratio < 0.0 or args.ratio > 1.0:
            raise ValueError("--ratio must be in [0, 1]")
    elif args.tree_shape == "caterpillar":
        if args.R < 0 or args.R > args.K:
            raise ValueError(f"R must be in [0, K] (K={args.K})")

    _run_one_trial(args)


if __name__ == "__main__":
    main()
