"""Leaf-probability plotting entrypoint built on top of simulate.py core APIs."""

from __future__ import annotations

import argparse
from pathlib import Path

from simulate import _init_wandb_run, load_environments, run_env_leaf_prob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leaf-probability plotting (reuses simulate.py core)")
    parser.add_argument("--input", required=True, help="path to JSON file (array of environments)")
    parser.add_argument("--output-dir", default="results_json", help="unused; kept for backward compatibility")
    parser.add_argument("--wandb-project", default="bandit", help="WandB project name")
    parser.add_argument("--wandb-entity", default="kaito15-sun-yat-sen-university", help="WandB entity/team")
    parser.add_argument("--wandb-group", default=None, help="optional WandB group")
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="WandB mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_file = Path(args.input)
    output_dir = Path(args.output_dir)

    envs = load_environments(input_file)
    for env in envs:
        run = _init_wandb_run(args, env, job_type="leaf_prob")
        try:
            run_env_leaf_prob(env, output_dir)
        finally:
            run.finish()


if __name__ == "__main__":
    main()
