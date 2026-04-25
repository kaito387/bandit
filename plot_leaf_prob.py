"""Leaf-probability plotting entrypoint built on top of simulate.py core APIs."""

from __future__ import annotations

import argparse
from pathlib import Path

from simulate import load_environments, run_env_leaf_prob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leaf-probability plotting (reuses simulate.py core)")
    parser.add_argument("--input", required=True, help="path to JSON file (array of environments)")
    parser.add_argument("--output-dir", default="results_json", help="output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_file = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    envs = load_environments(input_file)
    for env in envs:
        run_env_leaf_prob(env, output_dir)


if __name__ == "__main__":
    main()
