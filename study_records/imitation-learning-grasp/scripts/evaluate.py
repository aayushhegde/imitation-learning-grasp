#!/usr/bin/env python3
"""Evaluate a trained policy on fresh, reproducible simulator seeds."""

import argparse
import json
from pathlib import Path

from ilgrasp.evaluation import evaluate_policy
from ilgrasp.policies import GraspPolicy, NumpyPolicy
from ilgrasp.sim import GraspEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/bc_policy.pt"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = NumpyPolicy(GraspPolicy.load(args.model))
    with GraspEnv(render_mode="human" if args.render else None) as env:
        result = evaluate_policy(env, policy, args.episodes, args.seed_start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.as_dict(), indent=2))
    low, high = result.confidence_interval
    print(f"success rate: {result.success_rate:.1%} ({int(result.successes.sum())}/{args.episodes})")
    print(f"95% CI      : [{low:.1%}, {high:.1%}]")
    print(f"results     : {args.output}")


if __name__ == "__main__":
    main()

