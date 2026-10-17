#!/usr/bin/env python3
"""Train a behavior-cloning policy from successful SQLite demonstrations."""

import argparse
import json
from pathlib import Path

from ilgrasp.data import DemoDatabase
from ilgrasp.policies import train_behavior_cloning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("artifacts/demos.db"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/bc_policy.pt"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with DemoDatabase(args.db) as db:
        data = db.load_training_data(max_episodes=args.episodes)
    result = train_behavior_cloning(
        data,
        hidden_sizes=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.model.save(args.output)
    metrics_path = args.output.with_suffix(".training.json")
    metrics_path.write_text(
        json.dumps(
            {
                "train_loss": result.train_loss,
                "validation_loss": result.validation_loss,
                "train_episode_ids": result.train_episode_ids.tolist(),
                "validation_episode_ids": result.validation_episode_ids.tolist(),
            },
            indent=2,
        )
    )
    print(f"samples          : {len(data)}")
    print(f"training episodes: {len(result.train_episode_ids)}")
    print(f"validation eps   : {len(result.validation_episode_ids)}")
    print(f"final train loss : {result.train_loss[-1]:.6f}")
    print(f"model            : {args.output}")
    print(f"history          : {metrics_path}")


if __name__ == "__main__":
    main()

