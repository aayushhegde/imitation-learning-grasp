#!/usr/bin/env python3
"""Collect scripted-expert episodes into SQLite."""

import argparse
from pathlib import Path

from ilgrasp.data import DemoDatabase
from ilgrasp.experts import ScriptedExpert, collect_expert_episode
from ilgrasp.sim import GraspEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("artifacts/demos.db"))
    parser.add_argument("--successful", type=int, default=200, help="successful demos to collect")
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="open the optional GUI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.successful <= 0 or args.max_attempts < args.successful:
        raise SystemExit("Require 0 < --successful <= --max-attempts")
    args.db.parent.mkdir(parents=True, exist_ok=True)
    expert = ScriptedExpert()
    successful = 0

    with GraspEnv(render_mode="human" if args.render else None) as env, DemoDatabase(args.db) as db:
        for attempt in range(args.max_attempts):
            seed = args.seed_start + attempt
            episode = collect_expert_episode(env, expert, seed=seed)
            db.save_episode(episode)
            usable = bool(episode.success and episode.usable_for_training)
            successful += int(usable)
            if (attempt + 1) % 10 == 0 or usable:
                print(
                    f"attempt {attempt + 1:4d} | successful {successful:4d}/{args.successful} "
                    f"| last phase {episode.phases[-1]:>9s} | steps {len(episode):3d}"
                )
            if successful >= args.successful:
                break
        print("database:", args.db)
        print("summary :", db.summary())

    if successful < args.successful:
        raise SystemExit(
            f"Only collected {successful} successes in {args.max_attempts} attempts. "
            "Inspect the expert phase plots before increasing max attempts."
        )


if __name__ == "__main__":
    main()
