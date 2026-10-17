#!/usr/bin/env python3
"""Run data-efficiency and network-width experiments and save publication-ready plots."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ilgrasp.data import DemoDatabase
from ilgrasp.evaluation import evaluate_policy
from ilgrasp.policies import NumpyPolicy, train_behavior_cloning
from ilgrasp.sim import GraspEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("artifacts/demos.db"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--demo-counts", type=int, nargs="+", default=[10, 25, 50, 100, 200])
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=80)
    return parser.parse_args()


def run_setting(db: DemoDatabase, env: GraspEnv, demos: int, width: int, seed: int, args: argparse.Namespace) -> float:
    data = db.load_training_data(max_episodes=demos)
    trained = train_behavior_cloning(data, hidden_sizes=(width, width), epochs=args.epochs, seed=seed)
    evaluation = evaluate_policy(
        env,
        NumpyPolicy(trained.model),
        n_episodes=args.eval_episodes,
        seed_start=20_000 + 1_000 * seed,
    )
    return evaluation.success_rate


def save_plot(rows: list[dict[str, float | int | str]], experiment: str, x_key: str, xlabel: str, path: Path) -> None:
    subset = [row for row in rows if row["experiment"] == experiment]
    xs = sorted({int(row[x_key]) for row in subset})
    means, spreads = [], []
    for x in xs:
        values = np.asarray([float(row["success_rate"]) for row in subset if row[x_key] == x])
        means.append(values.mean())
        spreads.append(values.std(ddof=1) if len(values) > 1 else 0.0)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.errorbar(xs, means, yerr=spreads, marker="o", capsize=4, color="#2563eb", lw=2)
    ax.set(xlabel=xlabel, ylabel="Success rate", ylim=(-0.03, 1.03), title=experiment.replace("_", " ").title())
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    with DemoDatabase(args.db) as db, GraspEnv() as env:
        for demos in args.demo_counts:
            for seed in args.seeds:
                rate = run_setting(db, env, demos, 128, seed, args)
                rows.append({"experiment": "data_efficiency", "demos": demos, "width": 128, "seed": seed, "success_rate": rate})
                print(f"data demos={demos:3d} seed={seed} -> {rate:.1%}")
        fixed_demos = max(args.demo_counts)
        for width in args.widths:
            for seed in args.seeds:
                rate = run_setting(db, env, fixed_demos, width, seed, args)
                rows.append({"experiment": "network_width", "demos": fixed_demos, "width": width, "seed": seed, "success_rate": rate})
                print(f"width={width:3d} seed={seed} -> {rate:.1%}")

    csv_path = args.output_dir / "results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "demos", "width", "seed", "success_rate"])
        writer.writeheader()
        writer.writerows(rows)
    save_plot(rows, "data_efficiency", "demos", "Successful demonstrations", args.output_dir / "data_efficiency.png")
    save_plot(rows, "network_width", "width", "Hidden-layer width", args.output_dir / "network_width.png")
    print("saved:", args.output_dir)


if __name__ == "__main__":
    main()
