"""Describe input amplification on DEVELOPMENT trajectories, without replaying policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

RAW_NAMES = [
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_vx",
    "ee_vy",
    "ee_vz",
    "finger_width",
    "object_x",
    "object_y",
    "object_z",
    "object_roll",
    "object_pitch",
    "object_yaw",
    "object_vx",
    "object_vy",
    "object_vz",
    "object_wx",
    "object_wy",
    "object_wz",
    "goal_x",
    "goal_y",
    "goal_z",
]
EXTRA_NAMES = [
    "object_minus_ee_x",
    "object_minus_ee_y",
    "object_minus_ee_z",
    "goal_minus_ee_x",
    "goal_minus_ee_y",
    "goal_minus_ee_z",
    "goal_minus_object_x",
    "goal_minus_object_y",
    "goal_minus_object_z",
    "ee_object_distance",
    "goal_object_distance",
]


def features(x: np.ndarray, relative: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if not relative:
        return x.astype(np.float64)
    ee, obj, goal = x[..., :3], x[..., 7:10], x[..., 19:22]
    extras = np.concatenate(
        [
            obj - ee,
            goal - ee,
            goal - obj,
            np.linalg.norm(obj - ee, axis=-1, keepdims=True),
            np.linalg.norm(goal - obj, axis=-1, keepdims=True),
        ],
        axis=-1,
    )
    return np.concatenate([x, extras], axis=-1).astype(np.float64)


def analyze(data_path: Path, trace_dirs: list[Path], floor: float) -> tuple[dict, list[dict]]:
    with np.load(data_path, allow_pickle=False) as data:
        observations, ids = data["observations"], data["episode_ids"]
    unique = np.unique(ids)
    order = np.random.default_rng(7).permutation(unique)
    n_val = min(max(round(0.2 * len(unique)), 1), len(unique) - 1)
    train_ids = np.sort(order[n_val:])
    training = observations[np.isin(ids, train_ids)]
    summaries, dimension_rows = [], []
    for directory in trace_dirs:
        relative = directory.name.startswith("gru") or "33" in directory.name
        names = RAW_NAMES + EXTRA_NAMES if relative else RAW_NAMES
        train_features = features(training, relative)
        mean, std = train_features.mean(0), train_features.std(0)
        arrays, files = [], []
        for path in sorted(directory.glob("*.npz")):
            with np.load(path, allow_pickle=False) as trace:
                arrays.append(features(trace["observations"], relative))
            files.append(
                {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
        if not arrays:
            continue
        x = np.concatenate(arrays)
        legacy = np.abs((x - mean) / np.maximum(std, 1e-6))
        stabilized = np.abs((x - mean) / np.maximum(std, floor))
        maxima = legacy.max(0)
        for dimension, name in enumerate(names):
            dimension_rows.append(
                {
                    "trace_condition": directory.name,
                    "dimension": dimension,
                    "feature": name,
                    "training_std": float(std[dimension]),
                    "training_mean": float(mean[dimension]),
                    "development_min": float(x[:, dimension].min()),
                    "development_max": float(x[:, dimension].max()),
                    "legacy_max_absolute_z": float(maxima[dimension]),
                    "floor_max_absolute_z": float(stabilized[:, dimension].max()),
                }
            )
        extreme = np.unravel_index(np.argmax(legacy), legacy.shape)
        per_episode, cursor = [], 0
        for item, trace in zip(files, arrays):
            count = len(trace)
            per_episode.append(
                {
                    **item,
                    "steps": count,
                    "legacy_max_absolute_z": float(legacy[cursor : cursor + count].max()),
                    "floor_max_absolute_z": float(stabilized[cursor : cursor + count].max()),
                }
            )
            cursor += count
        summaries.append(
            {
                "trace_condition": directory.name,
                "feature_count": len(names),
                "episodes": len(arrays),
                "steps": len(x),
                "legacy_max_absolute_z": float(legacy.max()),
                "floor_max_absolute_z": float(stabilized.max()),
                "legacy_q99_absolute_z": float(np.quantile(legacy, 0.99)),
                "floor_q99_absolute_z": float(np.quantile(stabilized, 0.99)),
                "legacy_steps_with_max_z_above100": int((legacy.max(1) > 100).sum()),
                "floor_steps_with_max_z_above100": int((stabilized.max(1) > 100).sum()),
                "extreme_feature": names[extreme[1]],
                "extreme_training_std": float(std[extreme[1]]),
                "top_features_by_legacy_max": [names[i] for i in np.argsort(maxima)[-8:][::-1]],
                "per_episode": per_episode,
            }
        )
    return {
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "train_ids": train_ids.tolist(),
        "training_rows": len(training),
        "legacy_std_floor": 1e-6,
        "comparison_std_floor": floor,
        "statistic_precision": "NumPy float64 population moments of float32 policy features",
        "scope": "Same saved development observations re-expressed under two normalizations; no new policy outcomes",
        "conditions": summaries,
    }, dimension_rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=root / "artifacts/revision_study")
    parser.add_argument("--output", type=Path, default=root.parent / "project_Aayush/revision")
    parser.add_argument("--floor", type=float, default=0.05)
    parser.add_argument("--trace-glob", default="*_traces_30000_24")
    args = parser.parse_args()
    if args.floor <= 0:
        parser.error("floor must be positive")
    trace_dirs = sorted(args.study.glob(args.trace_glob))
    if not trace_dirs:
        parser.error("No saved development trace directories match; no results produced")
    result, rows = analyze(args.study / "demos.npz", trace_dirs, args.floor)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "normalization_audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    with (args.output / "normalization_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Development-trajectory normalization audit",
        "",
        (
            "Training-only moments are recomputed in NumPy float64 from the split7 training episodes. "
            "They can differ in the last digits from float32 PyTorch checkpoint buffers. "
            "All trajectory seeds are development scenes; this audit does not inspect held-out test scenes."
        ),
        "",
        (
            "The same saved observations are normalized using standard-deviation floors1e-6 and0.05. "
            "This measures numerical amplification. The0.05 column does not describe a rerun "
            "of the policy or demonstrate that the normalization change improves success."
        ),
        "",
        "| Saved development condition | Episodes/steps | Maximum absolute standardized input,1e-6 | Maximum,0.05 | Largest-amplification feature | Its training std |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for item in result["conditions"]:
        lines.append(
            f"| {item['trace_condition']} | {item['episodes']}/{item['steps']} | "
            f"{item['legacy_max_absolute_z']:.3g} | {item['floor_max_absolute_z']:.3g} | "
            f"{item['extreme_feature']} | {item['extreme_training_std']:.3g} |"
        )
    lines += [
        "",
        (
            "The uniform floor mixes quantities with different physical units; it is an empirical "
            "preprocessing intervention, not a dimensionless physical bound. A held-out, matched "
            "policy comparison is required before claiming a behavioral benefit."
        ),
    ]
    (args.output / "normalization_audit.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "conditions": len(result["conditions"]),
                "report": str(args.output / "normalization_audit.md"),
            }
        )
    )


if __name__ == "__main__":
    main()
