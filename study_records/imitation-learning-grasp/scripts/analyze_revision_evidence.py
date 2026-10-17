"""Descriptive phase-boundary diagnostic; frozen protocol lives in revision/neighbor_protocol.md."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

PHASES = (
    "approach",
    "descend",
    "close",
    "test_grasp",
    "lift_clear",
    "pre_cross",
    "cross_barrier",
    "above_goal",
    "lower",
    "open",
    "retreat_up",
    "retreat_home",
    "verify",
)
EXPECTED = list(pairwise(PHASES)) + [("test_grasp", "approach")]
TIMED = {("close", "test_grasp"), ("open", "retreat_up")}
EPS_OBS = 0.5
EPS_ACTION = 0.5
SPLIT_SEED = 7
BOOTSTRAP_SEED = 20260925


def geometric_guard_residual(before: str, after: str, obs: np.ndarray) -> float | None:
    """Negative means the geometric transition guard passes; distances in metres."""
    ee, obj, goal = obs[:3], obs[7:10], obs[19:22]
    if before == "approach":
        return float(np.linalg.norm(ee - obj - [0, 0, 0.12]) - 0.018)
    if before == "descend":
        return float(np.linalg.norm(ee - obj - [0, 0, 0.015]) - 0.018)
    if (before, after) == ("test_grasp", "lift_clear"):
        return float(0.05 - obj[2])
    if before in {"lift_clear", "retreat_up"}:
        return float(abs(ee[2] - 0.17) - 0.018)
    if before == "pre_cross":
        return float(np.linalg.norm(ee - [goal[0], -0.075, 0.17]) - 0.018)
    if before == "cross_barrier":
        return float(np.linalg.norm(ee - [goal[0], 0.075, 0.17]) - 0.018)
    if before == "above_goal":
        return float(np.linalg.norm(ee - [goal[0], goal[1], 0.17]) - 0.018)
    if before == "lower":
        return float(np.linalg.norm(obj - goal) - 0.035)
    if before == "retreat_home":
        return float(np.linalg.norm(ee - [0.04, 0, 0.20]) - 0.025)
    return None


def mechanism(boundary: tuple[str, str]) -> str:
    if boundary in TIMED:
        return "timed"
    if boundary == ("test_grasp", "approach"):
        return "timeout_and_failed_lift"
    return "geometric"


def split_ids(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(ids)
    if len(unique) < 2:
        raise ValueError("At least two trajectories are required for an episode split")
    order = np.random.default_rng(SPLIT_SEED).permutation(unique)
    n_val = min(max(round(0.2 * len(unique)), 1), len(unique) - 1)
    return np.sort(order[n_val:]), np.sort(order[:n_val])


def load_data(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        data = {key: raw[key] for key in ["observations", "actions", "phases", "episode_ids"]}
    n = len(data["observations"])
    if data["observations"].shape != (n, 22) or data["actions"].shape != (n, 4):
        raise ValueError("Expected observations(N,22), actions(N,4)")
    if any(len(v) != n for v in data.values()):
        raise ValueError("Misaligned arrays")
    if not np.isfinite(data["observations"]).all() or not np.isfinite(data["actions"]).all():
        raise ValueError("Nonfinite dataset values")
    data["phases"] = data["phases"].astype(str)
    unknown = set(np.unique(data["phases"])) - set(PHASES)
    if unknown:
        raise ValueError(f"Unknown phase labels: {unknown}")
    train_ids, val_ids = split_ids(data["episode_ids"])
    return data, train_ids, val_ids


def match_boundaries(data: dict[str, np.ndarray], train_ids: np.ndarray) -> list[dict]:
    obs = data["observations"].astype(np.float64)
    mask = np.isin(data["episode_ids"], train_ids)
    mean = obs[mask].mean(0)
    scale = np.maximum(obs[mask].std(0), 1e-6)
    standardized = (obs - mean) / scale
    boundaries: dict[tuple[str, str], list[tuple[int, int, int]]] = {key: [] for key in EXPECTED}
    for episode_id in train_ids:
        indices = np.flatnonzero(data["episode_ids"] == episode_id)
        for a, b in pairwise(indices):
            key = (data["phases"][a], data["phases"][b])
            if key[0] != key[1]:
                boundaries.setdefault(key, []).append((int(episode_id), int(a), int(b)))

    records: list[dict] = []
    for key, endpoints in boundaries.items():
        for episode_id, left, right in endpoints:
            candidates = [(ep, a, b) for ep, a, b in endpoints if ep != episode_id]
            selections = [("adjacent", left, right, episode_id)]
            if candidates:
                target = min(
                    candidates,
                    key=lambda row: np.linalg.norm(standardized[left] - standardized[row[2]]),
                )
                selections.append(("cross_forward", left, target[2], target[0]))
                target = min(
                    candidates,
                    key=lambda row: np.linalg.norm(standardized[right] - standardized[row[1]]),
                )
                selections.append(("cross_reverse", right, target[1], target[0]))
            for kind, query, reference, reference_episode in selections:
                delta_o = float(np.linalg.norm(standardized[query] - standardized[reference]))
                delta_a = float(np.linalg.norm(data["actions"][query] - data["actions"][reference]))
                records.append(
                    {
                        "boundary": "->".join(key),
                        "mechanism": mechanism(key),
                        "comparison": kind,
                        "query_episode": episode_id,
                        "reference_episode": reference_episode,
                        "query_row": query,
                        "reference_row": reference,
                        "observation_distance": delta_o,
                        "action_distance": delta_a,
                        "close_neighbor": delta_o <= EPS_OBS,
                        "action_inconsistent": delta_a > EPS_ACTION,
                        "guard_residual_m": geometric_guard_residual(*key, obs[left]),
                    }
                )
    return records


def quantiles(values: list[float]) -> list[float] | None:
    if not values:
        return None
    return np.quantile(values, [0.25, 0.5, 0.75]).tolist()


def summarize(records: list[dict], train_ids: np.ndarray, n_bootstrap: int) -> list[dict]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # Whole-query-trajectory resampling; the cross-trajectory reference pool stays fixed.
    draws = rng.integers(0, len(train_ids), size=(n_bootstrap, len(train_ids)))
    multiplicities = np.zeros((n_bootstrap, len(train_ids)), dtype=np.int64)
    for draw_index, draw in enumerate(draws):
        multiplicities[draw_index] = np.bincount(draw, minlength=len(train_ids))
    id_index = {int(ep): i for i, ep in enumerate(train_ids)}
    keys = [
        ("->".join(key), kind)
        for key in EXPECTED
        for kind in ["adjacent", "cross_forward", "cross_reverse"]
    ]
    keys += sorted({(r["boundary"], r["comparison"]) for r in records} - set(keys))
    output = []
    for boundary, comparison in keys:
        selected = [
            r for r in records if (r["boundary"], r["comparison"]) == (boundary, comparison)
        ]
        total = np.zeros(len(train_ids), dtype=np.int64)
        close = total.copy()
        inconsistent = total.copy()
        both = total.copy()
        for row in selected:
            index = id_index[row["query_episode"]]
            total[index] += 1
            close[index] += row["close_neighbor"]
            inconsistent[index] += row["action_inconsistent"]
            both[index] += row["close_neighbor"] and row["action_inconsistent"]

        def rate(numerator: np.ndarray, denominator: np.ndarray) -> dict:
            if not denominator.sum():
                return {"value": None, "ci95": None, "bootstrap_nonempty_draws": 0}
            numer = multiplicities @ numerator
            denom = multiplicities @ denominator
            valid = denom > 0
            interval = np.quantile(numer[valid] / denom[valid], [0.025, 0.975]).tolist()
            return {
                "value": float(numerator.sum() / denominator.sum()),
                "ci95": interval,
                "bootstrap_nonempty_draws": int(valid.sum()),
            }

        output.append(
            {
                "boundary": boundary,
                "mechanism": mechanism(tuple(boundary.split("->"))),
                "comparison": comparison,
                "n_pairs": len(selected),
                "n_query_episodes": int((total > 0).sum()),
                "n_close_pairs": int(close.sum()),
                "n_inconsistent_close_pairs": int(both.sum()),
                "observation_distance_q25_median_q75": quantiles(
                    [r["observation_distance"] for r in selected]
                ),
                "action_distance_q25_median_q75": quantiles(
                    [r["action_distance"] for r in selected]
                ),
                "guard_residual_m_q25_median_q75": quantiles(
                    [r["guard_residual_m"] for r in selected if r["guard_residual_m"] is not None]
                ),
                "close_neighbor_rate": rate(close, total),
                "action_inconsistency_rate": rate(inconsistent, total),
                "conditional_action_inconsistency_rate": rate(both, close),
            }
        )
    return output


def render_report(result: dict, target: Path) -> None:
    def interval(entry: dict) -> str:
        if entry["value"] is None:
            return "unavailable"
        return f"{entry['value']:.3f} [{entry['ci95'][0]:.3f}, {entry['ci95'][1]:.3f}]"

    lines = [
        "# Revision demonstration boundary diagnostic",
        "",
        (
            f"Dataset SHA-256: `{result['dataset_sha256']}`. Training/validation episodes: "
            f"{len(result['train_ids'])}/{len(result['validation_ids'])}. Split seed7. "
            f"Training transitions: {result['training_transitions']}."
        ),
        "",
        (
            "Raw22 observations are standardized using training episodes only. Distances are L2; "
            "observation neighborhood <=0.5; action inconsistency >0.5. Cross-trajectory matching "
            "uses one nearest opposite-phase boundary endpoint from a different trajectory. "
            "95% intervals cluster-bootstrap query episodes with the reference pool held fixed."
        ),
        "",
        "| Boundary | Comparison | Pairs | Median obs distance | Median action distance | Close pairs | Inconsistent among close (95% CI) |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in result["boundaries"]:
        obs_q = row["observation_distance_q25_median_q75"]
        act_q = row["action_distance_q25_median_q75"]
        obs_display = (
            f"{obs_q[1]:.2e}"
            if obs_q and 0 < obs_q[1] < 1e-3
            else f"{obs_q[1]:.3f}"
            if obs_q
            else "NA"
        )
        act_display = f"{act_q[1]:.3f}" if act_q else "NA"
        lines.append(
            f"| {row['boundary']} | {row['comparison']} | {row['n_pairs']} | "
            f"{obs_display} | {act_display} | {row['n_close_pairs']} | "
            f"{interval(row['conditional_action_inconsistency_rate'])} |"
        )
    lines += [
        "",
        (
            "These are descriptive neighborhood statistics. Zero observed conditional events "
            "can produce a degenerate bootstrap interval; this does not prove a population "
            "rate of zero. Unavailable conditional rates mean no sampled neighbor passed the "
            "observation threshold. Geometric guards provide controls; this analysis by itself "
            "does not establish that action selection requires hidden memory."
        ),
    ]
    target.write_text("\n".join(lines) + "\n")


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=repository / "artifacts/revision_study/demos.npz"
    )
    parser.add_argument(
        "--output", type=Path, default=repository.parent / "project_Aayush/revision"
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset does not exist; refusing to fabricate results: {args.data}")
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")
    data, train_ids, val_ids = load_data(args.data)
    records = match_boundaries(data, train_ids)
    result = {
        "dataset": str(args.data.resolve()),
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "protocol": {
            "split_seed": SPLIT_SEED,
            "epsilon_obs": EPS_OBS,
            "epsilon_action": EPS_ACTION,
            "neighbors": 1,
            "bootstrap_draws": args.bootstrap,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "cross_bootstrap_reference_pool": "fixed",
        },
        "train_ids": train_ids.tolist(),
        "validation_ids": val_ids.tolist(),
        "training_transitions": int(np.isin(data["episode_ids"], train_ids).sum()),
        "boundaries": summarize(records, train_ids, args.bootstrap),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "neighbor_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    if records:
        with (args.output / "neighbor_pairs.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    render_report(result, args.output / "neighbor_results.md")
    print(
        json.dumps(
            {
                "training_episodes": len(train_ids),
                "training_transitions": result["training_transitions"],
                "pairs": len(records),
                "report": str(args.output / "neighbor_results.md"),
            }
        )
    )


if __name__ == "__main__":
    main()
