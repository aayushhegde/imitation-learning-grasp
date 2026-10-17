"""Audit the recovery oracle or collect one mixed-policy DAgger round.

Collection saves contiguous reset-to-stop histories, including failed attempts.
An oracle abstention ends that history; no fabricated fallback action is used
as a training label. Teacher-mixture success is not learner-only success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ilgrasp.experts import HistoryAwareRecoveryOracle, VerifiedExpertPhase, VerifiedPickPlaceExpert
from ilgrasp.sim import VerifiedPlacementEnv


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
                               default=lambda item: item.item() if isinstance(item, np.generic)
                               else item.tolist()) + "\n")


def validate(output: Path, seed_start: int, count: int) -> None:
    records = []
    for condition in ("oracle", "transport_perturbation", "open_during_close", "expert_history"):
        with VerifiedPlacementEnv(reset_mode="independent") as env:
            for seed in range(seed_start, seed_start + count):
                env.reset(seed=seed)
                oracle = HistoryAwareRecoveryOracle(env)
                expert = VerifiedPickPlaceExpert()
                interventions = 0
                abstention = None
                queries = 0
                phase_matches = 0
                for _ in range(env.config.max_episode_steps):
                    label = oracle.query()
                    queries += 1
                    if not label.valid:
                        abstention = label.reason
                        break
                    action = label.action
                    if condition == "transport_perturbation" and interventions < 3:
                        if oracle.phase is VerifiedExpertPhase.CROSS_BARRIER:
                            action = np.array([0.3, -0.2, 0.0, -1.0], dtype=np.float32)
                            interventions += 1
                    elif condition == "open_during_close" and interventions < 3:
                        if oracle.phase is VerifiedExpertPhase.CLOSE:
                            action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                            interventions += 1
                    elif condition == "expert_history":
                        phase_matches += int(label.phase == expert.phase.value)
                        action = expert.act(env.raw_obs)
                    _, _, done, truncated, _ = env.step(action)
                    oracle.observe_transition()
                    if done or truncated:
                        break
                records.append({"condition": condition, "seed": seed, "interventions": interventions,
                                "queries": queries, "phase_matches": phase_matches,
                                "abstention": abstention, **env.last_info})
        summary = {
            "condition": condition,
            "episodes": count,
            "strict_successes": sum(r["strict_success"] for r in records if r["condition"] == condition),
            "abstentions": dict(Counter(r["abstention"] for r in records
                                        if r["condition"] == condition and r["abstention"])),
        }
        print(json.dumps(summary), flush=True)
        write_json(output / "oracle_validation.json", records)


def collect(output: Path, checkpoint: Path, seed_start: int, count: int, beta: float,
            mixture_seed: int) -> None:
    import torch

    from ilgrasp.policies.revision import RecurrentHybridPolicy, RecurrentNumpyPolicy

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
    model = RecurrentHybridPolicy(std_floor=saved["std_floor"])
    model.load_state_dict(saved["model"])
    learner = RecurrentNumpyPolicy(model)
    rng = np.random.default_rng(mixture_seed)
    xs, labels, phases, ids, executed, teacher_used = [], [], [], [], [], []
    episodes = []
    with VerifiedPlacementEnv(reset_mode="independent") as env:
        for seed in range(seed_start, seed_start + count):
            observation, _ = env.reset(seed=seed)
            learner.reset()
            oracle = HistoryAwareRecoveryOracle(env)
            start = len(xs)
            abstention = None
            for _ in range(env.config.max_episode_steps):
                # Advance the learner's recurrent state on every visited
                # observation, including steps on which the teacher acts.
                learner_action = learner(observation)
                label = oracle.query()
                if not label.valid:
                    abstention = label.reason
                    break
                use_teacher = bool(rng.random() < beta)
                actual = label.action if use_teacher else learner_action
                xs.append(observation.copy())
                labels.append(label.action.copy())
                phases.append(label.phase)
                ids.append(seed)
                executed.append(actual.copy())
                teacher_used.append(use_teacher)
                observation, _, done, truncated, _ = env.step(actual)
                oracle.observe_transition()
                if done or truncated:
                    break
            episodes.append({"seed": seed, "row_start": start, "row_stop": len(xs),
                             "abstention": abstention, "final_observation": observation.tolist(),
                             **env.last_info})
            if len(episodes) % 10 == 0:
                print(f"collected {len(episodes)}/{count} histories, {len(xs)} labels", flush=True)
    np.savez_compressed(
        output / "dagger_round1.npz", observations=np.asarray(xs, dtype=np.float32),
        actions=np.asarray(labels, dtype=np.float32), phases=np.asarray(phases),
        episode_ids=np.asarray(ids, dtype=np.int64),
        executed_actions=np.asarray(executed, dtype=np.float32),
        teacher_used=np.asarray(teacher_used, dtype=bool),
    )
    write_json(output / "dagger_round1_episodes.json", episodes)
    manifest = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "oracle_sha256": hashlib.sha256((ROOT / "src/ilgrasp/experts/recovery.py").read_bytes()).hexdigest(),
        "reset_mode": "independent", "beta": beta, "mixture_seed": mixture_seed,
        "seeds": list(range(seed_start, seed_start + count)), "rows": len(xs),
        "teacher_actions": int(sum(teacher_used)),
        "mixed_policy_successes": int(sum(r["strict_success"] for r in episodes)),
        "abstentions": dict(Counter(r["abstention"] for r in episodes if r["abstention"])),
        "training_rule": "All valid contiguous prefixes, irrespective of success; stop on abstention",
        "claim_limit": "Mixed-policy collection is not a learner-only DAgger evaluation",
    }
    write_json(output / "dagger_round1_manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


def train(output: Path, checkpoint: Path, epochs: int, seed: int) -> None:
    """Retrain from scratch with the baseline's original validation episodes."""
    import torch

    from ilgrasp.data import TrainingData
    from ilgrasp.policies.revision import train_recurrent_hybrid

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    baseline = torch.load(checkpoint, weights_only=False, map_location="cpu")
    original = np.load(output / "demos.npz")
    aggregated = np.load(output / "dagger_round1.npz")
    if np.intersect1d(original["episode_ids"], aggregated["episode_ids"]).size:
        raise ValueError("Original and aggregated episode IDs must be disjoint")
    data = TrainingData(**{
        key: np.concatenate([original[key], aggregated[key]])
        for key in ("observations", "actions", "episode_ids", "phases")
    })
    started = time.time()
    trained = train_recurrent_hybrid(
        data, epochs=epochs, seed=seed, std_floor=baseline["std_floor"],
        validation_ids=np.asarray(baseline["validation_ids"]),
    )
    np.testing.assert_array_equal(trained.validation_ids, baseline["validation_ids"])
    metadata = {
        "train_ids": trained.train_ids.tolist(), "validation_ids": trained.validation_ids.tolist(),
        "train_loss": trained.train_loss, "validation_loss": trained.validation_loss,
        "std_floor": baseline["std_floor"], "epochs": epochs, "seed": seed,
        "elapsed_seconds": time.time() - started,
        "best_epoch": int(np.argmin(trained.validation_loss)) + 1,
        "original_rows": len(original["observations"]), "aggregated_rows": len(aggregated["observations"]),
        "baseline_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "dataset_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                           for name in ("demos.npz", "dagger_round1.npz")},
        "initialization": "from scratch with original training seed; validation episodes unchanged",
    }
    label = f"dagger_gru33_floor_s{seed}"
    torch.save({"model": trained.model.state_dict(), **metadata}, output / f"{label}.pt")
    write_json(output / f"{label}_training.json", metadata)
    print(f"Trained {label}: {metadata['aggregated_rows']} new rows, "
          f"best epoch {metadata['best_epoch']}, {metadata['elapsed_seconds']:.1f}s", flush=True)


def replay_collection(output: Path) -> None:
    """Verify every recorded transition and retain the invalid final states."""
    recorded = np.load(output / "dagger_round1.npz")
    episodes = json.loads((output / "dagger_round1_episodes.json").read_text())
    with VerifiedPlacementEnv(reset_mode="independent") as env:
        for record in episodes:
            observation, _ = env.reset(seed=record["seed"])
            for index in range(record["row_start"], record["row_stop"]):
                np.testing.assert_array_equal(observation, recorded["observations"][index])
                observation, _, _, _, _ = env.step(recorded["executed_actions"][index])
            record["final_observation"] = observation.tolist()
            record["exact_replay_verified"] = True
    write_json(output / "dagger_round1_episodes.json", episodes)
    print(f"Exact physical replay verified for all {len(episodes)} collected histories", flush=True)


def evaluate_checkpoint(output: Path, checkpoint: Path, seed_start: int, count: int) -> None:
    import torch
    from run_revision_study import evaluate, summarize

    from ilgrasp.policies.revision import RecurrentHybridPolicy, RecurrentNumpyPolicy

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
    model = RecurrentHybridPolicy(std_floor=saved["std_floor"])
    model.load_state_dict(saved["model"])
    label = checkpoint.stem
    stamp = f"{seed_start}_{count}"
    result_path = output / f"{label}_eval_{stamp}.json"
    if result_path.exists():
        raise FileExistsError("Held-out result already exists; do not overwrite it")
    manifest = {
        "frozen_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "reset_mode": "independent", "seeds": list(range(seed_start, seed_start + count)),
        "learner_only": True, "oracle_used_at_evaluation": False,
        "selection": "Checkpoint selected on original expert validation episodes before test evaluation",
    }
    write_json(output / f"{label}_eval_manifest_{stamp}.json", manifest)
    trace_dir = output / f"{label}_traces_{stamp}"
    trace_dir.mkdir(exist_ok=True)
    results = evaluate(RecurrentNumpyPolicy(model), range(seed_start, seed_start + count),
                       reset_mode="independent", trajectories=trace_dir)
    write_json(result_path, results)
    print(json.dumps(summarize(results)), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["validate", "collect", "train", "replay", "evaluate"])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/revision_study")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--mixture-seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--train-seed", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.beta <= 1:
        parser.error("beta must be in [0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "validate":
        validate(args.output, args.seed_start if args.seed_start is not None else 2000,
                 args.episodes if args.episodes is not None else 30)
    elif args.mode == "collect":
        if args.checkpoint is None:
            parser.error("collect requires --checkpoint")
        collect(args.output, args.checkpoint,
                args.seed_start if args.seed_start is not None else 1000,
                args.episodes if args.episodes is not None else 100,
                args.beta, args.mixture_seed)
    elif args.mode == "train":
        if args.checkpoint is None:
            parser.error("train requires --checkpoint to preserve its validation split")
        train(args.output, args.checkpoint, args.epochs, args.train_seed)
    elif args.mode == "replay":
        replay_collection(args.output)
    else:
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint")
        evaluate_checkpoint(args.output, args.checkpoint,
                            args.seed_start if args.seed_start is not None else 40000,
                            args.episodes if args.episodes is not None else 100)


if __name__ == "__main__":
    main()
