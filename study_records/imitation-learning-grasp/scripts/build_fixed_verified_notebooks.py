#!/usr/bin/env python3
"""Build corrected student and instructor editions of the verified BC notebook.

The source notebooks are intentionally left untouched.  The fixed notebooks:

* align the student's expert implementation with the verified reference;
* collect a requested number of successful demonstrations rather than attempts;
* reset stateful policies during evaluation;
* replace hard-coded failure thresholds with the environment contract; and
* add a recurrent, relative-geometry BC policy with an auxiliary phase loss.
"""

from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path
from typing import Any

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
STUDENT_SOURCE = NOTEBOOK_DIR / "04_verified_expert_pipeline_student_en (2).ipynb"
INSTRUCTOR_SOURCE = NOTEBOOK_DIR / "04_verified_expert_pipeline_solutions_en.ipynb"
STUDENT_TARGET = NOTEBOOK_DIR / "04_verified_expert_pipeline_student_en_fixed.ipynb"
INSTRUCTOR_TARGET = NOTEBOOK_DIR / "04_verified_expert_pipeline_solutions_en_fixed.ipynb"


def source_lines(text: str) -> list[str]:
    """Return notebook-compatible source lines with stable trailing newlines."""
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def cell_by_id(notebook: dict[str, Any], cell_id: str) -> dict[str, Any]:
    matches = [cell for cell in notebook["cells"] if cell.get("id") == cell_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one cell with id={cell_id!r}, found {len(matches)}")
    return matches[0]


def markdown_cell(cell_id: str, text: str, tags: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {"tags": list(tags)} if tags else {},
        "source": source_lines(text),
    }


def code_cell(cell_id: str, text: str, tags: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {"tags": list(tags)} if tags else {},
        "outputs": [],
        "source": source_lines(text),
    }


STUDENT_EXPERT_FIX_IDS = (
    "e01-robot-state",
    "e03-pick-decisions",
    "e04-transport-decisions",
    "e05-finish-decisions",
    "e06-transition-guards",
    "e07-expert-class",
    "e08-collector",
)


COLLECT_SUCCESSFUL_DEMOS = r'''
store = VerifiedDemoStore(tutorial_dir / "verified_demos.sqlite")
data_env = VerifiedPlacementEnv()

# TARGET_DEMOS means successful, collision-free episodes—not merely attempts.
successful_demos = 0
collection_seed = 0
max_collection_attempts = 5 * TARGET_DEMOS
while successful_demos < TARGET_DEMOS and collection_seed < max_collection_attempts:
    trace = collect_trace(data_env, BarrierPickPlaceExpert(), collection_seed)
    store.save(trace)  # retain failures for diagnosis
    if trace.success and trace.infos[-1]["collision_free"]:
        successful_demos += 1
    collection_seed += 1
data_env.close()

if successful_demos < TARGET_DEMOS:
    raise RuntimeError(
        f"Collected only {successful_demos}/{TARGET_DEMOS} verified demonstrations "
        f"after {collection_seed} attempts."
    )

data = load_verified_training_data(store.connection, TARGET_DEMOS)
episode_rows = store.connection.execute("""
    SELECT COUNT(*), SUM(strict_success), SUM(native_success_seen), SUM(collision_free)
    FROM episodes
""").fetchone()
print({
    "attempts / strict / native-hit / collision-free": episode_rows,
    "selected_successful_demos": successful_demos,
    "observations": data.observations.shape,
    "actions": data.actions.shape,
})
'''


STRICT_EVALUATOR = r'''
@dataclass
class StrictEvaluation:
    strict_successes: np.ndarray
    native_hits: np.ndarray
    final_infos: list[dict[str, Any]]
    seeds: np.ndarray


def evaluate_strict(
    policy: Any, n_episodes: int, seed_start: int = 10_000
) -> StrictEvaluation:
    """Run a policy on fresh seeds and reset any recurrent state per episode."""
    env = VerifiedPlacementEnv()
    strict_successes, native_hits, final_infos, seeds = [], [], [], []

    try:
        for episode in range(n_episodes):
            seed = seed_start + episode
            observation, _ = env.reset(seed=seed)
            if hasattr(policy, "reset"):
                policy.reset()
            info: dict[str, Any] = {}

            for _ in range(env.config.max_episode_steps):
                action = np.asarray(policy(observation), dtype=np.float32)
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

            if not info:
                raise RuntimeError(f"Episode {seed} produced no steps.")
            strict_successes.append(bool(info["strict_success"]))
            native_hits.append(bool(info["native_success_seen"]))
            final_infos.append(dict(info))
            seeds.append(seed)
    finally:
        env.close()

    return StrictEvaluation(
        strict_successes=np.asarray(strict_successes, dtype=bool),
        native_hits=np.asarray(native_hits, dtype=bool),
        final_infos=final_infos,
        seeds=np.asarray(seeds, dtype=np.int64),
    )
'''


FAILURE_CLASSIFIER = r'''
def classify_failure(info: dict[str, Any]) -> str:
    """Classify the earliest missing lifecycle milestone from the environment contract."""
    if info["strict_success"]:
        return "success"
    if info["barrier_violation"]:
        return "barrier_collision"
    if not info["ever_lifted"]:
        return "grasp_failed"
    if not info["ever_crossed"]:
        return "dropped_before_barrier"
    if not info["ever_released"]:
        return "not_released"
    if not info["retreated"]:
        return "no_retreat"
    if info["stable_steps"] < info["required_stable_steps"]:
        return "unstable_placement"
    return "inconsistent_certificate"
'''


FIXED_INTRO = r'''
## 9. Fixed BC policy — relative geometry plus recurrent phase memory

The feed-forward baseline above is intentionally diagnostic: it maps only the current 22-value
observation to one action. The expert, however, also uses a phase, a phase timer, frozen grasp and
release targets, and retry memory. At `CLOSE → TEST_GRASP` and `OPEN → RETREAT_UP`, nearly identical
observations require different actions, so the original stateless mapping is not single-valued.

The repaired learner keeps the same expert data and strict evaluator, but makes three changes:

1. append translation-friendly relative geometry (`object−EE`, `goal−EE`, `goal−object`, distances);
2. train whole episode sequences with a GRU instead of shuffling isolated timesteps;
3. use the stored phase only as an auxiliary training target—the deployed policy receives no
   privileged phase input and carries only its recurrent hidden state.

The original MLP remains above as a controlled legacy baseline. Both policies are evaluated below
on the same new seed suite beginning at 20,000.
'''


RELATIVE_FEATURES = r'''
from torch.nn.utils.rnn import pad_sequence


FIXED_PHASES = tuple(phase.value for phase in ExpertPhase)
FIXED_PHASE_TO_ID = {phase: index for index, phase in enumerate(FIXED_PHASES)}


def relative_features(observations: np.ndarray) -> np.ndarray:
    """Append geometry that the controller otherwise has to rediscover from few demos."""
    x = np.asarray(observations, dtype=np.float32)
    ee = x[..., 0:3]
    obj = x[..., 7:10]
    goal = x[..., 19:22]
    extras = np.concatenate([
        obj - ee,
        goal - ee,
        goal - obj,
        np.linalg.norm(obj - ee, axis=-1, keepdims=True),
        np.linalg.norm(goal - obj, axis=-1, keepdims=True),
    ], axis=-1)
    return np.concatenate([x, extras], axis=-1).astype(np.float32)


def phase_sample_weights(phases: np.ndarray) -> np.ndarray:
    """Prevent long hold phases from completely dominating rare transitions."""
    labels, counts = np.unique(phases, return_counts=True)
    inverse = {
        label: len(phases) / (len(labels) * count)
        for label, count in zip(labels, counts)
    }
    weights = np.asarray([inverse[phase] for phase in phases], dtype=np.float32)
    return np.clip(weights, 0.25, 4.0)
'''


RECURRENT_MODEL = r'''
class RecurrentHybridPolicy(nn.Module):
    """Sequence policy with motion, gripper, and auxiliary phase heads."""

    def __init__(self, input_dim: int = 33, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.motion_head = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Tanh())
        self.gripper_head = nn.Linear(hidden_dim, 1)
        self.phase_head = nn.Linear(hidden_dim, len(FIXED_PHASES))
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))

    def set_normalization(self, features: np.ndarray) -> None:
        x = torch.as_tensor(features, dtype=torch.float32)
        self.feature_mean.copy_(x.mean(0))
        self.feature_std.copy_(x.std(0, unbiased=False).clamp_min(1e-6))

    def forward(
        self, features: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder((features - self.feature_mean) / self.feature_std)
        recurrent, next_hidden = self.gru(encoded, hidden)
        return (
            self.motion_head(recurrent),
            self.gripper_head(recurrent),
            self.phase_head(recurrent),
            next_hidden,
        )


class RecurrentNumpyPolicy:
    """Stateful NumPy adapter; evaluate_strict calls reset at every episode."""

    def __init__(self, model: RecurrentHybridPolicy) -> None:
        self.model = model.eval()
        self.reset()

    def reset(self) -> None:
        self.hidden: torch.Tensor | None = None

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        features = relative_features(np.asarray(observation)[None])[:, None, :]
        x = torch.as_tensor(features, dtype=torch.float32)
        with torch.inference_mode():
            motion, gripper_logit, _, self.hidden = self.model(x, self.hidden)
        gripper = torch.where(gripper_logit >= 0, 1.0, -1.0)
        action = torch.cat([motion, gripper], dim=-1)[0, 0]
        return action.cpu().numpy().astype(np.float32)


@dataclass
class RecurrentTrainResult:
    model: RecurrentHybridPolicy
    train_loss: list[float]
    validation_loss: list[float]
    train_ids: np.ndarray
    validation_ids: np.ndarray
'''


RECURRENT_TRAINING = r'''
def _episode_sequences(
    training_data: TrainingData,
    mask: np.ndarray,
    all_weights: np.ndarray,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    sequences = []
    for episode_id in np.unique(training_data.episode_ids[mask]):
        episode_mask = mask & (training_data.episode_ids == episode_id)
        sequences.append((
            torch.as_tensor(
                relative_features(training_data.observations[episode_mask]),
                dtype=torch.float32,
            ),
            torch.as_tensor(training_data.actions[episode_mask], dtype=torch.float32),
            torch.as_tensor(
                [FIXED_PHASE_TO_ID[p] for p in training_data.phases[episode_mask]],
                dtype=torch.long,
            ),
            torch.as_tensor(all_weights[episode_mask], dtype=torch.float32),
        ))
    return sequences


def _pad_episode_batch(
    episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = pad_sequence([episode[0] for episode in episodes], batch_first=True)
    actions = pad_sequence([episode[1] for episode in episodes], batch_first=True)
    phases = pad_sequence([episode[2] for episode in episodes], batch_first=True)
    weights = pad_sequence([episode[3] for episode in episodes], batch_first=True)
    lengths = torch.as_tensor([len(episode[0]) for episode in episodes])
    valid = torch.arange(features.shape[1])[None, :] < lengths[:, None]
    return features, actions, phases, weights, valid


def _recurrent_losses(
    model: RecurrentHybridPolicy,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    weighted: bool,
) -> torch.Tensor:
    features, actions, phases, weights, valid = batch
    motion, gripper_logit, phase_logit, _ = model(features)
    action_loss = (
        F.smooth_l1_loss(motion, actions[..., :3], reduction="none").mean(-1)
        + 0.35 * F.binary_cross_entropy_with_logits(
            gripper_logit,
            (actions[..., 3:4] > 0).to(motion.dtype),
            reduction="none",
        ).squeeze(-1)
    )
    if weighted:
        action_term = (action_loss[valid] * weights[valid]).sum() / weights[valid].sum()
    else:
        action_term = action_loss[valid].mean()
    phase_term = F.cross_entropy(phase_logit[valid], phases[valid])
    return action_term + 0.15 * phase_term


def train_recurrent_hybrid(
    training_data: TrainingData,
    epochs: int = 80,
    seed: int = 7,
) -> RecurrentTrainResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_mask, val_mask, train_ids, val_ids = split_episode_ids(
        training_data.episode_ids, validation_fraction=0.2, seed=seed
    )
    weights = phase_sample_weights(training_data.phases)
    train_episodes = _episode_sequences(training_data, train_mask, weights)
    val_episodes = _episode_sequences(training_data, val_mask, weights)

    model = RecurrentHybridPolicy()
    model.set_normalization(relative_features(training_data.observations[train_mask]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = np.random.default_rng(seed)
    best_state = deepcopy(model.state_dict())
    best_validation = float("inf")
    train_history, validation_history = [], []

    for _ in range(epochs):
        model.train()
        batch_losses = []
        order = generator.permutation(len(train_episodes))
        for start in range(0, len(order), 8):
            batch = _pad_episode_batch([train_episodes[index] for index in order[start:start + 8]])
            optimizer.zero_grad()
            loss = _recurrent_losses(model, batch, weighted=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        train_history.append(float(np.mean(batch_losses)))

        model.eval()
        with torch.inference_mode():
            validation = float(
                _recurrent_losses(model, _pad_episode_batch(val_episodes), weighted=False)
            )
        validation_history.append(validation)
        if validation < best_validation:
            best_validation = validation
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return RecurrentTrainResult(
        model=model,
        train_loss=train_history,
        validation_loss=validation_history,
        train_ids=train_ids,
        validation_ids=val_ids,
    )


FIXED_TRAIN_EPOCHS = 80
fixed_trained = train_recurrent_hybrid(data, epochs=FIXED_TRAIN_EPOCHS, seed=7)
fixed_policy = RecurrentNumpyPolicy(fixed_trained.model)

assert set(fixed_trained.train_ids).isdisjoint(fixed_trained.validation_ids)
plt.figure(figsize=(7, 3.5))
plt.plot(fixed_trained.train_loss, label="train")
plt.plot(fixed_trained.validation_loss, label="validation")
plt.xlabel("epoch")
plt.ylabel("recurrent hybrid loss")
plt.title("Fixed recurrent behavior cloning")
plt.legend()
plt.tight_layout()
plt.show()
'''


FIXED_EVALUATION = r'''
FIXED_EVAL_EPISODES = 24
FIXED_EVAL_SEED_START = 20_000

legacy_fixed_seeds = evaluate_strict(
    learned_policy, FIXED_EVAL_EPISODES, seed_start=FIXED_EVAL_SEED_START
)
fixed_evaluation = evaluate_strict(
    fixed_policy, FIXED_EVAL_EPISODES, seed_start=FIXED_EVAL_SEED_START
)


class ExpertFlatControl:
    def __init__(self) -> None:
        self.expert = BarrierPickPlaceExpert()

    def reset(self) -> None:
        self.expert.reset()

    def __call__(self, flat_observation: np.ndarray) -> np.ndarray:
        flat = np.asarray(flat_observation, dtype=np.float32)
        return self.expert.act({
            "observation": flat[:19],
            "desired_goal": flat[19:22],
        })


fixed_expert_control = evaluate_strict(
    ExpertFlatControl(), FIXED_EVAL_EPISODES, seed_start=FIXED_EVAL_SEED_START
)

comparison = {
    "seed_range": (
        int(FIXED_EVAL_SEED_START),
        int(FIXED_EVAL_SEED_START + FIXED_EVAL_EPISODES - 1),
    ),
    "legacy_stateless_strict": (
        f"{int(legacy_fixed_seeds.strict_successes.sum())}/{FIXED_EVAL_EPISODES}"
    ),
    "fixed_recurrent_strict": (
        f"{int(fixed_evaluation.strict_successes.sum())}/{FIXED_EVAL_EPISODES}"
    ),
    "fixed_recurrent_native": (
        f"{int(fixed_evaluation.native_hits.sum())}/{FIXED_EVAL_EPISODES}"
    ),
    "expert_strict": (
        f"{int(fixed_expert_control.strict_successes.sum())}/{FIXED_EVAL_EPISODES}"
    ),
    "fixed_failures": dict(Counter(
        classify_failure(info) for info in fixed_evaluation.final_infos
    )),
}
print(comparison)

labels = ["legacy stateless", "fixed recurrent", "expert control"]
strict_rates = [
    legacy_fixed_seeds.strict_successes.mean(),
    fixed_evaluation.strict_successes.mean(),
    fixed_expert_control.strict_successes.mean(),
]
plt.figure(figsize=(7.5, 3.8))
plt.bar(labels, strict_rates, color=["#E45756", "#35C48D", "#4C78A8"])
plt.ylim(0, 1)
plt.ylabel("strict success rate")
plt.title("Same fresh seeds: legacy versus fixed policy")
plt.tight_layout()
plt.show()

assert fixed_expert_control.strict_successes.all(), "The repaired expert must pass its control suite."
assert fixed_evaluation.strict_successes.any(), "The repaired learner should escape the zero-success floor."
'''


FIXED_CELLS = (
    markdown_cell("fixed-policy-intro", FIXED_INTRO, ("fixed-policy",)),
    code_cell("fixed-relative-features", RELATIVE_FEATURES, ("fixed-policy",)),
    code_cell("fixed-recurrent-model", RECURRENT_MODEL, ("fixed-policy",)),
    code_cell("fixed-recurrent-training", RECURRENT_TRAINING, ("fixed-policy", "learning")),
    code_cell("fixed-recurrent-evaluation", FIXED_EVALUATION, ("fixed-policy", "evaluation")),
)


def clear_execution(notebook: dict[str, Any]) -> None:
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def update_heading(notebook: dict[str, Any], cell_id: str, old: str, new: str) -> None:
    cell = cell_by_id(notebook, cell_id)
    source = "".join(cell["source"])
    if old not in source:
        raise ValueError(f"Could not find heading {old!r} in {cell_id}")
    cell["source"] = source_lines(source.replace(old, new, 1))


def build_fixed_student(reference: dict[str, Any]) -> dict[str, Any]:
    notebook = json.loads(STUDENT_SOURCE.read_text(encoding="utf-8"))
    # Preserve the student's stronger E02/E09--E12 validation and all added
    # diagnostics.  Only replace the incorrect expert core.
    for cell_id in STUDENT_EXPERT_FIX_IDS:
        cell_by_id(notebook, cell_id)["source"] = deepcopy(cell_by_id(reference, cell_id)["source"])

    # Restore the reference expert benchmark that was absent from the submitted copy.
    if not any(cell.get("id") == "expert-benchmark" for cell in notebook["cells"]):
        collector_index = next(
            index for index, cell in enumerate(notebook["cells"])
            if cell.get("id") == "e08-collector"
        )
        notebook["cells"].insert(
            collector_index + 1,
            deepcopy(cell_by_id(reference, "expert-benchmark")),
        )

    return notebook


def apply_common_fixes(notebook: dict[str, Any], edition: str) -> dict[str, Any]:
    title = cell_by_id(notebook, "title")
    title_text = "".join(title["source"])
    first, *rest = title_text.splitlines()
    title["source"] = source_lines(
        first + f" — Fixed {edition} Edition\n" + "\n".join(rest)
    )

    setup = cell_by_id(notebook, "setup")
    setup_text = "".join(setup["source"])
    setup_text = setup_text.replace(
        "TARGET_DEMOS = 24 if QUICK_MODE else 100",
        "TARGET_DEMOS = 40 if QUICK_MODE else 200",
    )
    setup["source"] = source_lines(setup_text)

    cell_by_id(notebook, "collect-database")["source"] = source_lines(COLLECT_SUCCESSFUL_DEMOS)
    cell_by_id(notebook, "e12-strict-evaluator")["source"] = source_lines(STRICT_EVALUATOR)
    cell_by_id(notebook, "e13-failure-classifier")["source"] = source_lines(FAILURE_CLASSIFIER)

    dagger_index = next(
        index for index, cell in enumerate(notebook["cells"])
        if cell.get("id") == "dagger"
    )
    notebook["cells"][dagger_index:dagger_index] = deepcopy(list(FIXED_CELLS))
    update_heading(notebook, "dagger", "## 9.", "## 10.")
    update_heading(notebook, "experiments", "## 10.", "## 11.")
    update_heading(notebook, "file-map", "## 11.", "## 12.")

    notebook.setdefault("metadata", {})["fixed_pipeline"] = {
        "edition": edition.lower(),
        "student_source": STUDENT_SOURCE.name,
        "instructor_source": INSTRUCTOR_SOURCE.name,
        "policy": "relative-geometry GRU with auxiliary phase loss",
        "evaluation_seed_start": 20_000,
    }
    clear_execution(notebook)
    return notebook


def validate(notebook: dict[str, Any]) -> None:
    ids = [cell.get("id") for cell in notebook["cells"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Notebook contains duplicate cell ids")
    required = {
        "e01-robot-state",
        "e06-transition-guards",
        "collect-database",
        "e12-strict-evaluator",
        "fixed-relative-features",
        "fixed-recurrent-model",
        "fixed-recurrent-training",
        "fixed-recurrent-evaluation",
    }
    missing = required - set(ids)
    if missing:
        raise ValueError(f"Fixed notebook is missing cells: {sorted(missing)}")

    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for expected in (
        "obj_velocity=main[13:16].copy()",
        "obj_angular_velocity=main[16:19].copy()",
        "memory.grasp_target = state.ee.copy()",
        "np.linalg.norm(state.obj - state.goal)",
        "class RecurrentHybridPolicy",
        "if hasattr(policy, \"reset\")",
    ):
        if expected not in source:
            raise ValueError(f"Expected fixed source fragment not found: {expected}")

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
    nbformat.validate(nbformat.from_dict(notebook))


def load_reference_from_builder() -> dict[str, Any]:
    """Load the canonical teacher source without running its overwrite-prone main()."""
    builder_path = ROOT / "scripts" / "build_verified_expert_notebooks.py"
    namespace = runpy.run_path(str(builder_path))
    reference = namespace["build"]("en", True)

    # Refuse to build from silently divergent teacher sources.
    saved = json.loads(INSTRUCTOR_SOURCE.read_text(encoding="utf-8"))
    saved_sources = {
        cell.get("id"): "".join(cell.get("source", [])) for cell in saved["cells"]
    }
    built_sources = {
        cell.get("id"): "".join(cell.get("source", [])) for cell in reference["cells"]
    }
    if saved_sources != built_sources:
        raise ValueError(
            "The saved instructor notebook no longer matches the canonical notebook builder."
        )
    return reference


def main() -> None:
    reference = load_reference_from_builder()
    student = apply_common_fixes(build_fixed_student(reference), "Student")
    instructor = apply_common_fixes(deepcopy(reference), "Instructor")

    for path, notebook in ((STUDENT_TARGET, student), (INSTRUCTOR_TARGET, instructor)):
        validate(notebook)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        print(f"wrote {path} ({len(notebook['cells'])} cells)")


if __name__ == "__main__":
    main()
