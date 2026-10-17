"""Reusable definitions extracted from the supplied verified student notebook.

The revised runner controls normalization, split, and train-only weighting explicitly.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

from ilgrasp.data import TrainingData
from ilgrasp.experts.verified import VerifiedExpertPhase

FIXED_PHASES = tuple(p.value for p in VerifiedExpertPhase)
FIXED_PHASE_TO_ID = {p: i for i, p in enumerate(FIXED_PHASES)}


def split_episode_ids(
    episode_ids: np.ndarray, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """E10: return train_mask, val_mask, train_ids, val_ids."""
    unique_ids = np.unique(episode_ids)
    if unique_ids.size < 2:
        raise ValueError("Need at least two episodes to form a split.")
    shuffled = np.random.default_rng(seed).permutation(unique_ids)
    n_val = round(validation_fraction * unique_ids.size)
    n_val = min(max(n_val, 1), unique_ids.size - 1)
    val_ids = np.sort(shuffled[:n_val])
    train_ids = np.sort(shuffled[n_val:])
    val_mask = np.isin(episode_ids, val_ids)
    train_mask = ~val_mask
    return (train_mask, val_mask, train_ids, val_ids)


def hybrid_bc_loss(
    motion_prediction: torch.Tensor, gripper_logit: torch.Tensor, expert_actions: torch.Tensor
) -> torch.Tensor:
    """E11: SmoothL1(xyz) + 0.35 * BCE(open/close)."""
    motion_loss = F.smooth_l1_loss(motion_prediction, expert_actions[:, :3])
    gripper_target = (expert_actions[:, 3:4] > 0).to(motion_prediction.dtype)
    gripper_loss = F.binary_cross_entropy_with_logits(gripper_logit, gripper_target)
    return motion_loss + 0.35 * gripper_loss


class HybridPolicy(nn.Module):
    def __init__(self, input_dim: int = 22, std_floor: float = 1e-6) -> None:
        super().__init__()
        self.std_floor = std_floor
        self.input_dim = input_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.motion_head = nn.Sequential(nn.Linear(128, 3), nn.Tanh())
        self.gripper_head = nn.Linear(128, 1)
        self.register_buffer("obs_mean", torch.zeros(input_dim))
        self.register_buffer("obs_std", torch.ones(input_dim))

    def set_normalization(self, observations: np.ndarray) -> None:
        x = torch.as_tensor(observations, dtype=torch.float32)
        self.obs_mean.copy_(x.mean(0))
        self.obs_std.copy_(x.std(0, unbiased=False).clamp_min(self.std_floor))

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk((observations - self.obs_mean) / self.obs_std)
        return (self.motion_head(hidden), self.gripper_head(hidden))


class HybridNumpyPolicy:
    def __init__(self, model: HybridPolicy) -> None:
        self.model = model.eval()

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        features = relative_features(observation) if self.model.input_dim == 33 else observation
        x = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            motion, logit = self.model(x)
        grip = torch.where(logit >= 0, 1.0, -1.0)
        return torch.cat([motion, grip], dim=1).squeeze(0).numpy().astype(np.float32)


@dataclass
class HybridTrainResult:
    model: HybridPolicy
    train_loss: list[float]
    validation_loss: list[float]
    train_ids: np.ndarray
    validation_ids: np.ndarray


def train_hybrid(
    data: TrainingData,
    epochs: int,
    seed: int = 0,
    split_seed: int = 7,
    std_floor: float = 1e-6,
    input_dim: int = 22,
) -> HybridTrainResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_mask, val_mask, train_ids, val_ids = split_episode_ids(data.episode_ids, 0.2, split_seed)
    model = HybridPolicy(input_dim=input_dim, std_floor=std_floor)
    x_features = relative_features(data.observations) if input_dim == 33 else data.observations
    model.set_normalization(x_features[train_mask])
    train_set = TensorDataset(
        torch.as_tensor(x_features[train_mask], dtype=torch.float32),
        torch.as_tensor(data.actions[train_mask], dtype=torch.float32),
    )
    loader = DataLoader(
        train_set,
        batch_size=min(256, len(train_set)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    x_val = torch.as_tensor(x_features[val_mask], dtype=torch.float32)
    y_val = torch.as_tensor(data.actions[val_mask], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_history, val_history = ([], [])
    best_state, best_val = (deepcopy(model.state_dict()), float("inf"))
    for _ in range(epochs):
        model.train()
        epoch_sum = 0.0
        n = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            motion, logit = model(x_batch)
            loss = hybrid_bc_loss(motion, logit, y_batch)
            loss.backward()
            optimizer.step()
            epoch_sum += float(loss.detach()) * len(x_batch)
            n += len(x_batch)
        train_history.append(epoch_sum / n)
        model.eval()
        with torch.inference_mode():
            motion, logit = model(x_val)
            val = float(hybrid_bc_loss(motion, logit, y_val))
        val_history.append(val)
        if val < best_val:
            best_val, best_state = (val, deepcopy(model.state_dict()))
    model.load_state_dict(best_state)
    return HybridTrainResult(model.eval(), train_history, val_history, train_ids, val_ids)


def relative_features(observations: np.ndarray) -> np.ndarray:
    """Append geometry that the controller otherwise has to rediscover from few demos."""
    x = np.asarray(observations, dtype=np.float32)
    ee = x[..., 0:3]
    obj = x[..., 7:10]
    goal = x[..., 19:22]
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
    return np.concatenate([x, extras], axis=-1).astype(np.float32)


def phase_sample_weights(phases: np.ndarray) -> np.ndarray:
    """Prevent long hold phases from completely dominating rare transitions."""
    labels, counts = np.unique(phases, return_counts=True)
    inverse = {label: len(phases) / (len(labels) * count) for label, count in zip(labels, counts)}
    weights = np.asarray([inverse[phase] for phase in phases], dtype=np.float32)
    return np.clip(weights, 0.25, 4.0)


class RecurrentHybridPolicy(nn.Module):
    """Sequence policy with motion, gripper, and auxiliary phase heads."""

    def __init__(self, input_dim: int = 33, hidden_dim: int = 128, std_floor: float = 1e-6) -> None:
        super().__init__()
        self.std_floor = std_floor
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
        self.feature_std.copy_(x.std(0, unbiased=False).clamp_min(self.std_floor))

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


def _episode_sequences(
    training_data: TrainingData, mask: np.ndarray, all_weights: np.ndarray
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    sequences = []
    for episode_id in np.unique(training_data.episode_ids[mask]):
        episode_mask = mask & (training_data.episode_ids == episode_id)
        sequences.append(
            (
                torch.as_tensor(
                    relative_features(training_data.observations[episode_mask]), dtype=torch.float32
                ),
                torch.as_tensor(training_data.actions[episode_mask], dtype=torch.float32),
                torch.as_tensor(
                    [FIXED_PHASE_TO_ID[p] for p in training_data.phases[episode_mask]],
                    dtype=torch.long,
                ),
                torch.as_tensor(all_weights[episode_mask], dtype=torch.float32),
            )
        )
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
    return (features, actions, phases, weights, valid)


def _recurrent_losses(
    model: RecurrentHybridPolicy,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    weighted: bool,
) -> torch.Tensor:
    features, actions, phases, weights, valid = batch
    motion, gripper_logit, phase_logit, _ = model(features)
    action_loss = F.smooth_l1_loss(motion, actions[..., :3], reduction="none").mean(
        -1
    ) + 0.35 * F.binary_cross_entropy_with_logits(
        gripper_logit, (actions[..., 3:4] > 0).to(motion.dtype), reduction="none"
    ).squeeze(-1)
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
    split_seed: int = 7,
    std_floor: float = 1e-6,
    validation_ids: np.ndarray | None = None,
) -> RecurrentTrainResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if validation_ids is None:
        train_mask, val_mask, train_ids, val_ids = split_episode_ids(
            training_data.episode_ids, validation_fraction=0.2, seed=split_seed
        )
    else:
        val_ids = np.asarray(validation_ids)
        val_mask = np.isin(training_data.episode_ids, val_ids)
        train_mask = ~val_mask
        train_ids = np.unique(training_data.episode_ids[train_mask])
        if not train_mask.any() or not val_mask.any():
            raise ValueError("Explicit validation IDs must leave nonempty train and validation sets")
    labels, counts = np.unique(training_data.phases[train_mask], return_counts=True)
    weight_map = {
        label: float(np.clip(train_mask.sum() / (len(labels) * count), 0.25, 4.0))
        for label, count in zip(labels, counts)
    }
    weights = np.array([weight_map.get(p, 1.0) for p in training_data.phases], dtype=np.float32)
    train_episodes = _episode_sequences(training_data, train_mask, weights)
    val_episodes = _episode_sequences(training_data, val_mask, weights)
    model = RecurrentHybridPolicy(std_floor=std_floor)
    model.set_normalization(relative_features(training_data.observations[train_mask]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    generator = np.random.default_rng(seed)
    best_state = deepcopy(model.state_dict())
    best_validation = float("inf")
    train_history, validation_history = ([], [])
    for _ in range(epochs):
        model.train()
        batch_losses = []
        order = generator.permutation(len(train_episodes))
        for start in range(0, len(order), 8):
            batch = _pad_episode_batch(
                [train_episodes[index] for index in order[start : start + 8]]
            )
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
