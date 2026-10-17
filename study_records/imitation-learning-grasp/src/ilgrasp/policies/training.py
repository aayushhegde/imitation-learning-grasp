"""Behavior cloning with episode-level validation and optional phase balancing."""

import random
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ilgrasp.data import TrainingData

from .mlp import GraspPolicy


@dataclass
class TrainResult:
    model: GraspPolicy
    train_loss: list[float]
    validation_loss: list[float]
    train_episode_ids: np.ndarray
    validation_episode_ids: np.ndarray


def _phase_weights(phases: np.ndarray) -> np.ndarray:
    """Give rare controller phases comparable influence to long common phases."""

    if len(phases) == 0 or np.all(phases == ""):
        return np.ones(len(phases), dtype=np.float32)
    labels, counts = np.unique(phases, return_counts=True)
    inverse = {label: len(phases) / (len(labels) * count) for label, count in zip(labels, counts)}
    weights = np.asarray([inverse[label] for label in phases], dtype=np.float32)
    return np.clip(weights, 0.25, 4.0)


def train_behavior_cloning(
    data: TrainingData,
    hidden_sizes: Sequence[int] = (128, 128),
    epochs: int = 80,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    seed: int = 0,
    balance_phases: bool = True,
    device: str = "cpu",
) -> TrainResult:
    """Fit an MLP to expert actions without leaking episodes across the split."""

    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    unique_ids = np.unique(data.episode_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)
    n_validation = 0
    if len(shuffled) >= 2 and validation_fraction > 0:
        n_validation = max(1, round(len(shuffled) * validation_fraction))
        n_validation = min(n_validation, len(shuffled) - 1)
    validation_ids = np.sort(shuffled[:n_validation])
    train_ids = np.sort(shuffled[n_validation:])
    train_mask = np.isin(data.episode_ids, train_ids)
    validation_mask = np.isin(data.episode_ids, validation_ids)

    model = GraspPolicy(hidden_sizes=hidden_sizes).to(device)
    model.set_normalization(data.observations[train_mask])

    weights = _phase_weights(data.phases[train_mask]) if balance_phases else np.ones(train_mask.sum())
    x_train = torch.as_tensor(data.observations[train_mask], dtype=torch.float32)
    y_train = torch.as_tensor(data.actions[train_mask], dtype=torch.float32)
    w_train = torch.as_tensor(weights, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train, w_train),
        batch_size=min(batch_size, len(x_train)),
        shuffle=True,
        generator=generator,
    )

    x_validation = torch.as_tensor(data.observations[validation_mask], dtype=torch.float32, device=device)
    y_validation = torch.as_tensor(data.actions[validation_mask], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_history: list[float] = []
    validation_history: list[float] = []
    best_state = deepcopy(model.state_dict())
    best_validation = float("inf")

    for _ in range(epochs):
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0
        for x_batch, y_batch, w_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            w_batch = w_batch.to(device)
            optimizer.zero_grad()
            per_sample = (model(x_batch) - y_batch).square().mean(dim=1)
            loss = (per_sample * w_batch).sum() / w_batch.sum()
            loss.backward()
            optimizer.step()
            loss_sum += float((per_sample.detach() * w_batch).sum())
            weight_sum += float(w_batch.sum())
        train_history.append(loss_sum / weight_sum)

        model.eval()
        if n_validation:
            with torch.inference_mode():
                validation_loss = float((model(x_validation) - y_validation).square().mean())
            validation_history.append(validation_loss)
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_state = deepcopy(model.state_dict())
        else:
            validation_history.append(float("nan"))
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model, train_history, validation_history, train_ids, validation_ids)
