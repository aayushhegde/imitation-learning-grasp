"""Regression checks for the revised BC data and state contracts."""

import numpy as np
import torch

from ilgrasp.data import TrainingData
from ilgrasp.policies.revision import (
    RecurrentHybridPolicy,
    RecurrentNumpyPolicy,
    split_episode_ids,
    train_recurrent_hybrid,
)


def test_validation_phase_labels_do_not_change_training_updates():
    torch.set_num_threads(1)
    rng = np.random.default_rng(20)
    obs = rng.normal(size=(24, 22)).astype(np.float32)
    actions = np.clip(rng.normal(size=(24, 4)), -1, 1).astype(np.float32)
    ids = np.repeat(np.arange(6), 4)
    phases = np.full(24, "approach", dtype="U20")
    _, val, _, _ = split_episode_ids(ids, 0.2, 7)
    original = train_recurrent_hybrid(
        TrainingData(obs, actions, ids, phases), epochs=1, seed=2, std_floor=0.05
    )
    changed_phases = phases.copy()
    changed_phases[val] = "verify"
    changed = train_recurrent_hybrid(
        TrainingData(obs, actions, ids, changed_phases), epochs=1, seed=2, std_floor=0.05
    )
    for key, tensor in original.model.state_dict().items():
        torch.testing.assert_close(tensor, changed.model.state_dict()[key], rtol=0, atol=0)


def test_scale_floor_and_recurrent_episode_reset():
    torch.manual_seed(9)
    model = RecurrentHybridPolicy(std_floor=0.05)
    model.set_normalization(np.ones((8, 33), dtype=np.float32))
    torch.testing.assert_close(model.feature_std, torch.full((33,), 0.05))
    policy = RecurrentNumpyPolicy(model)
    obs = np.zeros(22, dtype=np.float32)
    first = policy(obs)
    policy(np.ones(22, dtype=np.float32))
    policy.reset()
    np.testing.assert_array_equal(first, policy(obs))
