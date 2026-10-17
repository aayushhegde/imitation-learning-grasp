import numpy as np
import torch

from ilgrasp.data import TrainingData
from ilgrasp.policies import GraspPolicy, NumpyPolicy, train_behavior_cloning
from ilgrasp.sim import ACTION_DIM, OBS_DIM


def test_behavior_cloning_trains_saves_and_keeps_episode_split_disjoint(tmp_path):
    rng = np.random.default_rng(4)
    observations = rng.normal(size=(120, OBS_DIM)).astype(np.float32)
    actions = np.tanh(observations[:, :ACTION_DIM]).astype(np.float32)
    episode_ids = np.repeat(np.arange(6), 20)
    phases = np.resize(np.array(["approach", "grasp", "transport"]), len(observations))
    data = TrainingData(observations, actions, episode_ids, phases)

    result = train_behavior_cloning(
        data,
        hidden_sizes=(32, 32),
        epochs=20,
        batch_size=32,
        learning_rate=3e-3,
        seed=3,
    )

    assert result.train_loss[-1] < result.train_loss[0]
    assert set(result.train_episode_ids).isdisjoint(result.validation_episode_ids)
    assert set(result.train_episode_ids) | set(result.validation_episode_ids) == set(range(6))

    checkpoint = tmp_path / "policy.pt"
    result.model.save(checkpoint)
    loaded = GraspPolicy.load(checkpoint)
    action = NumpyPolicy(loaded)(observations[0])
    assert action.shape == (ACTION_DIM,)
    assert np.all(np.abs(action) <= 1.0)

    with torch.inference_mode():
        before = result.model(torch.as_tensor(observations[:2]))
        after = loaded(torch.as_tensor(observations[:2]))
    torch.testing.assert_close(before, after)
