import numpy as np

from ilgrasp.data import DemoDatabase
from ilgrasp.sim import ACTION_DIM, OBS_DIM, Episode


def make_episode(seed: int, success: bool) -> Episode:
    n = 3
    return Episode(
        observations=np.full((n, OBS_DIM), seed, dtype=np.float32),
        actions=np.full((n, ACTION_DIM), 0.25, dtype=np.float32),
        rewards=np.arange(n, dtype=np.float32),
        success=success,
        seed=seed,
        phases=["approach", "grasp", "lift"],
    )


def test_database_round_trip_filters_failures(tmp_path):
    path = tmp_path / "demos.db"
    with DemoDatabase(path) as db:
        failure_id = db.save_episode(make_episode(1, False))
        success_id = db.save_episode(make_episode(2, True))
        assert failure_id != success_id
        assert db.summary()["expert_success_rate"] == 0.5
        data = db.load_training_data(successful_only=True)

    assert len(data) == 3
    assert set(data.episode_ids) == {success_id}
    assert data.phases.tolist() == ["approach", "grasp", "lift"]
    np.testing.assert_allclose(data.observations, 2.0)

