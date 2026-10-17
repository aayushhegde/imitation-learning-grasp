import numpy as np
import pytest

from ilgrasp.sim.observation import OBS_DIM, flatten_obs


def test_flatten_obs_has_fixed_order_and_dtype():
    main = np.arange(19, dtype=np.float64)
    goal = np.array([101.0, 102.0, 103.0])
    result = flatten_obs({"observation": main, "desired_goal": goal, "achieved_goal": main[7:10]})
    assert result.shape == (OBS_DIM,)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[:19], main)
    np.testing.assert_array_equal(result[-3:], goal)


def test_flatten_obs_rejects_contract_change():
    with pytest.raises(ValueError, match="contract may have changed"):
        flatten_obs({"observation": np.zeros(18), "desired_goal": np.zeros(3)})

