import numpy as np

from ilgrasp.experts import ExpertConfig, ExpertPhase, ScriptedExpert


def raw_obs(ee, obj=(0.0, 0.0, 0.02), goal=(0.2, 0.1, 0.04)):
    main = np.zeros(19, dtype=np.float32)
    main[0:3] = ee
    main[7:10] = obj
    return {"observation": main, "desired_goal": np.asarray(goal, dtype=np.float32)}


def test_expert_starts_above_object_and_opens_gripper():
    expert = ScriptedExpert(ExpertConfig(action_gain=10.0))
    action = expert.act(raw_obs(ee=(0.1, 0.1, 0.3)))
    assert action.shape == (4,)
    assert np.all(np.abs(action) <= 1.0)
    assert action[3] > 0
    assert action[0] < 0 and action[1] < 0


def test_expert_transitions_from_approach_to_descend():
    cfg = ExpertConfig(position_tolerance=0.02, horizontal_tolerance=0.02)
    expert = ScriptedExpert(cfg)
    expert.act(raw_obs(ee=(0.0, 0.0, 0.12)))
    assert expert.phase is ExpertPhase.DESCEND

