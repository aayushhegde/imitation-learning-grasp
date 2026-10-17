# Recovery oracle implementation and evidence guide

Prepared on 2026-10-16 from the retained local files. This guide documents the implementation and distinguishes original experiment records from a new instrumented diagnostic replay. All code paths below are relative to `imitation-learning-grasp/` in this package. Numerical position thresholds are in metres.

## Source and historical identity

The recovery controller is `src/ilgrasp/experts/recovery.py`. Its SHA-256 is `2bedbae95ae6013246b0ccbe42b5fcfa8d32cebf0438d48214d02b2f4dedeee0`, matching both the original oracle-validation manifest and the aggregation manifest. The package includes the complete Python source tree, experiment scripts, tests, dependency declarations, original retained study artifacts, and manifests. It is a snapshot of local files, not an assertion that the entire study was committed at the repository HEAD.

Historical manifests contain an older absolute directory prefix `/Users/yubocai/Desktop/MIT/LaunchLab/Auyush/`. The project was moved to `/Users/yubocai/Desktop/Auyush/`. Those original manifest bytes are preserved. Resolve their remaining relative paths inside the package. Some provenance sidecars were added by a documented retrospective migration; their own migration fields and retained `provenance_migration/runner_before_fingerprinting.py` describe that history. The package does not present them as contemporaneous run logs.

## 1. Phase reconciliation rules

`recovery.py:141-189` separates action queries from updates. `query()` does not mutate the phase or timers. Exactly one `observe_transition()` follows each actual environment action. It reads `env.last_action`, which the environment records after action clipping (`src/ilgrasp/sim/verified_task.py:375-386`). Consecutive-close and consecutive-open counts advance only for negative and positive executed gripper commands, respectively; another sign resets the relevant counter. Reset, missing-update, and duplicate-update checks raise errors.

Reconciliation uses the post-action state and runs the following branches in priority order:

1. If the cube has ever crossed, its Euclidean goal error is less than 0.035, and gripper width exceeds 0.055, enter OPEN unless already finishing. Save the current measured end-effector position as the release target only when making that transition into OPEN.
2. If the cube is held, cube-centre z exceeds 0.05, and the current phase is APPROACH, DESCEND, or CLOSE, update the measured grasp offset `EE - cube` and enter LIFT_CLEAR.
3. If the cube is unheld, cube-centre z is below 0.04, the cube is outside the goal ball, and the controller is transporting or finishing, clear the grasp and release targets and restart APPROACH.
4. If transporting with a held cube, EE z is below `safe_z - 0.04 = 0.13`, the cube is outside the goal ball, and the phase is not LOWER, return to LIFT_CLEAR before horizontal transport.

Transporting phases are LIFT_CLEAR, PRE_CROSS, CROSS_BARRIER, ABOVE_GOAL, and LOWER. Finishing phases are OPEN, RETREAT_UP, RETREAT_HOME, and VERIFY.

Ordinary transitions then run (`recovery.py:191-238`). They use the stored previous state for many geometric arrival guards so that the update aligns with the expert's pre-action guard convention. Position tolerance is 0.018. CLOSE advances after 10 consecutive executed closing commands; TEST_GRASP requires a currently held cube and previous cube z > 0.05, or retries after 18 phase steps. Once `abs(previous_EE_z - 0.17) < 0.018`, LIFT_CLEAR enters ABOVE_GOAL if current cube y > 0.075, otherwise PRE_CROSS. LOWER opens after previous cube-goal error < 0.035 and freezes the current measured EE position; horizontal cube-goal error > 0.06 instead returns to LIFT_CLEAR. OPEN advances after 12 consecutive opening commands. Home arrival tolerance is 0.025.

## 2. Contact condition defining a held cube

`recovery.py:64-71` queries PyBullet `getContactPoints(bodyA=panda, bodyB=object)`. It keeps Panda link indices `contact[3]` only when `contact[8] < 0.002`, then requires both finger links 9 and 10:

```text
held = {9, 10} is a subset of
       {linkA(c) for Panda-cube contacts c with contactDistance(c) < 0.002}
```

This is a bilateral contact-distance criterion. There is no normal-force threshold or gripper-width threshold in `_held()`. The 2 mm tolerance includes reported contact points with small positive separation; it should not be described as a positive gripping-force test. These privileged contacts are used by the teacher; the learned policy does not receive them at evaluation.

## 3. Recovery targets

Defaults: `src/ilgrasp/experts/verified.py:30-46`. Target selection: `recovery.py:81-110`.

| Phase | End-effector target | Gripper command |
|---|---|---|
| APPROACH | cube position + (0, 0, 0.12) | +1, open |
| APPROACH low-hand recovery | If horizontal EE-cube distance > 0.03 and EE z < 0.10: retain EE x/y and lift to z = 0.17 first | +1 |
| DESCEND | cube position + (0, 0, 0.015) | +1 |
| CLOSE | frozen measured grasp target | -1, close |
| TEST_GRASP | frozen grasp target + (0, 0, 0.045) | -1 |
| LIFT_CLEAR | (current EE x, current EE y, 0.17) | -1 |
| PRE_CROSS | (goal x, -0.075, 0.17) | -1 |
| CROSS_BARRIER | (goal x, +0.075, 0.17) | -1 |
| ABOVE_GOAL | (goal x, goal y, 0.17) | -1 |
| LOWER | goal position + measured grasp offset | -1 |
| OPEN | frozen measured release target | +1 |
| RETREAT_UP | (current EE x, current EE y, 0.17) | +1 |
| RETREAT_HOME / VERIFY | (0.04, 0.0, 0.20) | +1 |

The translation command is componentwise `clip(20 * (target - EE), -1, +1)` (`recovery.py:137-139`). These are bounded normalized action components, not raw displacement commands in metres.

## 4. Exact abstention thresholds

`recovery.py:120-136` returns an invalid label with `action=None` on the first matching condition:

| Reason | Implemented condition |
|---|---|
| irreversible_barrier_violation | task's latched barrier-violation flag is true |
| nonfinite_state | any EE or cube position coordinate is nonfinite |
| object_outside_recovery_workspace | abs(cube x) > 0.25 OR abs(cube y) > 0.30 OR cube z < 0 |
| ungrasped_object_near_barrier | cube unheld AND cube z < 0.055 AND abs(cube y) < 0.065 |
| airborne_object_without_verified_grasp | cube unheld AND cube z > 0.065 AND phase is not OPEN, RETREAT_UP, RETREAT_HOME, or VERIFY |

The barrier proximity test is a coordinate strip around the barrier centreline y = 0, not a nearest-surface distance computation. There is no universal upper-z cutoff and no explicit EE workspace bound in this abstention function. The airborne condition has the stated finishing-phase exceptions.

The task separately latches a barrier violation on a barrier contact with `contact[8] < -0.0001` (`verified_task.py:200-227`). Barrier half extents are (0.20, 0.018, 0.04) and centre (0, 0, 0.04), placing its top at z = 0.08. A separately logged minimum crossing clearance is `cube_z - 0.02 - 0.08`, sampled when the cube has ever lifted and `abs(cube_y) < 0.058`; it is not an extra abstention threshold.

## 5. When validation perturbations start

The original protocol is event-triggered, not fixed-time. In `scripts/run_revision_dagger.py:52-59`:

- `transport_perturbation`: replace the first three actions queried while phase == CROSS_BARRIER with `[0.3, -0.2, 0, -1]`.
- `open_during_close`: replace the first three actions queried while phase == CLOSE with `[0, 0, 0, +1]`.
- `oracle`: execute oracle actions without a perturbation.
- `expert_history`: execute the original expert's actions, with the oracle observing the same history; there is no perturbation.

All four conditions use scene seeds 2000-2029 with independent simulator resets. The old `oracle_validation.json` stores intervention counts and final episode lengths, but does not store intervention indices or a per-step validation trace. Consequently those original records alone cannot establish a numeric start step for each seed. `new_diagnostic_2026-10-16/` contains a separately labelled instrumented replay, its script, start-step table, and comparison with the old records. All 31 recorded fields match exactly for all 120 replayed episodes. The first transport intervention is executed action 32-41, and the first opening intervention is action 10-18, depending on seed. For seed 2000 specifically, transport uses actions 34-36 and opening uses actions 12-14 (one-based). A replayed step must not be described as an originally logged step.

## 6. What the expert synchronization run actually checked

The original `expert_history` branch (`scripts/run_revision_dagger.py:60-64`) queries the oracle, compares `label.phase == expert.phase.value` before calling `expert.act`, executes the expert action, and then updates the oracle once. Original stored results show 2,239 / 2,239 phase matches across 30 scenes, with 30 strict completions and no abstentions.

This establishes phase synchronization along those expert-driven histories. It does not establish full equality of internal memory, timers, frozen targets, grasp offsets, or proposed actions. Indeed, the recovery oracle freezes the measured arrival EE position on release (`recovery.py:226`), whereas the original expert uses `goal + grasp_offset` (`verified.py:144`). The implementations can have different targets despite identical phase labels.

Recommended reviewer wording:

> We synchronized the recovery oracle to the executed expert history and checked the semantic phase label before each action. All 2,239 queried phase labels agreed across the 30 validation scenes. This was a phase-synchronization check, not an equality test of every internal memory variable or of the proposed actions. The recovery extension intentionally differs from the nominal expert in its reconciliation rules and release target.

The new diagnostic separately records common memory fields. Phase, phase-step count, grasp offset, and grasp target match in all 2,239 samples both before expert actions and after oracle updates (numeric comparison: absolute tolerance 1e-8, relative tolerance 0). Release targets differ in 822 pre-action samples and 852 post-update samples; differences occur in all 30 scenes, with maximum absolute coordinate difference 0.0203462420 m. These new findings demonstrate that full target-memory identity does not hold. They are supplementary evidence, not a replacement for or reinterpretation of the original test.

## Retained per-scene evidence

Under `imitation-learning-grasp/artifacts/revision_study/`:

- `*_eval_40000_100.json`: 14 suites of 100 scene records, with seeds 40000-40099 in each suite. This comprises four learner formulations x three training seeds, one expert suite, and the one-round aggregation learner.
- `*_traces_40000_100/<scene_seed>.npz`: 1,400 corresponding per-step trajectories.
- Baseline for the paired aggregation comparison: `gru33_floor_s0_eval_40000_100.json` and `gru33_floor_s0.pt`.
- Aggregated learner: `dagger_gru33_floor_s0_eval_40000_100.json`, its trace folder, checkpoint, evaluation manifest, and training JSON.
- `dagger_round1.npz`: 7,218 valid oracle-labelled rows; includes observations, oracle actions, phases, episode IDs, actually executed actions, and teacher-mixture indicators.
- `dagger_round1_episodes.json`: 100 collection histories on seeds 1000-1099, row boundaries, final observations, and abstention reasons.
- `dagger_round1_manifest.json`: beta = 0.5, mixture RNG seed = 7, checkpoint/source hashes, 3,608 teacher actions, 87/100 mixed-policy completions, and 12 abstentions.
- `oracle_validation.json`, `oracle_validation_manifest.json`, `oracle_validation.log`: original four-condition validation.
- `dagger_collection.log`, `dagger_replay.log`, `dagger_training.log`, `dagger_evaluation.log`: retained execution logs.
- `demos.npz`, all model checkpoints/training JSON, reset audit, development records, and provenance files are included for context and reproducibility.

Derived convenience CSVs are under `derived_tables/`. The original JSON/NPZ files remain authoritative. The paired learner result is 97/100 before aggregation and 85/100 afterwards, with 3 improvements and 15 regressions. The 87/100 mixed-policy collection result must not be substituted for the learner-only test result.

## Preservation and scope

Original source and retained artifacts are copied byte-for-byte. A fresh package checksum manifest identifies every file. The original experiment outputs are not regenerated, overwritten, or edited. The separate diagnostic adds logging to the original validation protocol; it does not retrain or select a new policy or tune against held-out data. Successful finite probes do not establish recovery from arbitrary learner states or physical-world safety.
