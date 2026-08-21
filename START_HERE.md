# Notebook 04 — Verified Pick–Cross–Place: missing modules

This bundle contains everything `notebooks/04_verified_expert_pipeline_student_en.ipynb`
needs. The `ModuleNotFoundError: No module named 'ilgrasp.data'` happened because these
packages were never committed to the repo — nothing is wrong with your setup.

You were right that `ilgrasp.data`, `ilgrasp.sim` and `ilgrasp.visualization` were missing.
There is a fourth one: `ilgrasp.experts`. The notebook never imports it directly, but
`ilgrasp/visualization/recording.py` does, so cell 1 needs it too.

## 1. Copy the files in

From the root of your clone:

```bash
rsync -av /path/to/send_to_student/ .
```

This overwrites `pyproject.toml` and `src/ilgrasp/__init__.py`, and adds
`src/ilgrasp/{sim,data,experts,visualization}/`, the notebook, the animation, some
scripts, and four tests.

## 2. Delete the two superseded modules

```bash
git rm src/ilgrasp/env.py src/ilgrasp/constants.py
```

`GraspEnv` now lives in `src/ilgrasp/sim/env.py` and the observation contract in
`src/ilgrasp/sim/observation.py`. If you keep the old files you end up with two
different `GraspEnv` classes and two different step budgets (50 vs 100 vs 180) that
silently disagree — exactly what the docstring in your `constants.py` warned about.

The new `src/ilgrasp/__init__.py` still re-exports `GraspEnv`, `flatten_obs`, `OBS_DIM`
and `ACTION_DIM`, so **notebook 01 keeps working unchanged**. Two names are gone
on purpose:

| Old name | Use instead |
|---|---|
| `OBJ_POS_SLICE` | `slice(7, 10)` — see `FLAT_LABELS` in `ilgrasp.sim.observation` |
| `MAX_EPISODE_STEPS` | read it off the env: `env.config.max_episode_steps` |

The step budget is per-task now (`GraspEnv` defaults to 100, `VerifiedPlacementEnv`
uses 180), so a single global constant no longer has a correct value.

## 3. Install the two missing dependencies

Your `pyproject.toml` listed only `gymnasium`, `panda-gym`, `numpy` and `torch`. Cell 1
also needs `matplotlib` (imported directly) and `pillow` (pulled in by
`ilgrasp.visualization.recording`). Add `pandas` too if you want `ilgrasp.demos`.

Use `--no-deps` so pip does not try to re-resolve torch and numpy and undo your pins:

```bash
python -m pip install -e . --no-deps && python -m pip install matplotlib pillow pandas pytest
```

## 4. About your macOS 13.4 pins

`torch<2.3` and `numpy<2` are correct and you should keep them. I checked every numpy
and torch symbol used by these modules and by the notebook — `np.asarray`, `np.r_`,
`np.isin`, `torch.as_tensor`, `torch.inference_mode`, `torch.where` and the rest all
exist in numpy 1.26 and torch 2.2. There is no numpy-2-only or torch-2.3-only API
anywhere in the bundle, so nothing here needs the newer wheels.

Two things that look like version hazards but are not:

- `sim/observation.py` uses `.astype(np.float32, copy=False)`. That is the ndarray
  method, which has always accepted `copy` — not `np.asarray(copy=...)`, which is
  numpy 2.0 only.
- `policies/mlp.py` passes `weights_only=True` to `torch.load` explicitly, so it
  behaves the same on old and new torch.

Your `gymnasium==1.3.0` pin is also what these modules were built against, and the
`render_mode` fix you made is already handled: `sim/env.py` and `sim/verified_task.py`
omit the `render_mode` kwarg entirely when it is `None` rather than passing `None`.

## 5. Verify, in this order

```bash
python -m pytest tests/ -q
```

Four tests should pass: observation contract, verified task, demo database, expert.

Then run notebook cell 1. It should print the project path and a temp directory. Cell 3
displays the animation — `notebooks/assets/verified_pick_cross_place.gif` is included in
this bundle, so it will not spend minutes re-recording 180 steps of 1280×720 video.

## 6. Where to start

E01 (`RobotState.from_raw_obs`) and E02 (`bounded_move_action`) are already filled in —
we did those together in class. Your first exercise is **E03** (`pick_decision`), then
E04–E13. Each one has a check cell right after it.

## What is in this bundle

```
pyproject.toml                     dependency + build config (replaces yours)
src/ilgrasp/__init__.py            re-exports for notebook 01 compatibility
src/ilgrasp/sim/                   GraspEnv, observation contract, Episode, VerifiedPlacementEnv
src/ilgrasp/data/                  DemoDatabase, TrainingData
src/ilgrasp/experts/               ScriptedExpert + reference verified expert
src/ilgrasp/visualization/         record_verified_expert_gif
notebooks/04_..._student_en.ipynb  the lab
notebooks/assets/*.gif             cell 3 animation (5.2 MB)
scripts/                           record_verified_animation, inspect_env, watch_random
tests/                             four tests covering the modules above
```
