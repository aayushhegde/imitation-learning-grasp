# imitation-learning-grasp

Behavior cloning for a simulated Franka Panda pick–cross–place task with an independent
strict-completion monitor (panda-gym 3.0.7, PyBullet).

- `src/ilgrasp/sim/verified_task.py`: the modified environment and the strict-completion monitor
- `src/ilgrasp/experts/verified.py`: the 13-phase scripted expert
- `notebooks/04_verified_expert_pipeline_student_en.ipynb`: demonstration collection loop and the
  original training and evaluation code
- `uv.lock`: pinned package versions for the checks below

## Checks added for the AJSR revision

Run from the repository root after installing the package (`pip install -e .`).

```bash
python scripts/monitor_predicate_checks.py --seeds 7 8 9 --out output/monitor_checks
python scripts/reset_history_audit.py --scenes 10000 10001 10002 10003 10004 --out output/reset_audit
python scripts/reset_history_audit.py --original-case --out output/reset_audit
python scripts/paired_statistics.py
```

- `monitor_predicate_checks.py` runs the expert with one deliberate defect per trajectory and
  checks that the monitor rejects it with the intended predicate (Supplementary Table S2).
- `reset_history_audit.py` replays reference trajectories after different earlier rollouts in a
  reused or rebuilt simulator and saves comparison traces (Supplementary Table S6). The
  `--original-case` option repeats the first audit (scene 10000 after a rollout at 10001).
- `paired_statistics.py` computes the Wilson intervals, exact McNemar tests, and Newcombe
  method-10 intervals for the paired comparisons (Table 2).

The saved outputs in `output/monitor_checks/` and `output/reset_audit/` come from macOS 13.4
(ARM64) with Python 3.11.15, PyBullet 3.2.7, NumPy 1.26.4, Gymnasium 1.3.0, and panda-gym 3.0.7.

## Original study records

`study_records/` holds the files from the runs reported in the paper, as prepared by the study
mentor on 2026-10-16. Start with `study_records/README_START_HERE.md` and
`study_records/REVIEWER_DETAILS.md`.

- `imitation-learning-grasp/`: the source snapshot used for the reported runs, including the
  recovery oracle (`src/ilgrasp/experts/recovery.py`), the study and aggregation scripts, tests, and
  `artifacts/revision_study/` (demonstrations, checkpoints, run manifests, logs, per-scene results,
  and per-step trajectories for every held-out suite)
- `derived_tables/`: CSV copies of the held-out scene records and the paired aggregation comparison
- `new_diagnostic_2026-10-16/`: a separately labeled replay of the oracle validation that records
  perturbation steps and memory comparisons; it is not part of the original logs
- `PACKAGE_SHA256.json`: a checksum for every file (`python3 study_records/verify_package.py`)

The original runs used a conda-forge environment (Python 3.12.13, PyTorch 2.13.0, NumPy 2.5.2,
Gymnasium 1.3.0, panda-gym 3.0.7, PyBullet 3.2.5) whose versions are recorded in every run
manifest. Some manifests still contain the original machine's absolute paths. Do not rerun the
original scripts into `artifacts/revision_study/`, because some modes overwrite their outputs.
