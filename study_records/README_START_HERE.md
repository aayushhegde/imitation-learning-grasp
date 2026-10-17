# Aayush reviewer evidence package

Prepared 2026-10-16. Start with `REVIEWER_DETAILS.md`, which answers the six requested implementation questions with source line references.

## What is included

- `imitation-learning-grasp/src/`: original Python implementation, including the recovery oracle, nominal expert, task/environment, training and evaluation components.
- `imitation-learning-grasp/scripts/` and `tests/`: original experiment scripts and tests, together with the repository README, LICENSE, pyproject.toml and uv.lock.
- `imitation-learning-grasp/artifacts/revision_study/`: retained original results, logs, checkpoints and arrays. The files that were ignored by Git are explicitly included here.
- `project_Aayush/revision/`: selected original summary and provenance records supporting the manuscript results.
- `derived_tables/heldout_per_scene.csv`: a readable consolidation of all 1,400 held-out scene records.
- `derived_tables/gru33_floor_s0_dagger_paired_per_scene.csv`: the 100 paired baseline-versus-aggregation scenes.
- `new_diagnostic_2026-10-16/`: NEW instrumented validation replay, onset-step table, memory-comparison results and the script used to produce them. These are explicitly separated from the original September records.
- `PACKAGE_SHA256.json` and `verify_package.py`: byte-level package checks and basic result summaries. Run `python3 verify_package.py` from any directory.

## The files requested most directly

| Request | File relative to imitation-learning-grasp/ |
|---|---|
| Recovery oracle source | src/ilgrasp/experts/recovery.py |
| Config defaults and nominal expert | src/ilgrasp/experts/verified.py |
| Physical monitor and executed-action record | src/ilgrasp/sim/verified_task.py |
| Validation and aggregation protocol | scripts/run_revision_dagger.py |
| Original oracle validation | artifacts/revision_study/oracle_validation.json |
| Original validation definitions | artifacts/revision_study/oracle_validation_manifest.json |
| Baseline GRU33-F seed-0 held-out scenes | artifacts/revision_study/gru33_floor_s0_eval_40000_100.json |
| Aggregated learner held-out scenes | artifacts/revision_study/dagger_gru33_floor_s0_eval_40000_100.json |
| Aggregation labels and executed actions | artifacts/revision_study/dagger_round1.npz |
| Aggregation scene outcomes and abstentions | artifacts/revision_study/dagger_round1_episodes.json |
| Aggregation settings and hashes | artifacts/revision_study/dagger_round1_manifest.json |

The corresponding `*_traces_40000_100/` directories contain a per-step NPZ for each of the 100 held-out scene seeds (40000-40099). The aggregation collection scenes are 1000-1099, and the oracle validation scenes are 2000-2029.

## Evidence interpretation

The paired learner result remains 97/100 before aggregation and 85/100 afterwards. Collection used a 50% teacher mixture and achieved 87/100; that is a different measurement and is not the deployed learner result. Only one aggregation round and one training seed were tested for this comparison.

The historical expert-history validation checked phase agreement, with 2,239/2,239 matches. It did not assert full memory or action equality. The new instrumented replay reproduces all 120 original validation records and records perturbation timing and common memory fields. Its release-target comparison demonstrates why full-memory identity must not be claimed.

The original validation logged the number of interventions, not the step indices. Numeric intervention indices in this package are newly reconstructed by running the unchanged protocol, and are labelled accordingly. One-based executed-step numbers and zero-based pre-action state indices are both recorded to avoid ambiguity.

## Reproduction and file preservation

The archive preserves historical result/source bytes. Original manifests may reference an earlier absolute project location. Resolve their relative suffix under this package; the original manifests are retained without rewriting those paths. The current local working tree contains uncommitted/untracked research code, so a bare `git archive` would omit required files; this package explicitly includes the source snapshot and ignored evidence. The package hash file identifies the supplied bytes. Versioned provenance that was reconstructed later is labelled as a migration in its original files.

Do not rerun original scripts into `artifacts/revision_study/`: some modes overwrite their output files. Use a new output directory for any new experiment. The bundled repository README documents environment setup. This archive does not include local virtual environments, simulator binaries, caches, `.git` history, or chat logs. Exact reproduction on a different machine or dependency version is not guaranteed; dependency versions for the historical study are recorded in its manifests.

A straightforward first inspection does not require a simulator: run the package verifier and open the JSON/CSV files. For the new diagnostic's execution options, run its Python script with `--help` after setting up the simulator environment. No training or held-out policy selection was performed for the new diagnostic.
