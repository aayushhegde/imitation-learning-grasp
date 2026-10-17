# Oracle validation evidence: new diagnostic replay, 2026-10-16

This directory contains a newly instrumented replay of the original `validate()` protocol. It is **not an original validation log**. No repository source or historical artifact was modified. All four original conditions were rerun on the original 30 scenes, seeds 2000–2029, with independent resets. Instrumentation only records state; it does not choose or modify control actions beyond the original perturbation protocol.

## What the original files establish

- `scripts/run_revision_dagger.py:52–59` starts the transport perturbation at the **first queried CROSS_BARRIER action** and replaces the first three such actions by `[0.3, -0.2, 0, -1]`. The opening perturbation starts at the **first queried CLOSE action** and replaces the first three such actions by `[0, 0, 0, 1]`. Onsets are phase-triggered, not fixed episode indices.
- The original output schema (`scripts/run_revision_dagger.py:67–69`) saves total intervention count and final episode step, but no intervention step indices or per-step traces. The original stored results therefore cannot independently supply exact numeric onset steps.
- `scripts/run_revision_dagger.py:60–64` queries the oracle, compares `label.phase == expert.phase.value` **before** calling `expert.act`, executes the expert action, then calls `oracle.observe_transition()` exactly once. The original check compares phase labels only. It never compares full memory or the oracle's proposed action to the expert's action.
- Original `artifacts/revision_study/oracle_validation.json` stores 120 episodes. Each condition has 30 strict successes and no abstention; both perturbation conditions have exactly 3 interventions in each episode. Expert-history has 2,239 phase matches in 2,239 queries, with equality in every episode. The expert-history block starts at line 2973.
- The original manifest records oracle source SHA-256 `2bedbae95ae6013246b0ccbe42b5fcfa8d32cebf0438d48214d02b2f4dedeee0`; the current recovery implementation matches it exactly. Other source hashes and serialized task/controller configs were not saved in that manifest, so those cannot independently be hash-verified against its original execution.

## What this new replay adds

All **31 fields of all 120 historical episode records match exactly**, including numeric final measurements, final episode length, query count, phase match count, intervention count, strict/native outcomes, and abstentions. The new manifest records current source/config hashes, package versions, and the exact executed script hash. Package versions also match the historical study manifest.

`perturbation_steps_by_seed.csv` is the compact per-seed onset and intervention table. Both 0-based pre-action steps and 1-based executed-action numbers are provided. The same information for all conditions is in `intervention_steps.json` / `.csv`.

- Transport first executed intervention action: **32–41**, varying by scene; all three interventions are consecutive.
- Open-during-close first executed intervention action: **10–18**, varying by scene; all three interventions are consecutive.
- Example seed 2000: transport actions **34, 35, 36** (pre-steps 33, 34, 35); opening actions **12, 13, 14** (pre-steps 11, 12, 13).

Full common-memory comparisons use `np.allclose(atol=1e-8, rtol=0)` and equality iff both values are `None` for absent targets. These are diagnostics, not assertions of equivalence.

| Common memory field | Pre-act equal / total | Post-observe equal / total |
|---|---:|---:|
| phase | 2239 / 2239 | 2239 / 2239 |
| phase_steps | 2239 / 2239 | 2239 / 2239 |
| grasp_offset | 2239 / 2239 | 2239 / 2239 |
| grasp_target | 2239 / 2239 | 2239 / 2239 |
| release_target | 1417 / 2239 | 1387 / 2239 |

Release targets differ in all 30 expert-history episodes: 822 pre-act samples and 852 post-observe samples. The first difference is seed 2000 immediately after executed action 46 (pre-step 45), visible before action 47; the largest component difference there is 0.0130307795 m. Maximum component difference over all records is 0.0203462420 m. This is consistent with the code: the oracle saves measured post-step `ee.copy()` (`recovery.py:226`); the expert saves its commanded `goal + grasp_offset` target (`verified.py:141–144`). **Phase synchronization is supported; full-memory identity is not.**

Oracle proposal versus executed expert action was recorded as an additional observational diagnostic. At strict absolute tolerance 1e-8, 497 of 2239 differ; 120 of these are float-level differences. At tolerance 1e-6, or NumPy's default rtol 1e-5 with atol 1e-8, 377 differ: 360 OPEN samples plus 17 APPROACH samples. The oracle intentionally has recovery behavior and a measured release target; this protocol executes only the expert's action and does not establish action equivalence. See `additional_checks.json` for counts and an initial example.

## Unit-test scope

`tests/test_recovery_oracle.py:10–32` defines a query-idempotence/timer test at seed 7: 30 repeated queries preserve `phase_steps` and `close_steps`, three executed closing actions produce `close_steps == 3`, and one actually executed opening action resets the close counter without leaving CLOSE. This is a separate test protocol, **not** the validation condition that forces opening on the first three CLOSE actions.

Lines 35–47 define missing-observe, double-observe, and stale-reset failures. Lines 50–58 define abstention after an irreversible collision flag. Lines 61–79 define transport recovery checks on seeds 3, 7, 13. None tests equality of full oracle/expert memory. These tests were inspected, not rerun by this diagnostic.

## Reproduction and files

The portable `replay_oracle_validation.py` accepts `--repo` and `--output`. Its default repository is `../imitation-learning-grasp` relative to the script, matching the reviewer package layout; `--output` is required. It refuses to use the original `artifacts/revision_study` directory or any directory containing an existing `diagnostic_manifest.json` as output, preserving delivered runs.

```sh
PYTHONDONTWRITEBYTECODE=1 /path/to/python replay_oracle_validation.py --repo /path/to/imitation-learning-grasp --output /path/to/new-diagnostic-output
```

The exact script used for this recorded run is preserved as `original_executed_script.py`. The portable script differs only in CLI path setup and guards against historical or existing diagnostic output; its `--help` and overwrite refusal were checked. `diagnostic_manifest.json` identifies the executed original script by hash, and `additional_checks.json` records both hashes. No controller or validation control logic differs between them.

- `diagnostic_manifest.json`: provenance, source/config hashes, versions, step convention, outputs.
- `diagnostic_episode_results.json`: all newly rerun final records, same schema as historical results.
- `diagnostic_summary.json`: original field comparisons, conditions, memory comparisons, onset ranges.
- `additional_checks.json`: exact full-record equality and additional action comparison.
- `perturbation_steps_by_seed.csv`: compact reviewer-facing per-seed intervention steps.
- `step_records.jsonl`: detailed pre-query and post-transition snapshots, original phase comparison, actual/proposed actions, perturbation flags.
- `execution.log`: run output.
