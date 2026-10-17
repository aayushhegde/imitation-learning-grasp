# Independent final evidence QA

The following checks were performed directly against persisted artifacts, not copied from the prose.

- All13 main100-scene result sets: exact seed order40000–40099; strict/native counts; categorized failures; milestone counts; Wilson intervals; provenance sidecar artifact hashes.
- All four learned-condition summaries: three-seed mean, sample standard deviation and range.
- All nine paired comparisons: gained/lost outcomes, percentage-point differences and two-sided exact McNemar probabilities recomputed from per-scene booleans.
- All1,300 saved main-test trajectory files, representing149,874 commands: shape/finite-value checks, normalized action bounds, binary gripper commands and result step counts.
- Independently reconstructed lift, crossing, release, final retreat, native hit and consecutive stability count from each complete observation sequence plus final observation. Every reconstructed value matched its evaluation record. Strict success recomputed from stability and the retained barrier-violation flag also matched. Penetration/contact validity itself depends on the simulator's saved contact telemetry and was not inferred from observation geometry.
- Revised notebook: every code cell parses; all prior output/execution counts were cleared before the new execution. Cell53 calls the shared recurrent trainer with split seed7 and scale floor0.05, and cell38 uses independent physics resets. The historical executed notebook is preserved separately.

Verified main strict results (training seeds0/1/2): MLP22=18/24/23, GRU33=93/89/84, GRU33floor0.05=97/96/97, MLP33floor0.05=2/45/17. Expert=100. Every denominator is100. The new MLP22seed0 value18/100 is not evidence that the student's unavailable historical18/100 run was reproduced.

Prose issues reported for correction: notebook's claim that nearly identical observations establish a non-single-valued map; obsolete exact student/solutions-cell-order assertion; precision of floor-selection chronology; distinction between observed phase coverage and a phase-coverage eligibility filter; expert control has no three-model training-seed replication. Substantive changed manuscript passages inspected at this stage were marked with `revision`/`revisionblock` using RGB112,48,160.

These checks establish internal consistency of the stored experiment evidence. They do not provide an independent rerun of all training, a causal history-only ablation, or validation of unrecorded student claims.

The reproducible audit is `project_Aayush/revision/verify_final_evidence.py`. Run it with the simulator Python interpreter (NumPy required). It writes `final_evidence_checks.json`, including the audit-script hash, input-summary hash, all13 result-file hashes, and checked episode/command/test counts. The final successful execution reported1,300 trajectories,149,874 commands, four training-seed aggregates and nine paired tests. These checks exclude the exploratory DAgger result and do not read an in-progress notebook execution.

## Final paper pass after quantitative tables were written

All13 success-table rows (strict/native counts, both Wilson intervals, collision counts) and all13 milestone-table rows (four lifecycle counts, minimum sampled clearance and contributing sample count) were independently checked against the raw JSON files. Paired, loss and neighbor tables matched their audited artifacts.

The later exploratory DAgger result was separately checked at the aggregate/per-scene level:85 strict and90 native successes; Wilson95% intervals76.7164–90.6940% and82.5634–94.4771%;3 improved scenes and15 regressions relative to the seed0 baseline; exact two-sided McNemar p=0.007537841796875; best validation epoch67. This supplemental check does not add its trajectories to the1,300 main-study trace audit above.

A final scope error was found and corrected by the paper authoring agent: the three DAgger collisions occur **before** crossing (scenes40036,40086,40098), so the abstract/conclusion's after-crossing statement must refer only to the two main behavior-cloned GRU conditions. The0.00345160067m reset-audit discrepancy is a maximum absolute position-coordinate difference, not an L2 distance; this wording clarification was also sent to the paper agent. No other material numeric/protocol inconsistencies were found in the final source pass.
