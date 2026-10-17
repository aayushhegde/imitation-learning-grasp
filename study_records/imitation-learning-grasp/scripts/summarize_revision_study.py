"""Create manuscript tables/figure inputs from complete per-episode result records."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_revision_study import save_json, summarize

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts/revision_study"
PAPER = ROOT.parent / "project_Aayush"
CONDITIONS = ["mlp22", "gru33", "gru33_floor", "mlp33_floor"]
LABELS = ["MLP22", "GRU33", "GRU33 + floor", "MLP33 + floor"]


def main():
    episodes = {}
    rows = []
    for condition in CONDITIONS + ["expert"]:
        for seed in [0] if condition == "expert" else [0, 1, 2]:
            label = f"{condition}_s{seed}"
            path = STUDY / f"{label}_eval_40000_100.json"
            records = json.loads(path.read_text())
            assert [r["seed"] for r in records] == list(range(40000, 40100))
            assert all(r["reset_mode"] == "independent" for r in records)
            assert all(not r["strict_success"] or r["native_success_seen"] for r in records)
            episodes[label] = records
            training_path = STUDY / f"{label}_training.json"
            training = json.loads(training_path.read_text()) if training_path.exists() else None
            contacts = [
                r["first_barrier_contact"] for r in records if r.get("first_barrier_contact")
            ]
            row = {
                "condition": condition,
                "training_seed": seed,
                **summarize(records),
                "milestones": {
                    k: sum(bool(r[k]) for r in records)
                    for k in ("ever_lifted", "ever_crossed", "ever_released", "retreated")
                },
                "first_contact_contexts": dict(Counter(c["context"] for c in contacts)),
                "first_contact_bodies": dict(
                    Counter(c.get("body_name", str(c.get("body"))) for c in contacts)
                ),
            }
            if training:
                row.update(
                    {
                        "best_epoch": training["best_epoch"],
                        "best_validation_loss": min(training["validation_loss"]),
                        "final_validation_loss": training["validation_loss"][-1],
                        "final_training_loss": training["train_loss"][-1],
                        "training_seconds": training["elapsed_seconds"],
                    }
                )
            rows.append(row)
    aggregate = []
    for condition in CONDITIONS:
        rates = [r["strict"] for r in rows if r["condition"] == condition]
        native = [r["native"] for r in rows if r["condition"] == condition]
        aggregate.append(
            {
                "condition": condition,
                "strict_percent_mean": float(np.mean(rates)),
                "strict_percent_sd": float(np.std(rates, ddof=1)),
                "strict_percent_range": [min(rates), max(rates)],
                "native_percent_mean": float(np.mean(native)),
            }
        )
    comparisons = []
    for reference, revised in [
        ("gru33", "gru33_floor"),
        ("mlp22", "gru33"),
        ("mlp33_floor", "gru33_floor"),
    ]:
        for seed in [0, 1, 2]:
            a = np.array([r["strict_success"] for r in episodes[f"{reference}_s{seed}"]])
            b = np.array([r["strict_success"] for r in episodes[f"{revised}_s{seed}"]])
            gained = int((~a & b).sum())
            lost = int((a & ~b).sum())
            n = gained + lost
            p = (
                min(1.0, 2 * sum(math.comb(n, i) for i in range(min(gained, lost) + 1)) / 2**n)
                if n
                else 1.0
            )
            comparisons.append(
                {
                    "reference": reference,
                    "revised": revised,
                    "seed": seed,
                    "gained": gained,
                    "lost": lost,
                    "difference_percentage_points": int(b.sum() - a.sum()),
                    "exact_mcnemar_p": p,
                }
            )
    summary = {
        "test_seeds": [40000, 40099],
        "n_scenes": 100,
        "training_seeds": [0, 1, 2],
        "rows": rows,
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "inference_note": "Per-model Wilson intervals and unadjusted paired tests; shared scenes/training dataset prevent treating 300 runs as independent.",
    }
    save_json(PAPER / "revision/study_summary.json", summary)
    lines = [
        "# Executed matched holdout study",
        "",
        "Each model uses the same 100 independently reset scenes (40000–40099).",
        "",
        "| Condition | Train seed | Strict | Native | Wilson 95% |",
        "|---|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['condition']} | {r['training_seed']} | {r['strict']}/100 | {r['native']}/100 | {r['wilson95'][0]:.3f}–{r['wilson95'][1]:.3f} |"
        )
    lines.extend(
        [
            "",
            "Means and sample SD across the three training seeds (not a pooled n=300 binomial interval):",
            "",
        ]
    )
    for r in aggregate:
        lines.append(
            f"- {r['condition']}: {r['strict_percent_mean']:.2f}% ± {r['strict_percent_sd']:.2f} pp."
        )
    (PAPER / "revision/study_summary.md").write_text("\n".join(lines) + "\n")
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    colors = ["#777777", "#2370A3", "#7030A0", "#D18C2A"]
    for i, (condition, label, color) in enumerate(zip(CONDITIONS, LABELS, colors)):
        rr = [r for r in rows if r["condition"] == condition]
        vals = np.array([r["strict"] for r in rr])
        axes[0].bar(i, vals.mean(), color=color, alpha=0.75)
        axes[0].scatter(i + np.array([-0.12, 0, 0.12]), vals, c="black", s=25, zorder=3)
        axes[1].plot([0, 1, 2], vals, "o-", label=label, color=color)
    axes[0].set_xticks(range(4), ["MLP22", "GRU33", "GRU33\n+ floor", "MLP33\n+ floor"])
    axes[0].set(
        ylabel="Strict success (%)", ylim=(0, 105), title="Mean and individual training seeds"
    )
    axes[1].set(
        xticks=[0, 1, 2], xlabel="Training seed", ylabel="Strict success (%)", ylim=(0, 105)
    )
    axes[1].legend(fontsize=8, loc="lower left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(PAPER / "figures/revision_success.pdf")
    fig.savefig(PAPER / "figures/revision_success.png", dpi=180)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
