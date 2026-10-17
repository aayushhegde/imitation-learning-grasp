from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = [
    "04_verified_expert_pipeline_student_en.ipynb",
    "04_verified_expert_pipeline_solutions_en.ipynb",
    "04_verified_expert_pipeline_student_zh.ipynb",
    "04_verified_expert_pipeline_solutions_zh.ipynb",
]


def test_four_verified_notebooks_are_synchronized_and_compile() -> None:
    structures = []
    for name in NAMES:
        notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        structures.append(
            [
                (
                    cell["id"],
                    cell["cell_type"],
                    # Completion provenance can differ after a student fills
                    # an exercise; it is not part of the shared cell structure.
                    tuple(tag for tag in cell.get("metadata", {}).get("tags", [])
                          if tag != "solution-replaced"),
                )
                for cell in notebook["cells"]
            ]
        )
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{name}:{cell['id']}", "exec")
        exercise_count = sum(
            "exercise" in cell.get("metadata", {}).get("tags", [])
            for cell in notebook["cells"]
        )
        assert exercise_count == 13
    assert all(structure == structures[0] for structure in structures)


def test_solution_notebooks_have_outputs_and_no_placeholders() -> None:
    for name in (NAMES[1], NAMES[3]):
        notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        assert "TODO" not in source
        assert "NotImplementedError" not in source
        assert all(
            output.get("output_type") != "error"
            for cell in notebook["cells"]
            for output in cell.get("outputs", [])
        )
        assert sum(cell.get("execution_count") is not None for cell in notebook["cells"]) == 29
