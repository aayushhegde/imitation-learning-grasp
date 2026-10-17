"""Execute the revised notebook with the explicit simulator kernel and atomic save."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "notebooks/04_verified_expert_pipeline_student_en_fixed.ipynb"
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(
    notebook,
    timeout=600,
    kernel_name="revision-sim",
    resources={"metadata": {"path": str(ROOT)}},
)
client.execute()
temporary = path.with_suffix(".executed.ipynb")
nbformat.write(notebook, temporary)
temporary.replace(path)
print(f"Executed and saved {path}; no cell errors.")
