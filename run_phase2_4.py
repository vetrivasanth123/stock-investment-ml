
from pathlib import Path

import nbformat
from nbclient import NotebookClient


PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = (
    PROJECT_DIR
    / "notebooks"
    / "02-pandas-data-handling.ipynb"
)


def run_phase2_4():
    print("=" * 60)
    print("PHASE 2.4 — PANDAS DATA HANDLING")
    print("=" * 60)

    print("\n[1/2] Loading notebook...")

    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"Notebook not found: {NOTEBOOK_PATH}"
        )

    print("Notebook:", NOTEBOOK_PATH)

    with NOTEBOOK_PATH.open(
        "r",
        encoding="utf-8",
    ) as notebook_file:
        notebook = nbformat.read(
            notebook_file,
            as_version=4,
        )

    print("Notebook loaded successfully.")

    print("\n[2/2] Executing notebook...")

    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={
            "metadata": {
                "path": str(PROJECT_DIR),
            }
        },
    )

    client.execute()

    print("Notebook execution: PASSED")

    print("\n" + "=" * 60)
    print("PHASE 2.4 PANDAS DATA HANDLING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_phase2_4()
