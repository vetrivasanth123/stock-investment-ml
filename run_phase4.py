from pathlib import Path
import json
from IPython.display import display


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/04-ml-model.ipynb"


def run_phase4():
    print("=" * 60)
    print("PHASE 4 — ML MODEL")
    print("=" * 60)

    if not NOTEBOOK.exists():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK}")

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

    if len(cells) != 5:
        raise ValueError(
            f"Expected 5 code cells, found {len(cells)}."
        )

    ns = {
        "__name__": "__main__",
        "__file__": str(NOTEBOOK),
        "display": display,
    }

    for i, cell in enumerate(cells, 1):
        print(f"\n{'=' * 60}")
        print(f"EXECUTING PHASE 4 CELL {i}/{len(cells)}")
        print("=" * 60)

        source = "".join(cell["source"])
        exec(
            compile(source, f"<phase4_cell_{i}>", "exec"),
            ns
        )

    required = {
        "model",
        "ranking",
        "metadata",
    }

    missing = required - ns.keys()
    if missing:
        raise RuntimeError(
            f"Phase 4 did not produce required outputs: {sorted(missing)}"
        )

    print("\n" + "=" * 60)
    print("PHASE 4 COMPLETE")
    print("=" * 60)
    print("Inference date:", metadata["inference_date"])
    print("Stocks scored :", metadata["rows_scored"])
    print("Target        :", metadata["target"])
    print(
        "Validation MAE:",
        round(metadata["validation_mae"], 6)
    )
    print(
        "Validation R² :",
        round(metadata["validation_r2"], 6)
    )
    print(
        "Mean Daily IC :",
        round(
            metadata["mean_daily_information_coefficient"],
            6
        )
    )
    print("\nTop 10:")
    display(ranking.head(10))


if __name__ == "__main__":
    run_phase4()
