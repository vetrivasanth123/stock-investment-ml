# ============================================================
# PHASE 3 — TARGET CREATION
# run_phase3.py
# ============================================================

from pathlib import Path
import json
import runpy
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "03-target-creation.ipynb"
)


def main():
    print("=" * 60)
    print("PHASE 3 — TARGET CREATION")
    print("=" * 60)

    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"Notebook not found: {NOTEBOOK_PATH}"
        )

    # --------------------------------------------------------
    # Load notebook
    # --------------------------------------------------------

    with NOTEBOOK_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        notebook = json.load(file)

    if notebook.get("nbformat") != 4:
        raise ValueError(
            "Expected nbformat 4 notebook."
        )

    cells = notebook.get("cells", [])

    code_cells = [
        cell
        for cell in cells
        if cell.get("cell_type") == "code"
    ]

    if len(code_cells) != 4:
        raise ValueError(
            f"Phase 3 notebook must contain exactly "
            f"4 code cells; found {len(code_cells)}."
        )

    # --------------------------------------------------------
    # Execute cells in notebook order
    # --------------------------------------------------------

    namespace = {
        "__name__": "__main__",
        "__file__": str(NOTEBOOK_PATH),
    }

    for index, cell in enumerate(
        code_cells,
        start=1,
    ):

        source = "".join(
            cell.get("source", [])
        )

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"EXECUTING PHASE 3 CELL {index}/4"
        )

        print(
            f"{'=' * 60}"
        )

        exec(
            compile(
                source,
                f"<phase3_cell_{index}>",
                "exec",
            ),
            namespace,
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    required_variables = [
        "recent_history",
        "nifty_ntr_benchmark",
        "phase3_target",
        "final_target",
        "phase3_target_metadata",
    ]

    missing_variables = [
        name
        for name in required_variables
        if name not in namespace
    ]

    if missing_variables:
        raise RuntimeError(
            "Phase 3 did not produce required outputs: "
            f"{missing_variables}"
        )

    final_target = namespace["final_target"]

    if final_target.empty:
        raise RuntimeError(
            "Final Phase 3 target is empty."
        )

    target_column = (
        "target_excess_return_252d"
    )

    if target_column not in final_target.columns:
        raise RuntimeError(
            f"Missing final target column: {target_column}"
        )

    if final_target[target_column].isna().any():
        raise RuntimeError(
            "Final target contains NaN values."
        )

    print("\n" + "=" * 60)
    print("PHASE 3 RUNNER VALIDATION")
    print("=" * 60)

    print(
        f"\nFinal target rows : "
        f"{len(final_target):,}"
    )

    print(
        f"Companies         : "
        f"{final_target['isin'].nunique():,}"
    )

    print(
        f"Decision dates    : "
        f"{final_target['decision_date'].nunique():,}"
    )

    print(
        f"Target column     : "
        f"{target_column}"
    )

    print("\n" + "=" * 60)
    print("PHASE 3: COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
