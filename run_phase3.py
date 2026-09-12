# ============================================================
# PHASE 3 — TARGET CREATION
# run_phase3.py
# ============================================================

from datetime import date
from pathlib import Path
import json

import pandas as pd

from src.data.providers import nse_market


PROJECT_ROOT = Path(__file__).resolve().parent

NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "03-target-creation.ipynb"
)

# Verified NSE UDiFF history boundary used by Phase 3.
# The end date remains fully dynamic.
HISTORY_START = date(2024, 7, 8)


def load_phase3_market_history():
    """Acquire the verified historical NSE market dataset."""

    today = date.today()

    print("\n" + "=" * 60)
    print("PHASE 3 — ACQUIRING HISTORICAL MARKET DATA")
    print("=" * 60)

    print(
        f"\nRange: {HISTORY_START} → {today}"
    )

    recent_history = (
        nse_market.collect_market_data(
            HISTORY_START,
            today,
        )
    )

    if recent_history.empty:
        raise RuntimeError(
            "Historical market acquisition returned no data."
        )

    recent_history["trade_date"] = pd.to_datetime(
        recent_history["trade_date"],
        errors="coerce",
    )

    recent_history = (
        recent_history
        .drop_duplicates(
            subset=[
                "trade_date",
                "instrument_id",
            ]
        )
        .sort_values(
            [
                "nse_symbol",
                "trade_date",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"\nRows       : {len(recent_history):,}"
    )

    print(
        f"Companies  : "
        f"{recent_history['isin'].nunique():,}"
    )

    print(
        f"Trading days: "
        f"{recent_history['trade_date'].nunique():,}"
    )

    print(
        f"Date range : "
        f"{recent_history['trade_date'].min().date()}"
        f" → "
        f"{recent_history['trade_date'].max().date()}"
    )

    return recent_history


def main():

    print("=" * 60)
    print("PHASE 3 — TARGET CREATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Validate notebook
    # --------------------------------------------------------

    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"Notebook not found:\n{NOTEBOOK_PATH}"
        )

    with NOTEBOOK_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        notebook = json.load(file)

    if notebook.get("nbformat") != 4:
        raise ValueError(
            "Expected nbformat 4 notebook."
        )

    code_cells = [
        cell
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]

    if len(code_cells) != 4:
        raise ValueError(
            "Phase 3 notebook must contain exactly "
            f"4 code cells; found {len(code_cells)}."
        )

    # --------------------------------------------------------
    # Acquire full Phase 3 history
    # --------------------------------------------------------

    recent_history = load_phase3_market_history()

    # --------------------------------------------------------
    # Execute four notebook cells
    # --------------------------------------------------------

    namespace = {
        "__name__": "__main__",
        "__file__": str(NOTEBOOK_PATH),
        "recent_history": recent_history,
    }

    for index, cell in enumerate(
        code_cells,
        start=1,
    ):

        source = "".join(
            cell.get("source", [])
        )

        print("\n" + "=" * 60)
        print(
            f"EXECUTING PHASE 3 CELL {index}/4"
        )
        print("=" * 60)

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
        "nifty_ntr_benchmark",
        "phase3_target",
        "final_target",
        "phase3_target_metadata",
    ]

    missing = [
        name
        for name in required_variables
        if name not in namespace
    ]

    if missing:
        raise RuntimeError(
            "Phase 3 did not produce required outputs:\n"
            f"{missing}"
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
            f"Missing target column: {target_column}"
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
