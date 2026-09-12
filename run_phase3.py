# ============================================================
# PHASE 3 — TARGET CREATION
# run_phase3.py
# ============================================================

from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "03-target-creation.ipynb"
)

MARKET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nse"
    / "market"
)


def load_market_history():
    """Load all processed NSE market Parquet files."""

    files = sorted(
        MARKET_DIR.glob("market_*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            "No processed NSE market Parquet files found in:\n"
            f"{MARKET_DIR}\n\n"
            "Run Phase 2 / acquire the historical market data first."
        )

    print(
        f"\nLoading {len(files)} market Parquet file(s)..."
    )

    frames = []

    for path in files:

        print(f" - {path.name}")

        df = pd.read_parquet(path)

        frames.append(df)

    recent_history = pd.concat(
        frames,
        ignore_index=True,
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
                "trade_date",
                "instrument_id",
            ]
        )
        .reset_index(drop=True)
    )

    if recent_history.empty:
        raise ValueError(
            "Processed market dataset is empty."
        )

    print(
        f"\nLoaded market rows : "
        f"{len(recent_history):,}"
    )

    print(
        f"Companies           : "
        f"{recent_history['isin'].nunique():,}"
    )

    print(
        f"Trading days        : "
        f"{recent_history['trade_date'].nunique():,}"
    )

    print(
        f"Date range          : "
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
    # 1. Validate notebook
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
    # 2. Load Phase 2 market data
    # --------------------------------------------------------

    recent_history = load_market_history()

    # --------------------------------------------------------
    # 3. Execute the four Phase 3 cells
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
    # 4. Final validation
    # --------------------------------------------------------

    required_variables = [
        "recent_history",
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
