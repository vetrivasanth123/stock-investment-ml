# ============================================================
# PHASE 3 — TARGET CREATION
# run_phase3.py
# ============================================================

from datetime import date
from pathlib import Path
import json

import pandas as pd
from IPython.display import display

from src.data.providers import nse_market


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03-target-creation.ipynb"
TARGET_DIR = PROJECT_ROOT / "data" / "processed" / "targets"

HISTORY_START = date(2024, 7, 8)


def load_market_history():
    """Acquire the historical NSE market data required by Phase 3."""

    print("\n" + "=" * 60)
    print("PHASE 3 — ACQUIRING HISTORICAL MARKET DATA")
    print("=" * 60)

    today = date.today()
    print(f"\nRange: {HISTORY_START} → {today}")

    df = nse_market.collect_market_data(
        HISTORY_START,
        today,
    )

    if df.empty:
        raise RuntimeError(
            "Historical market acquisition returned no data."
        )

    df["trade_date"] = pd.to_datetime(
        df["trade_date"],
        errors="coerce",
    )

    df = (
        df
        .drop_duplicates(
            subset=["trade_date", "instrument_id"]
        )
        .sort_values(["nse_symbol", "trade_date"])
        .reset_index(drop=True)
    )

    print(f"\nRows        : {len(df):,}")
    print(f"Companies   : {df['isin'].nunique():,}")
    print(f"Trading days: {df['trade_date'].nunique():,}")
    print(
        f"Date range  : "
        f"{df['trade_date'].min().date()} → "
        f"{df['trade_date'].max().date()}"
    )

    return df


def main():
    print("=" * 60)
    print("PHASE 3 — TARGET CREATION")
    print("=" * 60)

    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"Notebook not found: {NOTEBOOK_PATH}"
        )

    notebook = json.loads(
        NOTEBOOK_PATH.read_text(encoding="utf-8")
    )

    if notebook.get("nbformat") != 4:
        raise ValueError("Expected nbformat 4 notebook.")

    cells = [
        c for c in notebook.get("cells", [])
        if c.get("cell_type") == "code"
    ]

    if len(cells) != 4:
        raise ValueError(
            f"Expected exactly 4 Phase 3 code cells; found {len(cells)}."
        )

    namespace = {
        "__name__": "__main__",
        "__file__": str(NOTEBOOK_PATH),
        "recent_history": load_market_history(),
        "display": display,
    }

    for i, cell in enumerate(cells, 1):
        print("\n" + "=" * 60)
        print(f"EXECUTING PHASE 3 CELL {i}/4")
        print("=" * 60)

        exec(
            compile(
                "".join(cell.get("source", [])),
                f"<phase3_cell_{i}>",
                "exec",
            ),
            namespace,
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    required = [
        "nifty_ntr_benchmark",
        "phase3_target",
        "final_target",
        "phase3_target_metadata",
    ]

    missing = [
        name for name in required
        if name not in namespace
    ]

    if missing:
        raise RuntimeError(
            f"Missing Phase 3 outputs: {missing}"
        )

    final_target = namespace["final_target"]
    target_column = "target_excess_return_252d"

    if final_target.empty:
        raise RuntimeError("Final target is empty.")

    if target_column not in final_target.columns:
        raise RuntimeError(
            f"Missing target column: {target_column}"
        )

    if final_target[target_column].isna().any():
        raise RuntimeError(
            "Final target contains NaN values."
        )

    # --------------------------------------------------------
    # Persist final target
    # --------------------------------------------------------

    TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_path = (
        TARGET_DIR / "phase3_target.parquet"
    )

    metadata_path = (
        TARGET_DIR / "phase3_target_metadata.json"
    )

    final_target.to_parquet(
        target_path,
        index=False,
    )

    metadata_path.write_text(
        json.dumps(
            namespace["phase3_target_metadata"],
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PHASE 3 RUNNER VALIDATION")
    print("=" * 60)

    print(f"\nFinal target rows : {len(final_target):,}")
    print(f"Companies         : {final_target['isin'].nunique():,}")
    print(
        f"Decision dates    : "
        f"{final_target['decision_date'].nunique():,}"
    )
    print(f"Target column     : {target_column}")

    print(f"\nTarget saved      : {target_path}")
    print(f"Metadata saved    : {metadata_path}")

    print("\n" + "=" * 60)
    print("PHASE 3: COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
