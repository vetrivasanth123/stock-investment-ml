from pathlib import Path
import json
import pandas as pd
from IPython.display import display
from src.data.providers.nse_market import load_market_data

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "notebooks/03-target-creation.ipynb"
TARGET_DIR = ROOT / "data/processed/targets"


def main():
    print("=" * 60)
    print("PHASE 3 — TARGET CREATION")
    print("=" * 60)

    if not NOTEBOOK.exists():
        raise FileNotFoundError(NOTEBOOK)

    history = load_market_data()

    if history.empty:
        raise RuntimeError(
            "Phase 2 market history not found. Run Phase 2 first."
        )

    history["trade_date"] = pd.to_datetime(history["trade_date"])
    history = (
        history
        .drop_duplicates(["trade_date", "instrument_id"])
        .sort_values(["nse_symbol", "trade_date"])
        .reset_index(drop=True)
    )

    print(f"\nRows        : {len(history):,}")
    print(f"Companies   : {history['isin'].nunique():,}")
    print(f"Trading days: {history['trade_date'].nunique():,}")
    print(
        f"Date range  : {history['trade_date'].min().date()} → "
        f"{history['trade_date'].max().date()}"
    )

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

    if len(cells) != 4:
        raise ValueError(f"Expected 4 code cells, found {len(cells)}")

    ns = {
        "__name__": "__main__",
        "__file__": str(NOTEBOOK),
        "recent_history": history,
        "display": display,
    }

    for i, cell in enumerate(cells, 1):
        print("\n" + "=" * 60)
        print(f"EXECUTING PHASE 3 CELL {i}/4")
        print("=" * 60)
        exec(
            compile(
                "".join(cell["source"]),
                f"<phase3_cell_{i}>",
                "exec",
            ),
            ns,
        )

    required = {
        "nifty_ntr_benchmark",
        "phase3_target",
        "final_target",
        "phase3_target_metadata",
    }

    missing = required - ns.keys()
    if missing:
        raise RuntimeError(f"Missing outputs: {sorted(missing)}")

    target = ns["final_target"]
    col = "target_excess_return_252d"

    if target.empty or col not in target.columns or target[col].isna().any():
        raise RuntimeError("Invalid final target.")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    target_path = TARGET_DIR / "phase3_target.parquet"
    metadata_path = TARGET_DIR / "phase3_target_metadata.json"

    target.to_parquet(target_path, index=False)
    metadata_path.write_text(
        json.dumps(ns["phase3_target_metadata"], indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETE")
    print("=" * 60)
    print("Rows:", f"{len(target):,}")
    print("Companies:", f"{target['isin'].nunique():,}")
    print("Decision dates:", f"{target['decision_date'].nunique():,}")
    print("Target:", target_path)
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
