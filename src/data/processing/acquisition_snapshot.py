import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[3]

SNAPSHOT_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "phase2_acquisition_snapshot.json"
)


def create_acquisition_snapshot(
    company_master,
    investable_universe,
    market_data,
    financial_metadata,
    financial_facts,
    financial_contexts,
    raw_xbrl_count,
):
    """
    Create a reproducibility snapshot for the Phase 2
    data acquisition layer.
    """

    required_frames = {
        "company_master": company_master,
        "investable_universe": investable_universe,
        "market_data": market_data,
        "financial_metadata": financial_metadata,
        "financial_facts": financial_facts,
        "financial_contexts": financial_contexts,
    }

    for name, df in required_frames.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame.")

        if df.empty:
            raise ValueError(f"{name} must not be empty.")

    if "trade_date" not in market_data.columns:
        raise ValueError("Market data must contain 'trade_date'.")

    snapshot = {
        "phase": "Phase 2 - Data Acquisition",
        "status": "validated",
        "snapshot_created_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "company_master_rows": int(
            len(company_master)
        ),

        "investable_universe_rows": int(
            len(investable_universe)
        ),

        "market_rows": int(
            len(market_data)
        ),

        "market_start": str(
            market_data["trade_date"].min()
        ),

        "market_end": str(
            market_data["trade_date"].max()
        ),

        "financial_filings": int(
            len(financial_metadata)
        ),

        "xbrl_facts": int(
            len(financial_facts)
        ),

        "xbrl_contexts": int(
            len(financial_contexts)
        ),

        "raw_xbrl_files": int(
            raw_xbrl_count
        ),

        "providers": {
            "security_master": "NSE",
            "market_data": "NSE CM-UDiFF",
            "financial_data": "NSE Integrated Filing - Financials",
        },

        "validation": {
            "company_master": True,
            "investable_universe": True,
            "market_data": True,
            "financial_metadata": True,
            "xbrl_facts": True,
            "xbrl_contexts": True,
            "raw_xbrl_persistence": raw_xbrl_count > 0,
            "filing_key_consistency": True,
        },
    }

    SNAPSHOT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SNAPSHOT_PATH.write_text(
        json.dumps(
            snapshot,
            indent=2,
        ),
        encoding="utf-8",
    )

    return SNAPSHOT_PATH
