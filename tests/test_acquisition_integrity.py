from pathlib import Path

import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
NSE_DIR = PROCESSED_DIR / "nse"
MARKET_DIR = NSE_DIR / "market"


def require_acquired_data():
    company_master = NSE_DIR / "company_master.parquet"
    market_files = list(MARKET_DIR.glob("market_*.parquet"))

    if not company_master.exists() or not market_files:
        pytest.skip(
            "Acquired datasets are not present. "
            "Run run_phase2.py first."
        )


def test_company_master():
    require_acquired_data()

    path = NSE_DIR / "company_master.parquet"

    df = pd.read_parquet(path)

    assert not df.empty
    assert "isin" in df.columns
    assert "series" in df.columns
    assert df["isin"].notna().all()


def test_market_data():
    require_acquired_data()

    market_files = sorted(
        MARKET_DIR.glob("market_*.parquet")
    )

    assert market_files

    df = pd.read_parquet(market_files[-1])

    assert not df.empty

    required_columns = {
        "isin",
        "trade_date",
        "close",
    }

    assert required_columns.issubset(df.columns)
    assert df["isin"].notna().all()
    assert df["trade_date"].notna().all()
    assert df["close"].notna().all()
    assert (df["close"] >= 0).all()


def test_market_dates():
    require_acquired_data()

    market_files = sorted(
        MARKET_DIR.glob("market_*.parquet")
    )

    df = pd.read_parquet(market_files[-1])

    assert df["trade_date"].min() <= df["trade_date"].max()


def test_market_no_duplicate_instruments():
    require_acquired_data()

    market_files = sorted(
        MARKET_DIR.glob("market_*.parquet")
    )

    df = pd.read_parquet(market_files[-1])

    duplicates = df.duplicated(
        subset=["trade_date", "instrument_id"]
    )

    assert not duplicates.any()
