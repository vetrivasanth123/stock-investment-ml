from datetime import date, timedelta
from pathlib import Path
from zipfile import ZipFile
import json

import pandas as pd
import requests

PROJECT_DIR = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_DIR / "data/raw/nse/market"
PROCESSED_DIR = PROJECT_DIR / "data/processed/nse/market"
METADATA_DIR = PROJECT_DIR / "data/raw/nse/metadata"

NSE_URL = "https://nsearchives.nseindia.com/content/cm"
CANONICAL_PATH = PROCESSED_DIR / "market_history.parquet"
MANIFEST_PATH = METADATA_DIR / "market_downloads.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nseindia.com/",
}


def generate_dates(start, end):
    while start <= end:
        yield start
        start += timedelta(days=1)


def _raw_path(d):
    return RAW_DIR / (
        f"BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    )


def download_nse_udiff_bhavcopy(d):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _raw_path(d)
    if path.exists():
        return path
    url = f"{NSE_URL}/{path.name}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def read_nse_udiff_bhavcopy(path):
    with ZipFile(path) as z:
        csv = next((x for x in z.namelist() if x.lower().endswith(".csv")), None)
        if not csv:
            raise ValueError(f"No CSV in {path.name}")
        return pd.read_csv(z.open(csv))


def normalize_nse_market_data(df):
    cols = {
        "TradDt": "trade_date", "Sgmt": "segment",
        "FinInstrmTp": "instrument_type", "FinInstrmId": "instrument_id",
        "ISIN": "isin", "TckrSymb": "nse_symbol", "SctySrs": "series",
        "FinInstrmNm": "instrument_name", "OpnPric": "open",
        "HghPric": "high", "LwPric": "low", "ClsPric": "close",
        "LastPric": "last_price", "PrvsClsgPric": "previous_close",
        "TtlTradgVol": "volume", "TtlTrfVal": "turnover",
        "TtlNbOfTxsExctd": "number_of_trades",
    }
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[list(cols)].rename(columns=cols).copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")

    for c in ["segment", "instrument_type", "isin", "nse_symbol",
              "series", "instrument_name"]:
        df[c] = df[c].astype("string").str.strip()

    for c in ["open", "high", "low", "close", "last_price",
              "previous_close", "volume", "turnover", "number_of_trades"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def validate_market_data(df):
    required = {"trade_date", "instrument_id", "isin", "nse_symbol",
                "series", "close", "volume"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    if df.empty:
        raise ValueError("Market dataset is empty.")
    if df["trade_date"].isna().any():
        raise ValueError("Invalid trade dates.")
    if df["isin"].isna().any():
        raise ValueError("Missing ISIN.")
    if df["close"].isna().any():
        raise ValueError("Missing close prices.")
    if (df["close"] < 0).any():
        raise ValueError("Negative close prices.")
    if df.duplicated(["trade_date", "instrument_id"]).any():
        raise ValueError("Duplicate trade_date/instrument_id.")


def record_market_download(d, path, status):
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    if MANIFEST_PATH.exists():
        records = json.loads(MANIFEST_PATH.read_text())

    records = [r for r in records if r["data_date"] != str(d)]
    records.append({
        "provider": "NSE",
        "dataset": "CM UDiFF Bhavcopy",
        "data_date": str(d),
        "source_url": f"{NSE_URL}/{_raw_path(d).name}",
        "local_file": str(path) if path else None,
        "status": status,
    })
    MANIFEST_PATH.write_text(json.dumps(records, indent=2))


def collect_market_data(start_date, end_date):
    frames = []

    for d in generate_dates(start_date, end_date):
        try:
            path = download_nse_udiff_bhavcopy(d)
            status = "cached" if path.exists() else "downloaded"
            record_market_download(d, path, status)

            df = normalize_nse_market_data(read_nse_udiff_bhavcopy(path))
            validate_market_data(df)
            frames.append(df)

        except Exception as e:
            record_market_download(d, None, f"unavailable: {e}")

    if not frames:
        raise RuntimeError("No valid NSE market data collected.")

    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["trade_date", "instrument_id"])
        .sort_values(["trade_date", "instrument_id"])
        .reset_index(drop=True)
    )


def load_market_data(path=CANONICAL_PATH):
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    validate_market_data(df)

    return df.sort_values(
        ["trade_date", "instrument_id"]
    ).reset_index(drop=True)


def save_market_data(df, path=CANONICAL_PATH):
    validate_market_data(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def update_market_data(start_date=date(2024, 7, 8), end_date=None):
    end_date = end_date or date.today()
    existing = load_market_data()

    if existing.empty:
        new_data = collect_market_data(start_date, end_date)
    else:
        latest = existing["trade_date"].max().date()
        if latest >= end_date:
            return existing
        new_data = collect_market_data(latest + timedelta(days=1), end_date)

    combined = pd.concat([existing, new_data], ignore_index=True)
    combined = (
        combined
        .drop_duplicates(["trade_date", "instrument_id"])
        .sort_values(["trade_date", "instrument_id"])
        .reset_index(drop=True)
    )

    save_market_data(combined)
    return combined
