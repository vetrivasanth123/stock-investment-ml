from pathlib import Path

import pandas as pd
import requests


PROJECT_DIR = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_DIR / "data" / "raw" / "nse" / "security_master"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed" / "nse"

NSE_EQUITY_URL = (
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
)

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/",
}


def download_nse_equity_master():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    path = RAW_DIR / "EQUITY_L.csv"

    response = requests.get(
        NSE_EQUITY_URL,
        headers={
            **NSE_HEADERS,
            "Accept": "text/csv,*/*",
        },
        timeout=30,
    )
    response.raise_for_status()

    path.write_bytes(response.content)

    return path


def load_nse_equity_master(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df = df.rename(
        columns={
            "SYMBOL": "nse_symbol",
            "NAME OF COMPANY": "company_name",
            "SERIES": "series",
            "DATE OF LISTING": "listing_date",
            "PAID UP VALUE": "paid_up_value",
            "MARKET LOT": "market_lot",
            "ISIN NUMBER": "isin",
            "FACE VALUE": "face_value",
        }
    )

    df["listing_date"] = pd.to_datetime(
        df["listing_date"],
        format="%d-%b-%Y",
        errors="coerce",
    )

    for column in [
        "nse_symbol",
        "company_name",
        "series",
        "isin",
    ]:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    return df


def validate_security_master(df):
    required = {
        "nse_symbol",
        "company_name",
        "series",
        "listing_date",
        "paid_up_value",
        "market_lot",
        "isin",
        "face_value",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Security master is empty.")

    if df["isin"].isna().any():
        raise ValueError("Missing ISIN values.")

    if df["nse_symbol"].isna().any():
        raise ValueError("Missing NSE symbols.")


def build_security_master():
    raw_path = download_nse_equity_master()
    df = load_nse_equity_master(raw_path)

    validate_security_master(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / "company_master.parquet"
    df.to_parquet(output_path, index=False)

    return df


def build_investable_universe():
    company_master = build_security_master()

    return (
        company_master[
            company_master["series"].eq("EQ")
        ]
        .reset_index(drop=True)
    )
