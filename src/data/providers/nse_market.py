from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zipfile import ZipFile
import json

import pandas as pd
import requests


PROJECT_DIR = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_DIR / "data" / "raw" / "nse" / "market"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed" / "nse" / "market"
METADATA_DIR = PROJECT_DIR / "data" / "raw" / "nse" / "metadata"

NSE_MARKET_BASE_URL = (
    "https://nsearchives.nseindia.com/content/cm"
)

MARKET_MANIFEST_PATH = (
    METADATA_DIR / "market_downloads.json"
)

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/",
}


def generate_dates(start_date, end_date):
    current = start_date

    while current <= end_date:
        yield current
        current += timedelta(days=1)


def download_nse_udiff_bhavcopy(trading_date):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    filename = (
        f"BhavCopy_NSE_CM_0_0_0_"
        f"{trading_date:%Y%m%d}_F_0000.csv.zip"
    )

    path = RAW_DIR / filename

    if path.exists():
        return path

    url = f"{NSE_MARKET_BASE_URL}/{filename}"

    response = requests.get(
        url,
        headers=NSE_HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    path.write_bytes(response.content)

    return path


def read_nse_udiff_bhavcopy(path):
    with ZipFile(path, "r") as archive:
        csv_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise ValueError(
                f"No CSV inside {path.name}"
            )

        with archive.open(csv_files[0]) as csv_file:
            df = pd.read_csv(csv_file)

    df.columns = df.columns.str.strip()

    return df


def normalize_nse_market_data(df):
    required = {
        "TradDt",
        "Sgmt",
        "FinInstrmTp",
        "FinInstrmId",
        "ISIN",
        "TckrSymb",
        "SctySrs",
        "FinInstrmNm",
        "OpnPric",
        "HghPric",
        "LwPric",
        "ClsPric",
        "LastPric",
        "PrvsClsgPric",
        "TtlTradgVol",
        "TtlTrfVal",
        "TtlNbOfTxsExctd",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing UDiFF columns: {sorted(missing)}"
        )

    df = df[list(required)].copy()

    df = df.rename(
        columns={
            "TradDt": "trade_date",
            "Sgmt": "segment",
            "FinInstrmTp": "instrument_type",
            "FinInstrmId": "instrument_id",
            "ISIN": "isin",
            "TckrSymb": "nse_symbol",
            "SctySrs": "series",
            "FinInstrmNm": "instrument_name",
            "OpnPric": "open",
            "HghPric": "high",
            "LwPric": "low",
            "ClsPric": "close",
            "LastPric": "last_price",
            "PrvsClsgPric": "previous_close",
            "TtlTradgVol": "volume",
            "TtlTrfVal": "turnover",
            "TtlNbOfTxsExctd": "number_of_trades",
        }
    )

    df["trade_date"] = pd.to_datetime(
        df["trade_date"],
        errors="coerce",
    )

    for column in [
        "segment",
        "instrument_type",
        "isin",
        "nse_symbol",
        "series",
        "instrument_name",
    ]:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "last_price",
        "previous_close",
        "volume",
        "turnover",
        "number_of_trades",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


def validate_market_data(df):
    required = {
        "trade_date",
        "isin",
        "nse_symbol",
        "series",
        "close",
        "volume",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing normalized columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Market dataset is empty.")

    if df["trade_date"].isna().any():
        raise ValueError("Invalid trade dates.")

    if df["isin"].isna().any():
        raise ValueError("Missing ISIN values.")

    if df["close"].isna().any():
        raise ValueError("Missing close prices.")

    if (df["close"] < 0).any():
        raise ValueError("Negative close prices.")

    if df.duplicated(
        ["trade_date", "instrument_id"]
    ).any():
        raise ValueError(
            "Duplicate trade-date/instrument records."
        )


def record_market_download(
    trading_date,
    raw_path,
    status,
):
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    if MARKET_MANIFEST_PATH.exists():
        manifest = json.loads(
            MARKET_MANIFEST_PATH.read_text(
                encoding="utf-8"
            )
        )
    else:
        manifest = []

    record = {
        "provider": "National Stock Exchange of India Limited (NSE)",
        "dataset": "CM-UDiFF Common Bhavcopy Final",
        "data_date": trading_date.isoformat(),
        "source_url": (
            f"{NSE_MARKET_BASE_URL}/"
            f"BhavCopy_NSE_CM_0_0_0_"
            f"{trading_date:%Y%m%d}_F_0000.csv.zip"
        ),
        "local_file": str(raw_path),
        "status": status,
        "accessed_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    manifest = [
        item
        for item in manifest
        if not (
            item["dataset"] == record["dataset"]
            and item["data_date"] == record["data_date"]
        )
    ]

    manifest.append(record)

    MARKET_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def collect_market_data(start_date, end_date):
    datasets = []

    for trading_date in generate_dates(
        start_date,
        end_date,
    ):
        filename = (
            f"BhavCopy_NSE_CM_0_0_0_"
            f"{trading_date:%Y%m%d}_F_0000.csv.zip"
        )

        raw_path = RAW_DIR / filename

        try:
            status = "cached" if raw_path.exists() else "downloaded"

            raw_path = download_nse_udiff_bhavcopy(
                trading_date
            )

            record_market_download(
                trading_date,
                raw_path,
                status,
            )

            df = normalize_nse_market_data(
                read_nse_udiff_bhavcopy(raw_path)
            )

            validate_market_data(df)
            datasets.append(df)

        except Exception:
            record_market_download(
                trading_date,
                raw_path,
                "unavailable",
            )

    if not datasets:
        raise RuntimeError(
            "No valid market data collected."
        )

    result = (
        pd.concat(datasets, ignore_index=True)
        .drop_duplicates(
            ["trade_date", "instrument_id"]
        )
        .sort_values(
            ["trade_date", "instrument_id"]
        )
        .reset_index(drop=True)
    )

    return result


def save_market_data(df):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    start = df["trade_date"].min().strftime("%Y%m%d")
    end = df["trade_date"].max().strftime("%Y%m%d")

    path = (
        PROCESSED_DIR /
        f"market_{start}_{end}.parquet"
    )

    df.to_parquet(path, index=False)

    return path
