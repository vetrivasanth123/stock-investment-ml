from pathlib import Path

import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
)

FINANCIAL_DIR = (
    PROCESSED_DIR
    / "nse"
    / "financials"
)

XBRL_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "nse"
    / "financials"
    / "xbrl"
)


def require_acquired_data():
    """
    Skip acquisition-data tests when the repository
    has been cloned without local acquired datasets.
    """

    if not (
        PROCESSED_DIR.exists()
        and FINANCIAL_DIR.exists()
    ):
        pytest.skip(
            "Acquired data is not present. "
            "Run the acquisition notebook first."
        )


def test_company_master():
    require_acquired_data()

    path = (
        PROCESSED_DIR
        / "nse"
        / "company_master.parquet"
    )

    assert path.exists()

    df = pd.read_parquet(path)

    assert not df.empty
    assert "isin" in df.columns
    assert df["isin"].notna().all()


def test_financial_metadata():
    require_acquired_data()

    path = (
        FINANCIAL_DIR
        / "filing_metadata.parquet"
    )

    assert path.exists()

    df = pd.read_parquet(path)

    assert not df.empty
    assert "filing_key" in df.columns
    assert df["filing_key"].notna().all()


def test_xbrl_facts():
    require_acquired_data()

    path = (
        FINANCIAL_DIR
        / "xbrl_facts.parquet"
    )

    assert path.exists()

    df = pd.read_parquet(path)

    assert not df.empty
    assert "filing_key" in df.columns
    assert df["filing_key"].notna().all()


def test_xbrl_contexts():
    require_acquired_data()

    path = (
        FINANCIAL_DIR
        / "xbrl_contexts.parquet"
    )

    assert path.exists()

    df = pd.read_parquet(path)

    assert not df.empty
    assert "filing_key" in df.columns
    assert df["filing_key"].notna().all()


def test_financial_filing_keys():
    require_acquired_data()

    metadata = pd.read_parquet(
        FINANCIAL_DIR
        / "filing_metadata.parquet"
    )

    facts = pd.read_parquet(
        FINANCIAL_DIR
        / "xbrl_facts.parquet"
    )

    contexts = pd.read_parquet(
        FINANCIAL_DIR
        / "xbrl_contexts.parquet"
    )

    metadata_keys = set(
        metadata["filing_key"]
    )

    fact_keys = set(
        facts["filing_key"]
    )

    context_keys = set(
        contexts["filing_key"]
    )

    assert fact_keys.issubset(
        metadata_keys
    )

    assert context_keys.issubset(
        metadata_keys
    )


def test_raw_xbrl_files():
    require_acquired_data()

    if not XBRL_DIR.exists():
        pytest.skip(
            "Raw XBRL directory is not present."
        )

    files = list(
        XBRL_DIR.glob("*.xml")
    )

    assert files

    for path in files:
        assert path.stat().st_size > 0


def test_financial_provenance():
    require_acquired_data()

    path = (
        FINANCIAL_DIR
        / "financial_downloads.json"
    )

    assert path.exists()
    assert path.stat().st_size > 0
