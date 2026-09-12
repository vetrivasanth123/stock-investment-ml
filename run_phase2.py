from datetime import date
from pathlib import Path

from src.data.providers.nse_security_master import (
    build_security_master,
)
from src.data.providers.nse_market import (
    collect_market_data,
    save_market_data,
)
from src.data.providers.nse_financials import (
    collect_filing_metadata,
    acquire_financial_filings,
)
from src.data.processing.market_processing import (
    integrate_market_with_company_master,
)
from src.data.processing.financial_normalization import (
    combine_filing_data,
    validate_filing_dataset,
)
from src.data.processing.acquisition_snapshot import (
    create_acquisition_snapshot,
)


PROJECT_DIR = Path(__file__).resolve().parent

MARKET_START = date(2026, 8, 18)
MARKET_END = date(2026, 8, 22)

FILING_PAGE_SIZE = 100
FILING_MAX_PAGES = 1


def run_phase2():
    print("=" * 60)
    print("PHASE 2 — NSE DATA ACQUISITION")
    print("=" * 60)

    print("\n[1/6] Building NSE company master...")

    company_master = build_security_master()

    investable_universe = (
        company_master[
            company_master["series"].eq("EQ")
        ]
        .reset_index(drop=True)
    )

    print("Company master:", len(company_master))
    print("EQ universe:", len(investable_universe))

    print("\n[2/6] Acquiring NSE market data...")

    market_data = collect_market_data(
        MARKET_START,
        MARKET_END,
    )

    company_prices = integrate_market_with_company_master(
        market_data,
        investable_universe,
    )

    market_path = save_market_data(
        company_prices
    )

    print("Market rows:", len(company_prices))
    print(
        "Companies:",
        company_prices["isin"].nunique(),
    )
    print("Market saved:", market_path)

    print(
        "\n[3/6] Acquiring financial filing metadata..."
    )

    filing_records = collect_filing_metadata(
        page_size=FILING_PAGE_SIZE,
        max_pages=FILING_MAX_PAGES,
    )

    print(
        "Financial filings:",
        len(filing_records),
    )

    print("\n[4/6] Acquiring XBRL financial data...")

    (
        financial_metadata_df,
        financial_facts_df,
        financial_contexts_df,
        financial_manifest_df,
    ) = acquire_financial_filings(
        filing_records
    )

    validate_filing_dataset(
        financial_metadata_df,
        financial_facts_df,
        financial_contexts_df,
    )

    print(
        "Filings:",
        len(financial_metadata_df),
    )
    print(
        "Facts:",
        len(financial_facts_df),
    )
    print(
        "Contexts:",
        len(financial_contexts_df),
    )

    print(
        "\n[5/6] Processing and saving financial data..."
    )

    financial_dataset = combine_filing_data(
        financial_metadata_df,
        financial_facts_df,
        financial_contexts_df,
    )

    output_dir = (
        PROJECT_DIR
        / "data"
        / "processed"
        / "nse"
        / "financials"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    financial_metadata_df.to_parquet(
        output_dir / "filing_metadata.parquet",
        index=False,
    )

    financial_facts_df.to_parquet(
        output_dir / "xbrl_facts.parquet",
        index=False,
    )

    financial_contexts_df.to_parquet(
        output_dir / "xbrl_contexts.parquet",
        index=False,
    )

    financial_dataset.to_parquet(
        output_dir / "financial_dataset.parquet",
        index=False,
    )

    financial_manifest_df.to_json(
        output_dir / "financial_downloads.json",
        orient="records",
        indent=2,
    )

    print(
        "Financial dataset:",
        financial_dataset.shape,
    )
    print(
        "Financial data saved:",
        output_dir,
    )

    print("\n[6/6] Creating Phase 2 snapshot...")

    raw_xbrl_dir = (
        PROJECT_DIR
        / "data"
        / "raw"
        / "nse"
        / "financials"
        / "xbrl"
    )

    snapshot_path = create_acquisition_snapshot(
        company_master=company_master,
        investable_universe=investable_universe,
        market_data=company_prices,
        financial_metadata=financial_metadata_df,
        financial_facts=financial_facts_df,
        financial_contexts=financial_contexts_df,
        raw_xbrl_count=len(
            list(raw_xbrl_dir.glob("*.xml"))
        ),
    )

    print("Snapshot:", snapshot_path)

    print("\n" + "=" * 60)
    print("PHASE 2 ACQUISITION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_phase2()
