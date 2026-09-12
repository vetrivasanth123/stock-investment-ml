from datetime import date
from pathlib import Path

from src.data.providers.nse_security_master import (
    build_security_master,
)
from src.data.providers.nse_market import (
    collect_market_data,
    save_market_data,
)
from src.data.processing.market_processing import (
    integrate_market_with_company_master,
)


PROJECT_DIR = Path(__file__).resolve().parent

MARKET_START = date(2026, 8, 18)
MARKET_END = date(2026, 8, 22)


def run_phase2():
    print("=" * 60)
    print("PHASE 2 — NSE DATA ACQUISITION")
    print("=" * 60)

    print("\n[1/3] Building NSE company master...")

    company_master = build_security_master()

    investable_universe = (
        company_master[
            company_master["series"].eq("EQ")
        ]
        .reset_index(drop=True)
    )

    print("Company master:", len(company_master))
    print("EQ universe:", len(investable_universe))

    print("\n[2/3] Acquiring NSE market data...")

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

    print("\n[3/3] Acquisition validation...")

    required_columns = {
        "isin",
        "trade_date",
        "close",
    }

    missing = required_columns - set(
        company_prices.columns
    )

    if missing:
        raise ValueError(
            f"Missing market columns: {sorted(missing)}"
        )

    if company_prices.empty:
        raise ValueError(
            "Integrated market dataset is empty."
        )

    print("Market data validation: PASSED")

    print("\n" + "=" * 60)
    print("PHASE 2 ACQUISITION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_phase2()
