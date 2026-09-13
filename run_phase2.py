from datetime import date
from src.data.providers.nse_security_master import build_security_master
from src.data.providers.nse_market import update_market_data

def run_phase2():
    print("=" * 60)
    print("PHASE 2 — NSE DATA ACQUISITION")
    print("=" * 60)

    print("\n[1/3] Building NSE company master...")
    company_master = build_security_master()
    universe = company_master[
        company_master["series"].eq("EQ")
    ].reset_index(drop=True)

    print("Company master:", len(company_master))
    print("EQ universe:", len(universe))

    print("\n[2/3] Updating NSE market history...")
    market = update_market_data(
        start_date=date(2024, 7, 8),
        end_date=date.today(),
    )

    print("Market rows:", len(market))
    print("Companies:", market["isin"].nunique())
    print("Trading days:", market["trade_date"].nunique())
    print("Market saved: data/processed/nse/market/market_history.parquet")

    print("\n[3/3] Acquisition validation...")
    required = {"isin", "trade_date", "close"}
    missing = required - set(market.columns)

    if missing or market.empty:
        raise ValueError(
            f"Invalid market data: missing={sorted(missing)}, "
            f"empty={market.empty}"
        )

    print("Market data validation: PASSED")
    print("\n" + "=" * 60)
    print("PHASE 2 ACQUISITION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_phase2()
