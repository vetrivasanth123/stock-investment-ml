from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[3]


def integrate_market_with_company_master(
    market_df,
    company_master,
):
    company_columns = [
        "isin",
        "company_name",
        "listing_date",
    ]

    required_market = {
        "isin",
        "series",
        "trade_date",
    }

    missing = required_market - set(market_df.columns)

    if missing:
        raise ValueError(
            f"Missing market columns: {sorted(missing)}"
        )

    result = market_df.merge(
        company_master[company_columns],
        on="isin",
        how="inner",
        validate="many_to_one",
    )

    result = (
        result[
            result["series"].eq("EQ")
        ]
        .sort_values(
            ["trade_date", "isin"]
        )
        .reset_index(drop=True)
    )

    if result.empty:
        raise ValueError(
            "No market records matched the company master."
        )

    return result
