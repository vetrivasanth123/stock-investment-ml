from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
IN = ROOT / "data/processed/model/latest_stock_ranking.parquet"
FUND = ROOT / "data/processed/recommendations/fundamentals_input.csv"
OUT = ROOT / "data/processed/recommendations"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 5 — FINAL INVESTMENT RANKING")
print("=" * 60)

# Load Phase 4 ranking
df = pd.read_parquet(IN)
if df.empty:
    raise RuntimeError("Phase 4 ranking is empty.")

df = df[
    df["close"].gt(0) &
    df["predicted_excess_return_252d"].notna()
].copy()

# Load optional fundamentals
if FUND.exists():
    fund = pd.read_csv(FUND).drop_duplicates("nse_symbol")
    df = df.merge(fund, on="nse_symbol", how="left")
else:
    fund = pd.DataFrame()
    print("Warning: fundamentals file not found. Using ML-only ranking.")

# Required fundamental columns
FUND_COLS = [
    "roe_reported", "roce_reported", "pe",
    "sales_growth_5y", "profit_growth_5y",
    "debt_equity", "opm"
]

for c in FUND_COLS:
    if c not in df:
        df[c] = np.nan

# ML score
df["ml_score"] = df["predicted_excess_return_252d"].rank(
    pct=True, method="average"
) * 100

# Fundamental component scores
scores = pd.DataFrame(index=df.index)

for c in [
    "roe_reported",
    "roce_reported",
    "sales_growth_5y",
    "profit_growth_5y",
    "opm"
]:
    scores[c] = df[c].rank(pct=True)

scores["debt_equity"] = 1 - df["debt_equity"].rank(pct=True)
scores["pe"] = 1 - df["pe"].rank(pct=True)

df["fundamental_coverage"] = df[FUND_COLS].notna().mean(axis=1)
df["fundamental_score"] = scores.mean(axis=1, skipna=True) * 100

# Use fundamentals only where enough data actually exists
df["final_score"] = np.where(
    df["fundamental_coverage"] >= 0.70,
    0.70 * df["ml_score"] + 0.30 * df["fundamental_score"],
    df["ml_score"]
)

# Transparent recommendation tiers
df["recommendation"] = "AVOID"

df.loc[
    df["final_score"] >= 70,
    "recommendation"
] = "WATCH"

df.loc[
    (df["final_score"] >= 90) &
    (df["fundamental_coverage"] >= 0.70),
    "recommendation"
] = "BUY"

# Highest score first
df = df.sort_values(
    "final_score",
    ascending=False
).reset_index(drop=True)

df["final_rank"] = np.arange(1, len(df) + 1)

cols = [
    "final_rank",
    "rank",
    "isin",
    "nse_symbol",
    "company_name",
    "close",
    "predicted_excess_return_252d",
    "ml_score",
    "roe_reported",
    "roce_reported",
    "pe",
    "sales_growth_5y",
    "profit_growth_5y",
    "debt_equity",
    "opm",
    "fundamental_coverage",
    "fundamental_score",
    "final_score",
    "recommendation"
]

result = df[cols]

# Save
ranking_path = OUT / "latest_recommendations.parquet"
metadata_path = OUT / "phase5_metadata.json"

result.to_parquet(
    ranking_path,
    index=False
)

ral = result[
    result["nse_symbol"].str.upper().isin(["RALLIS", "RAL"])
]

print("Stocks evaluated:", f"{len(result):,}")
print(
    "Stocks with >=70% fundamentals:",
    f"{(result['fundamental_coverage'] >= 0.70).sum():,}"
)

print("\n=== TOP 25 ===")
print(
    result.head(25).to_string(index=False)
)

if not ral.empty:
    print("\n=== RALLIS ===")
    print(
        ral.to_string(index=False)
    )

metadata = {
    "stocks_evaluated": int(len(result)),
    "fundamental_records": int(len(fund)),
    "stocks_with_70pct_fundamentals": int(
        (result["fundamental_coverage"] >= 0.70).sum()
    ),
    "top_stock": str(result.iloc[0]["nse_symbol"]),
    "top_final_score": float(result.iloc[0]["final_score"]),
    "rallis_present": bool(not ral.empty)
}

metadata_path.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

print("\nSaved:")
print(ranking_path)
print(metadata_path)

print("\n" + "=" * 60)
print("PHASE 5 COMPLETE")
print("=" * 60)
