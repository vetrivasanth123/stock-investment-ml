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
print("PHASE 5 — INVESTMENT RECOMMENDATION + FUNDAMENTALS")
print("=" * 60)

df = pd.read_parquet(IN)

if df.empty:
    raise RuntimeError("Phase 4 ranking is empty.")
if not FUND.exists():
    raise FileNotFoundError(f"Missing fundamentals file: {FUND}")

df = df[
    df["close"].gt(0) &
    df["predicted_excess_return_252d"].notna()
].copy()

fund = pd.read_csv(FUND).drop_duplicates("nse_symbol")
df = df.merge(fund, on="nse_symbol", how="left")

# ML percentile
df["ml_score"] = df["predicted_excess_return_252d"].rank(pct=True) * 100

# Independent ROE calculation cannot be done without balance-sheet fields,
# so reported ROE/ROCE are retained and explicitly marked as source values.
fund_cols = [
    "roe_reported", "roce_reported", "pe",
    "sales_growth_5y", "profit_growth_5y",
    "debt_equity", "opm"
]

missing = [c for c in fund_cols if c not in df]
if missing:
    raise ValueError(f"Missing fundamentals columns: {missing}")

df["fundamental_coverage"] = df[fund_cols].notna().mean(axis=1)

score = pd.DataFrame(index=df.index)

for c in [
    "roe_reported",
    "roce_reported",
    "sales_growth_5y",
    "profit_growth_5y",
    "opm"
]:
    score[c] = df[c].rank(pct=True)

score["debt_equity"] = 1 - df["debt_equity"].rank(pct=True)
score["pe"] = 1 - df["pe"].rank(pct=True)

df["fundamental_score"] = score.mean(axis=1, skipna=True) * 100

# Final combined investment score
df["final_score"] = (
    0.70 * df["ml_score"] +
    0.30 * df["fundamental_score"]
)

df["recommendation"] = "AVOID"
df.loc[
    (df["final_score"] >= 70) &
    (df["fundamental_coverage"] >= 0.70),
    "WATCH"
] = "WATCH"
df.loc[
    (df["final_score"] >= 90) &
    (df["fundamental_coverage"] >= 0.70),
    "BUY"
] = "BUY"

df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
df["final_rank"] = np.arange(1, len(df) + 1)

cols = [
    "final_rank", "rank", "isin", "nse_symbol", "company_name",
    "close", "predicted_excess_return_252d", "ml_score",
    "roe_reported", "roce_reported", "pe",
    "sales_growth_5y", "profit_growth_5y",
    "debt_equity", "opm", "fundamental_coverage",
    "fundamental_score", "final_score", "recommendation"
]

result = df[cols]
path = OUT / "latest_recommendations.parquet"
result.to_parquet(path, index=False)

ral = result[result["nse_symbol"].str.upper().eq("RALLIS")]

print("Stocks evaluated:", f"{len(result):,}")
print("\n=== TOP 25 ===")
print(result.head(25).to_string(index=False))

if not ral.empty:
    print("\n=== RALLIS ===")
    print(ral.to_string(index=False))

metadata = {
    "stocks_evaluated": int(len(result)),
    "fundamental_records": int(len(fund)),
    "top_stock": str(result.iloc[0]["nse_symbol"]),
    "top_final_score": float(result.iloc[0]["final_score"]),
    "rallis_present": bool(not ral.empty)
}

(OUT / "phase5_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

print("\nSaved:")
print(path)
print(OUT / "phase5_metadata.json")
print("\n" + "=" * 60)
print("PHASE 5 COMPLETE")
print("=" * 60)
