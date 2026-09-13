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

# -------------------- LOAD DATA --------------------
df = pd.read_parquet(IN)

if df.empty:
    raise RuntimeError("Phase 4 ranking is empty.")

df = df[
    df["close"].gt(0) &
    df["predicted_excess_return_252d"].notna()
].copy()

if FUND.exists():
    fund = pd.read_csv(FUND).drop_duplicates("nse_symbol")
    df = df.merge(fund, on="nse_symbol", how="left")
else:
    fund = pd.DataFrame()
    print(f"Warning: fundamentals file not found:\n{FUND}")

# -------------------- FUNDAMENTALS --------------------
FUND_COLS = [
    "roe_reported", "roce_reported", "pe",
    "sales_growth_5y", "profit_growth_5y",
    "debt_equity", "opm"
]

for col in FUND_COLS:
    if col not in df:
        df[col] = np.nan

df["fundamental_coverage"] = df[FUND_COLS].notna().mean(axis=1)

# -------------------- SCORES --------------------
df["ml_score"] = (
    df["predicted_excess_return_252d"]
    .rank(pct=True)
    .mul(100)
)

higher_better = [
    "roe_reported",
    "roce_reported",
    "sales_growth_5y",
    "profit_growth_5y",
    "opm"
]

scores = pd.DataFrame(index=df.index)

for col in higher_better:
    scores[col] = df[col].rank(pct=True)

scores["debt_equity"] = 1 - df["debt_equity"].rank(pct=True)
scores["pe"] = 1 - df["pe"].rank(pct=True)

df["fundamental_score"] = scores.mean(axis=1, skipna=True).mul(100)

# Use 70:30 only when >=70% fundamentals are available
has_fundamentals = df["fundamental_coverage"] >= 0.70

df["final_score"] = np.where(
    has_fundamentals,
    0.70 * df["ml_score"] + 0.30 * df["fundamental_score"],
    df["ml_score"]
)

# -------------------- RECOMMENDATION --------------------
df["recommendation"] = np.select(
    [
        has_fundamentals & df["final_score"].ge(90),
        df["final_score"].ge(70)
    ],
    [
        "BUY",
        "WATCH"
    ],
    default="AVOID"
)

# -------------------- FINAL RANK --------------------
df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
df["final_rank"] = np.arange(1, len(df) + 1)

cols = [
    "final_rank", "rank", "isin", "nse_symbol", "company_name",
    "close", "predicted_excess_return_252d", "ml_score",
    "roe_reported", "roce_reported", "pe",
    "sales_growth_5y", "profit_growth_5y",
    "debt_equity", "opm",
    "fundamental_coverage", "fundamental_score",
    "final_score", "recommendation"
]

result = df[cols]

# -------------------- SAVE --------------------
ranking_path = OUT / "latest_recommendations.parquet"
metadata_path = OUT / "phase5_metadata.json"

result.to_parquet(ranking_path, index=False)

rallis = result[
    result["nse_symbol"].str.upper().eq("RALLIS")
]

metadata = {
    "stocks_evaluated": int(len(result)),
    "fundamental_records": int(len(fund)),
    "stocks_with_70pct_fundamentals": int(has_fundamentals.sum()),
    "top_stock": str(result.iloc[0]["nse_symbol"]),
    "top_final_score": float(result.iloc[0]["final_score"]),
    "rallis_present": bool(not rallis.empty),
    "fundamental_weight": 0.30,
    "ml_weight": 0.70,
    "fundamental_threshold": 0.70
}

metadata_path.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

# -------------------- OUTPUT --------------------
print(f"Stocks evaluated: {len(result):,}")
print(
    f"Stocks with >=70% fundamentals: "
    f"{has_fundamentals.sum():,}"
)

print("\n=== TOP 25 ===")
print(result.head(25).to_string(index=False))

if not rallis.empty:
    print("\n=== RALLIS ===")
    print(rallis.to_string(index=False))

print("\nSaved:")
print(ranking_path)
print(metadata_path)

print("\n" + "=" * 60)
print("PHASE 5 COMPLETE")
print("=" * 60)
