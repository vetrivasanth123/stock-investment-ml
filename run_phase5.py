from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
IN = ROOT / "data/processed/model/latest_stock_ranking.parquet"
OUT = ROOT / "data/processed/recommendations"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 5 — INVESTMENT RECOMMENDATION")
print("=" * 60)

df = pd.read_parquet(IN)

if df.empty:
    raise RuntimeError("Phase 4 ranking is empty.")

df = df[
    df["close"].gt(0) &
    df["predicted_excess_return_252d"].notna()
].copy()

# Higher predicted excess return = stronger signal
df["score_pct"] = (
    df["predicted_excess_return_252d"]
    .rank(pct=True, method="average") * 100
)

df["recommendation"] = "AVOID"
df.loc[df["score_pct"] >= 70, "recommendation"] = "WATCH"
df.loc[df["score_pct"] >= 90, "recommendation"] = "BUY"

# Sort by actual model strength
df = df.sort_values(
    "predicted_excess_return_252d",
    ascending=False
).reset_index(drop=True)

df["final_rank"] = range(1, len(df) + 1)

result = df[
    [
        "final_rank",
        "rank",
        "isin",
        "nse_symbol",
        "company_name",
        "close",
        "predicted_excess_return_252d",
        "score_pct",
        "recommendation",
    ]
]

path = OUT / "latest_recommendations.parquet"
result.to_parquet(path, index=False)

ral = result[
    result["nse_symbol"].str.upper().isin(["RALLIS", "RAL"])
]

print("Stocks evaluated:", f"{len(result):,}")

print("\n=== TOP 25 ===")
print(result.head(25).to_string(index=False))

if not ral.empty:
    print("\n=== RALLIS POSITION ===")
    print(ral.to_string(index=False))

summary = {
    "stocks_evaluated": int(len(result)),
    "buy_count": int((result["recommendation"] == "BUY").sum()),
    "watch_count": int((result["recommendation"] == "WATCH").sum()),
    "avoid_count": int((result["recommendation"] == "AVOID").sum()),
    "top_stock": str(result.iloc[0]["nse_symbol"]),
    "top_prediction": float(result.iloc[0]["predicted_excess_return_252d"]),
}

(OUT / "phase5_metadata.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\nSaved:")
print(path)
print(OUT / "phase5_metadata.json")

print("\n" + "=" * 60)
print("PHASE 5 COMPLETE")
print("=" * 60)
