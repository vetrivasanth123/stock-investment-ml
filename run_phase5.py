from pathlib import Path
import json
import re
import requests
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
IN = ROOT / "data/processed/model/latest_stock_ranking.parquet"
OUT = ROOT / "data/processed/recommendations"
CACHE = OUT / "fundamentals.parquet"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 5 — INVESTMENT RECOMMENDATION + FUNDAMENTALS")
print("=" * 60)

df = pd.read_parquet(IN)
if df.empty:
    raise RuntimeError("Phase 4 ranking is empty.")

df = df[
    df["close"].gt(0) &
    df["predicted_excess_return_252d"].notna()
].copy()

df["score_pct"] = (
    df["predicted_excess_return_252d"]
    .rank(pct=True) * 100
)

# Fundamental check for strongest ML candidates + RALLIS
symbols = list(df.head(250)["nse_symbol"].dropna().unique())
symbols.append("RALLIS")
symbols = list(dict.fromkeys(symbols))

headers = {"User-Agent": "Mozilla/5.0"}
session = requests.Session()
session.headers.update(headers)

def num(x):
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except Exception:
        return np.nan

def get_row(table, label):
    if table is None:
        return None
    for _, r in table.iterrows():
        if str(r.iloc[0]).strip().lower() == label.lower():
            return r
    return None

def fetch_fundamental(symbol):
    try:
        url = f"https://www.screener.in/company/{symbol}/"
        html = session.get(url, timeout=10).text
        tables = pd.read_html(html)

        # Reported headline ratios
        text = " ".join(tables[0].astype(str).fillna("").values.flatten())
        roe = re.search(r"ROE\s+([\d.]+)\s*%", html)
        roce = re.search(r"ROCE\s+([\d.]+)\s*%", html)
        pe = re.search(r"Stock P/E\s+([\d.]+)", html)

        reported_roe = num(roe.group(1)) if roe else np.nan
        reported_roce = num(roce.group(1)) if roce else np.nan
        pe_value = num(pe.group(1)) if pe else np.nan

        annual = next(
            (t for t in tables if "Sales +" in t.iloc[:, 0].astype(str).tolist()),
            None
        )

        calc_roe = np.nan
        sales_growth_5y = np.nan
        profit_growth_5y = np.nan

        if annual is not None and annual.shape[1] >= 7:
            labels = annual.iloc[:, 0].astype(str).str.strip()
            sales = get_row(annual, "Sales +")
            profit = get_row(annual, "Net Profit +")

            if sales is not None:
                vals = pd.to_numeric(
                    sales.iloc[1:].astype(str).str.replace(",", "", regex=False),
                    errors="coerce"
                ).dropna()

                if len(vals) >= 6 and vals.iloc[-6] > 0:
                    sales_growth_5y = (
                        (vals.iloc[-1] / vals.iloc[-6]) ** (1 / 5) - 1
                    ) * 100

            if profit is not None:
                vals = pd.to_numeric(
                    profit.iloc[1:].astype(str).str.replace(",", "", regex=False),
                    errors="coerce"
                ).dropna()

                if len(vals) >= 6 and vals.iloc[-6] > 0:
                    profit_growth_5y = (
                        (vals.iloc[-1] / vals.iloc[-6]) ** (1 / 5) - 1
                    ) * 100

        # Balance-sheet based ROE check
        balance = next(
            (t for t in tables if "Equity Capital" in t.iloc[:, 0].astype(str).tolist()),
            None
        )

        if balance is not None:
            net = get_row(annual, "Net Profit +")
            equity = get_row(balance, "Reserves")
            capital = get_row(balance, "Equity Capital")

            if net is not None and equity is not None and capital is not None:
                n = num(net.iloc[-1])
                e1 = num(equity.iloc[-2]) + num(capital.iloc[-2])
                e2 = num(equity.iloc[-1]) + num(capital.iloc[-1])
                if np.isfinite(n) and e1 > 0 and e2 > 0:
                    calc_roe = n / ((e1 + e2) / 2) * 100

        return {
            "nse_symbol": symbol,
            "roe_reported": reported_roe,
            "roe_calculated": calc_roe,
            "roe_difference": (
                abs(reported_roe - calc_roe)
                if np.isfinite(reported_roe) and np.isfinite(calc_roe)
                else np.nan
            ),
            "roce_reported": reported_roce,
            "pe": pe_value,
            "sales_cagr_5y": sales_growth_5y,
            "profit_cagr_5y": profit_growth_5y,
        }

    except Exception:
        return {"nse_symbol": symbol}

# Cache so later runs do not repeatedly download fundamentals
if CACHE.exists():
    fundamentals = pd.read_parquet(CACHE)
    cached = set(fundamentals["nse_symbol"])
else:
    fundamentals = pd.DataFrame()
    cached = set()

rows = [
    fetch_fundamental(s)
    for s in symbols
    if s not in cached
]

if rows:
    fundamentals = pd.concat(
        [fundamentals, pd.DataFrame(rows)],
        ignore_index=True
    )

fundamentals.to_parquet(CACHE, index=False)

# Merge fundamentals into ML ranking
df = df.merge(fundamentals, on="nse_symbol", how="left")

# Fundamental score: quality + growth, without overriding ML completely
df["fundamental_score"] = (
    df["roe_reported"].rank(pct=True).fillna(0.5) * 0.30 +
    df["roce_reported"].rank(pct=True).fillna(0.5) * 0.30 +
    df["sales_cagr_5y"].rank(pct=True).fillna(0.5) * 0.15 +
    df["profit_cagr_5y"].rank(pct=True).fillna(0.5) * 0.25
) * 100

df["final_score"] = (
    0.70 * df["score_pct"] +
    0.30 * df["fundamental_score"]
)

df["recommendation"] = "AVOID"
df.loc[df["final_score"] >= 70, "recommendation"] = "WATCH"
df.loc[df["final_score"] >= 90, "recommendation"] = "BUY"

df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
df["final_rank"] = range(1, len(df) + 1)

cols = [
    "final_rank", "rank", "isin", "nse_symbol", "company_name",
    "close", "predicted_excess_return_252d", "score_pct",
    "roe_reported", "roe_calculated", "roe_difference",
    "roce_reported", "pe", "sales_cagr_5y", "profit_cagr_5y",
    "fundamental_score", "final_score", "recommendation"
]

result = df[[c for c in cols if c in df.columns]]

path = OUT / "latest_recommendations.parquet"
result.to_parquet(path, index=False)

ral = result[result["nse_symbol"].str.upper().eq("RALLIS")]

print("Stocks evaluated:", f"{len(result):,}")
print("\n=== TOP 25 ===")
print(result.head(25).to_string(index=False))

if not ral.empty:
    print("\n=== RALLIS ===")
    print(ral.to_string(index=False))

summary = {
    "stocks_evaluated": int(len(result)),
    "fundamental_companies": int(fundamentals["nse_symbol"].nunique()),
    "buy_count": int((result["recommendation"] == "BUY").sum()),
    "watch_count": int((result["recommendation"] == "WATCH").sum()),
    "avoid_count": int((result["recommendation"] == "AVOID").sum()),
    "top_stock": str(result.iloc[0]["nse_symbol"]),
    "top_final_score": float(result.iloc[0]["final_score"])
}

(OUT / "phase5_metadata.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\nSaved:")
print(path)
print(CACHE)
print(OUT / "phase5_metadata.json")

print("\n" + "=" * 60)
print("PHASE 5 COMPLETE")
print("=" * 60)
