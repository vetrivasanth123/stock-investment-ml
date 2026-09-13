from pathlib import Path
from io import StringIO
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

df["score_pct"] = df["predicted_excess_return_252d"].rank(pct=True) * 100

# Fundamentals: top ML candidates + RALLIS
symbols = list(df.head(250)["nse_symbol"].dropna().unique())
symbols = list(dict.fromkeys(symbols + ["RALLIS"]))

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

def clean(x):
    try:
        return float(
            str(x).replace(",", "").replace("%", "").replace("−", "-").strip()
        )
    except Exception:
        return np.nan

def row(table, name):
    if table is None or table.empty:
        return None
    m = table.iloc[:, 0].astype(str).str.strip().str.lower().eq(name.lower())
    return table.loc[m].iloc[0] if m.any() else None

def values(r):
    if r is None:
        return []
    return pd.to_numeric(
        r.iloc[1:].astype(str).str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False),
        errors="coerce"
    ).dropna().tolist()

def table_with(tables, key):
    for t in tables:
        if not t.empty and key.lower() in t.iloc[:, 0].astype(str).str.lower().tolist():
            return t
    return None

def fetch(symbol):
    try:
        r = session.get(
            f"https://www.screener.in/company/{symbol}/",
            timeout=12
        )
        if r.status_code != 200:
            return {"nse_symbol": symbol}

        html = r.text
        soup_text = re.sub(r"\s+", " ", re.sub("<[^>]+>", " ", html))

        tables = pd.read_html(StringIO(html))

        # Headline reported metrics
        def metric(pattern):
            m = re.search(pattern, soup_text, re.I)
            return clean(m.group(1)) if m else np.nan

        roe = metric(r"\bROE\s+([\d.]+)\s*%")
        roce = metric(r"\bROCE\s+([\d.]+)\s*%")
        pe = metric(r"\bStock P/E\s+([\d.]+)")
        book = metric(r"\bBook Value\s+([\d.]+)")
        dividend = metric(r"\bDividend Yield\s+([\d.]+)\s*%")

        pnl = table_with(tables, "Sales +")
        bal = table_with(tables, "Equity Capital")
        cash = table_with(tables, "Cash from Operating Activity")

        sales = row(pnl, "Sales +")
        profit = row(pnl, "Net Profit +")
        eps = row(pnl, "EPS in Rs")
        opm = row(pnl, "OPM %")
        reserves = row(bal, "Reserves")
        capital = row(bal, "Equity Capital")
        borrowings = row(bal, "Borrowings +")
        cfo = row(cash, "Cash from Operating Activity")
        cfi = row(cash, "Cash from Investing Activity")

        sv = values(sales)
        pv = values(profit)
        ev = values(eps)
        ov = values(opm)
        rv = values(reserves)
        cv = values(capital)
        bv = values(borrowings)
        cfov = values(cfo)
        cfiv = values(cfi)

        sales_cagr = ((sv[-1] / sv[-6]) ** .2 - 1) * 100 if len(sv) >= 6 and sv[-6] > 0 else np.nan
        profit_cagr = ((pv[-1] / pv[-6]) ** .2 - 1) * 100 if len(pv) >= 6 and pv[-6] > 0 else np.nan

        equity_now = rv[-1] + cv[-1] if rv and cv else np.nan
        equity_prev = rv[-2] + cv[-2] if len(rv) >= 2 and len(cv) >= 2 else np.nan
        net_profit = pv[-1] if pv else np.nan

        roe_calc = (
            net_profit / ((equity_now + equity_prev) / 2) * 100
            if np.isfinite(net_profit) and equity_now > 0 and equity_prev > 0
            else np.nan
        )

        debt_equity = (
            bv[-1] / equity_now
            if bv and np.isfinite(equity_now) and equity_now > 0
            else np.nan
        )

        fcf = (
            cfov[-1] + cfiv[-1]
            if cfov and cfiv
            else np.nan
        )

        return {
            "nse_symbol": symbol,
            "roe_reported": roe,
            "roe_calculated": roe_calc,
            "roe_difference": abs(roe - roe_calc) if np.isfinite(roe) and np.isfinite(roe_calc) else np.nan,
            "roce_reported": roce,
            "pe": pe,
            "book_value": book,
            "dividend_yield": dividend,
            "sales_cagr_5y": sales_cagr,
            "profit_cagr_5y": profit_cagr,
            "eps_latest": ev[-1] if ev else np.nan,
            "opm_latest": ov[-1] if ov else np.nan,
            "debt_equity": debt_equity,
            "free_cash_flow": fcf,
        }

    except Exception:
        return {"nse_symbol": symbol}

# Reuse only complete cache rows; retry incomplete ones
old = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame()
good = old[
    old.get("nse_symbol", pd.Series(dtype=str)).isin(symbols) &
    old.get("roe_reported", pd.Series(dtype=float)).notna()
] if not old.empty else pd.DataFrame()

cached = set(good["nse_symbol"]) if not good.empty else set()
new = pd.DataFrame([fetch(s) for s in symbols if s not in cached])

fund = pd.concat([good, new], ignore_index=True).drop_duplicates(
    "nse_symbol", keep="last"
)
fund.to_parquet(CACHE, index=False)

# Merge
df = df.merge(fund, on="nse_symbol", how="left")

fund_cols = [
    "roe_reported", "roce_reported", "sales_cagr_5y",
    "profit_cagr_5y", "eps_latest", "opm_latest",
    "debt_equity", "pe", "free_cash_flow"
]

df["fundamental_coverage"] = df[fund_cols].notna().mean(axis=1)

# Fundamental score from available values
scores = pd.DataFrame(index=df.index)

for c in [
    "roe_reported", "roce_reported",
    "sales_cagr_5y", "profit_cagr_5y",
    "opm_latest", "free_cash_flow"
]:
    scores[c] = df[c].rank(pct=True)

# Lower debt and valuation are preferable
scores["debt_equity"] = 1 - df["debt_equity"].rank(pct=True)
scores["pe"] = 1 - df["pe"].rank(pct=True)

df["fundamental_score"] = (
    scores.mean(axis=1, skipna=True).fillna(0.5) * 100
)

df["final_score"] = (
    0.70 * df["score_pct"] +
    0.30 * df["fundamental_score"]
)

# Missing fundamentals prevent a BUY classification
df["recommendation"] = "AVOID"
df.loc[
    (df["final_score"] >= 70) &
    (df["fundamental_coverage"] >= 0.50),
    "recommendation"
] = "WATCH"
df.loc[
    (df["final_score"] >= 90) &
    (df["fundamental_coverage"] >= 0.70),
    "recommendation"
] = "BUY"

df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
df["final_rank"] = range(1, len(df) + 1)

cols = [
    "final_rank", "rank", "isin", "nse_symbol", "company_name", "close",
    "predicted_excess_return_252d", "score_pct",
    "roe_reported", "roe_calculated", "roe_difference", "roce_reported",
    "pe", "book_value", "dividend_yield",
    "sales_cagr_5y", "profit_cagr_5y", "eps_latest", "opm_latest",
    "debt_equity", "free_cash_flow", "fundamental_coverage",
    "fundamental_score", "final_score", "recommendation"
]

result = df[[c for c in cols if c in df.columns]]

path = OUT / "latest_recommendations.parquet"
result.to_parquet(path, index=False)

ral = result[result["nse_symbol"].str.upper().eq("RALLIS")]

print("Stocks evaluated:", f"{len(result):,}")
print("Fundamental records:", f"{len(fund):,}")

print("\n=== TOP 25 ===")
print(result.head(25).to_string(index=False))

if not ral.empty:
    print("\n=== RALLIS ===")
    print(ral.to_string(index=False))

summary = {
    "stocks_evaluated": int(len(result)),
    "fundamental_records": int(len(fund)),
    "buy_count": int((result.recommendation == "BUY").sum()),
    "watch_count": int((result.recommendation == "WATCH").sum()),
    "avoid_count": int((result.recommendation == "AVOID").sum()),
    "top_stock": str(result.iloc[0].nse_symbol),
    "top_final_score": float(result.iloc[0].final_score),
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
