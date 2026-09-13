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

FUND_COLS = [
    "roe_reported", "roe_calculated", "roe_difference", "roce_reported",
    "pe", "book_value", "dividend_yield", "sales_cagr_5y",
    "profit_cagr_5y", "eps_latest", "opm_latest", "debt_equity",
    "free_cash_flow"
]

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

# Top ML candidates + RALLIS
symbols = list(df.head(250)["nse_symbol"].dropna().unique())
symbols = list(dict.fromkeys(symbols + ["RALLIS"]))

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

def empty_row(symbol):
    return {"nse_symbol": symbol, **{c: np.nan for c in FUND_COLS}}

def clean(x):
    try:
        return float(
            str(x).replace(",", "")
            .replace("%", "")
            .replace("−", "-")
            .strip()
        )
    except Exception:
        return np.nan

def row(table, name):
    if table is None or table.empty:
        return None
    m = table.iloc[:, 0].astype(str).str.strip().str.lower().eq(name.lower())
    return table.loc[m].iloc[0] if m.any() else None

def vals(r):
    if r is None:
        return []
    return pd.to_numeric(
        r.iloc[1:].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False),
        errors="coerce"
    ).dropna().tolist()

def find_table(tables, key):
    key = key.lower()
    for t in tables:
        if not t.empty and key in set(t.iloc[:, 0].astype(str).str.lower()):
            return t
    return None

def fetch(symbol):
    out = empty_row(symbol)

    try:
        r = session.get(
            f"https://www.screener.in/company/{symbol}/",
            timeout=12
        )
        if r.status_code != 200:
            return out

        html = r.text
        text = re.sub(r"\s+", " ", re.sub("<[^>]+>", " ", html))
        tables = pd.read_html(StringIO(html))

        def metric(pattern):
            m = re.search(pattern, text, re.I)
            return clean(m.group(1)) if m else np.nan

        out["roe_reported"] = metric(r"\bROE\s+([\d.]+)\s*%")
        out["roce_reported"] = metric(r"\bROCE\s+([\d.]+)\s*%")
        out["pe"] = metric(r"\bStock P/E\s+([\d.]+)")
        out["book_value"] = metric(r"\bBook Value\s+([\d.]+)")
        out["dividend_yield"] = metric(r"\bDividend Yield\s+([\d.]+)\s*%")

        pnl = find_table(tables, "sales +")
        bal = find_table(tables, "equity capital")
        cash = find_table(tables, "cash from operating activity")

        sales = vals(row(pnl, "Sales +"))
        profit = vals(row(pnl, "Net Profit +"))
        eps = vals(row(pnl, "EPS in Rs"))
        opm = vals(row(pnl, "OPM %"))
        reserves = vals(row(bal, "Reserves"))
        capital = vals(row(bal, "Equity Capital"))
        borrowings = vals(row(bal, "Borrowings +"))
        cfo = vals(row(cash, "Cash from Operating Activity"))
        cfi = vals(row(cash, "Cash from Investing Activity"))

        if len(sales) >= 6 and sales[-6] > 0:
            out["sales_cagr_5y"] = ((sales[-1] / sales[-6]) ** .2 - 1) * 100

        if len(profit) >= 6 and profit[-6] > 0:
            out["profit_cagr_5y"] = ((profit[-1] / profit[-6]) ** .2 - 1) * 100

        if eps:
            out["eps_latest"] = eps[-1]

        if opm:
            out["opm_latest"] = opm[-1]

        if reserves and capital:
            eq_now = reserves[-1] + capital[-1]
            eq_prev = (
                reserves[-2] + capital[-2]
                if len(reserves) >= 2 and len(capital) >= 2
                else np.nan
            )

            if profit and eq_now > 0 and np.isfinite(eq_prev) and eq_prev > 0:
                out["roe_calculated"] = (
                    profit[-1] / ((eq_now + eq_prev) / 2) * 100
                )

            if borrowings and eq_now > 0:
                out["debt_equity"] = borrowings[-1] / eq_now

        if cfo and cfi:
            out["free_cash_flow"] = cfo[-1] + cfi[-1]

        if (
            np.isfinite(out["roe_reported"]) and
            np.isfinite(out["roe_calculated"])
        ):
            out["roe_difference"] = abs(
                out["roe_reported"] - out["roe_calculated"]
            )

    except Exception:
        pass

    return out

# Cache only complete/usable records; incomplete records are retried.
if CACHE.exists():
    old = pd.read_parquet(CACHE)
else:
    old = pd.DataFrame(columns=["nse_symbol"] + FUND_COLS)

for c in ["nse_symbol"] + FUND_COLS:
    if c not in old:
        old[c] = np.nan

old = old[["nse_symbol"] + FUND_COLS]

complete = old[
    old["nse_symbol"].isin(symbols) &
    old[FUND_COLS].notna().any(axis=1)
].drop_duplicates("nse_symbol", keep="last")

cached = set(complete["nse_symbol"])
new = pd.DataFrame([fetch(s) for s in symbols if s not in cached])

fund = pd.concat(
    [complete, new],
    ignore_index=True
).drop_duplicates("nse_symbol", keep="last")

fund = fund[["nse_symbol"] + FUND_COLS]
fund.to_parquet(CACHE, index=False)

# Merge Phase 4 ranking + fundamentals
df = df.merge(fund, on="nse_symbol", how="left")

df["fundamental_coverage"] = df[FUND_COLS].notna().mean(axis=1)

# Cross-sectional fundamental score
scores = pd.DataFrame(index=df.index)

positive = [
    "roe_reported", "roce_reported",
    "sales_cagr_5y", "profit_cagr_5y",
    "opm_latest", "free_cash_flow"
]

for c in positive:
    scores[c] = df[c].rank(pct=True)

scores["debt_equity"] = 1 - df["debt_equity"].rank(pct=True)
scores["pe"] = 1 - df["pe"].rank(pct=True)

df["fundamental_score"] = (
    scores.mean(axis=1, skipna=True).fillna(0.5) * 100
)

# 70% ML + 30% fundamentals
df["final_score"] = (
    0.70 * df["score_pct"] +
    0.30 * df["fundamental_score"]
)

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
df["final_rank"] = np.arange(1, len(df) + 1)

cols = [
    "final_rank", "rank", "isin", "nse_symbol", "company_name", "close",
    "predicted_excess_return_252d", "score_pct",
    "roe_reported", "roe_calculated", "roe_difference", "roce_reported",
    "pe", "book_value", "dividend_yield", "sales_cagr_5y",
    "profit_cagr_5y", "eps_latest", "opm_latest", "debt_equity",
    "free_cash_flow", "fundamental_coverage",
    "fundamental_score", "final_score", "recommendation"
]

result = df[cols]

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
    "fundamental_nonempty": int(fund[FUND_COLS].notna().any(axis=1).sum()),
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
