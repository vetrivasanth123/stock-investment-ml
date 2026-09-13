from pathlib import Path
from io import StringIO
import re
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent
RANKING = ROOT / "data/processed/model/latest_stock_ranking.parquet"
OUT = ROOT / "data/processed/recommendations/fundamentals.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

BASE = "https://www.screener.in/screens/480643/fundamentals/"
S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.screener.in/"
})

rank = pd.read_parquet(RANKING)
wanted = set(rank.head(100)["company_name"].dropna())
wanted.add("RALLIS INDIA LTD")

def norm(x):
    return re.sub(r"[^A-Z0-9]", "", str(x).upper())

wanted_n = {norm(x) for x in wanted}
rows = []

for page in range(1, 200):
    url = BASE if page == 1 else f"{BASE}?page={page}"
    r = S.get(url, timeout=15)
    r.raise_for_status()

    tables = pd.read_html(StringIO(r.text))
    if not tables:
        break

    t = max(tables, key=len)
    if "Name" not in t.columns:
        break

    hit = t[t["Name"].map(norm).isin(wanted_n)]
    if not hit.empty:
        rows.append(hit)

    found = set(pd.concat(rows)["Name"].map(norm)) if rows else set()
    if wanted_n <= found:
        break

fund = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

if fund.empty:
    raise RuntimeError("No matching fundamentals found from Screener.")

fund = fund.drop_duplicates("Name")
fund = fund.rename(columns={
    "Name": "company_name",
    "P/E": "pe",
    "Div Yld %": "dividend_yield",
    "Qtr Profit Var %": "profit_growth_qtr",
    "Qtr Sales Var %": "sales_growth_qtr",
    "ROCE %": "roce_reported",
    "ROE %": "roe_reported",
    "B.V. Rs.": "book_value"
})

fund["match"] = fund["company_name"].map(norm)
rank["match"] = rank["company_name"].map(norm)

cols = [
    "company_name", "pe", "dividend_yield",
    "profit_growth_qtr", "sales_growth_qtr",
    "roce_reported", "roe_reported", "book_value"
]

result = rank[
    ["isin", "nse_symbol", "company_name", "match"]
].merge(
    fund[["match"] + [c for c in cols if c in fund.columns]],
    on="match",
    how="inner"
).drop(columns="match")

result.to_parquet(OUT, index=False)

print("=" * 60)
print("FUNDAMENTALS ACQUISITION COMPLETE")
print("=" * 60)
print("Requested:", len(wanted))
print("Found    :", len(result))
print("Saved    :", OUT)
print("\nRALLIS:")
print(result[result["nse_symbol"].eq("RALLIS")].to_string(index=False))
