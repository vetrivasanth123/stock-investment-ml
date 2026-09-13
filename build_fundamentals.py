from pathlib import Path
from io import StringIO
import re
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
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
wanted = set(rank.head(100)["company_name"].str.strip()) | {"RALLIS INDIA LTD"}

def norm(x):
    return re.sub(r"[^A-Z0-9]", "", str(x).upper())

wanted_n = {norm(x) for x in wanted}
rows = []

# Screener public screen: 50/page
for page in range(1, 200):
    url = BASE if page == 1 else f"{BASE}?page={page}"
    r = S.get(url, timeout=15)
    r.raise_for_status()

    tables = pd.read_html(StringIO(r.text))
    if not tables:
        break

    t = max(tables, key=lambda x: len(x))
    if "Name" not in t.columns:
        break

    t["match"] = t["Name"].map(norm).isin(wanted_n)
    hit = t[t["match"]].copy()

    if not hit.empty:
        rows.append(hit)

    if len(pd.concat(rows, ignore_index=True)["Name"].map(norm).unique()) >= len(wanted_n):
        break

fund = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

if fund.empty:
    raise RuntimeError("Screener fundamentals table returned no matching companies.")

fund = fund.drop(columns="match", errors="ignore")
fund = fund.drop_duplicates("Name")

rename = {
    "Name": "company_name",
    "P/E": "pe",
    "Div Yld %": "dividend_yield",
    "Qtr Profit Var %": "profit_growth_qtr",
    "Qtr Sales Var %": "sales_growth_qtr",
    "ROCE %": "roce_reported",
    "ROE %": "roe_reported",
    "B.V. Rs.": "book_value"
}
fund = fund.rename(columns=rename)

# Match back to the project's NSE symbols/company names
fund["match"] = fund["company_name"].map(norm)
rank["match"] = rank["company_name"].map(norm)

cols = [
    "company_name", "pe", "dividend_yield",
    "profit_growth_qtr", "sales_growth_qtr",
    "roce_reported", "roe_reported", "book_value"
]
fund = fund[[c for c in cols if c in fund.columns] + ["match"]]

result = rank[["isin", "nse_symbol", "company_name"]].merge(
    fund.drop_duplicates("match"),
    on="match",
    how="inner"
).drop(columns="match")

result.to_parquet(OUT, index=False)

print("=" * 60)
print("FUNDAMENTALS ACQUISITION COMPLETE")
print("=" * 60)
print("Candidates requested:", len(wanted))
print("Fundamentals found  :", len(result))
print("Saved               :", OUT)
print("\nRALLIS:")
print(result[result["nse_symbol"].eq("RALLIS")].to_string(index=False))
