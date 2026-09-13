from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from src.data.providers.nse_market import load_market_data

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "data/processed/targets/phase3_target.parquet"
OUT = ROOT / "data/processed/model"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHASE 4 — ML MODEL")
print("=" * 60)

# 1. Load Phase 2 + Phase 3
market = load_market_data().copy()
target = pd.read_parquet(TARGET)

market["trade_date"] = pd.to_datetime(market["trade_date"])
target["decision_date"] = pd.to_datetime(target["decision_date"])

if "series" in market.columns:
    market = market[market["series"].eq("EQ")].copy()

market = (
    market.drop_duplicates(["isin", "trade_date"])
    .sort_values(["isin", "trade_date"])
    .reset_index(drop=True)
)

if market.empty:
    raise RuntimeError("Phase 2 EQ market data is empty.")
if target.empty:
    raise RuntimeError("Phase 3 target data is empty.")

print("Market EQ rows:", f"{len(market):,}")
print("Target rows   :", f"{len(target):,}")
print("Latest date  :", market["trade_date"].max().date())

# 2. Historical features
g = market.groupby("isin", sort=False)

for n in [1, 5, 20, 60, 126, 252]:
    market[f"ret_{n}d"] = g["close"].pct_change(n)

for n in [20, 60, 126]:
    market[f"vol_{n}d"] = g["ret_1d"].transform(
        lambda x, n=n: x.rolling(n).std()
    )

market["volume_ratio_20d"] = (
    market["volume"] /
    g["volume"].transform(lambda x: x.rolling(20).mean())
)
market["turnover_ratio_20d"] = (
    market["turnover"] /
    g["turnover"].transform(lambda x: x.rolling(20).mean())
)
market["price_vs_20d"] = (
    market["close"] /
    g["close"].transform(lambda x: x.rolling(20).mean()) - 1
)
market["price_vs_60d"] = (
    market["close"] /
    g["close"].transform(lambda x: x.rolling(60).mean()) - 1
)

FEATURES = [
    "ret_1d", "ret_5d", "ret_20d", "ret_60d", "ret_126d", "ret_252d",
    "vol_20d", "vol_60d", "vol_126d",
    "volume_ratio_20d", "turnover_ratio_20d",
    "price_vs_20d", "price_vs_60d"
]

features = market[["isin", "trade_date"] + FEATURES].rename(
    columns={"trade_date": "decision_date"}
)

# 3. Connect Phase 3 targets + Phase 4 features
model_df = (
    target.merge(
        features,
        on=["isin", "decision_date"],
        how="inner"
    )
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=FEATURES + ["target_excess_return_252d"])
)

if model_df.empty:
    raise RuntimeError("No usable training rows after merge.")

print("Training rows :", f"{len(model_df):,}")
print("Companies     :", f"{model_df['isin'].nunique():,}")
print("Decision dates:", f"{model_df['decision_date'].nunique():,}")

# 4. Time-aware validation
dates = np.sort(model_df["decision_date"].unique())
if len(dates) < 20:
    raise RuntimeError("Not enough historical decision dates for validation.")

split_date = dates[int(len(dates) * 0.80)]

train = model_df[model_df["decision_date"] < split_date]
valid = model_df[model_df["decision_date"] >= split_date]

model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=250,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)

model.fit(train[FEATURES], train["target_excess_return_252d"])
valid_prediction = model.predict(valid[FEATURES])

mae = mean_absolute_error(
    valid["target_excess_return_252d"],
    valid_prediction
)
r2 = r2_score(
    valid["target_excess_return_252d"],
    valid_prediction
)

evaluation = valid[
    ["decision_date", "target_excess_return_252d"]
].copy()
evaluation["prediction"] = valid_prediction

daily_ic = (
    evaluation.groupby("decision_date")
    .apply(
        lambda x: x["prediction"].corr(
            x["target_excess_return_252d"]
        ),
        include_groups=False
    )
    .dropna()
)

mean_ic = float(daily_ic.mean())

print("\n=== MODEL VALIDATION ===")
print("Train rows :", f"{len(train):,}")
print("Valid rows :", f"{len(valid):,}")
print("Split date :", pd.Timestamp(split_date).date())
print("MAE        :", round(mae, 6))
print("R2         :", round(r2, 6))
print("Mean Daily IC:", round(mean_ic, 6))

# 5. Retrain on all historical data
model.fit(
    model_df[FEATURES],
    model_df["target_excess_return_252d"]
)

# 6. Score latest available trading date
latest_date = market["trade_date"].max()

latest = market[
    market["trade_date"].eq(latest_date)
].copy()

latest = (
    latest.replace([np.inf, -np.inf], np.nan)
    .dropna(subset=FEATURES)
)

latest["predicted_excess_return_252d"] = model.predict(
    latest[FEATURES]
)

latest = latest.sort_values(
    "predicted_excess_return_252d",
    ascending=False
)

name_col = (
    "instrument_name"
    if "instrument_name" in latest.columns
    else "company_name"
)

ranking = latest[
    [
        "isin",
        "nse_symbol",
        name_col,
        "close",
        "predicted_excess_return_252d"
    ]
].rename(columns={name_col: "company_name"})

ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

# 7. Save
ranking_path = OUT / "latest_stock_ranking.parquet"
training_path = OUT / "training_dataset.parquet"
metadata_path = OUT / "phase4_metadata.json"

ranking.to_parquet(ranking_path, index=False)
model_df.to_parquet(training_path, index=False)

metadata = {
    "model": "HistGradientBoostingRegressor",
    "training_start": str(model_df["decision_date"].min().date()),
    "training_end": str(model_df["decision_date"].max().date()),
    "inference_date": str(latest_date.date()),
    "target": "target_excess_return_252d",
    "features": FEATURES,
    "validation_mae": float(mae),
    "validation_r2": float(r2),
    "mean_daily_information_coefficient": mean_ic,
    "rows_scored": int(len(ranking))
}

metadata_path.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

# 8. Output
print("\n" + "=" * 60)
print("CURRENT MODEL RANKING")
print("=" * 60)
print("Inference date:", latest_date.date())
print("Stocks scored :", f"{len(ranking):,}")

print("\nTOP 25:")
print(ranking.head(25).to_string(index=False))

ral = ranking[
    ranking["nse_symbol"].str.upper().isin(["RALLIS", "RAL"])
]

if not ral.empty:
    print("\n=== RAL POSITION ===")
    print(ral.to_string(index=False))

print("\nSaved:")
print(ranking_path)
print(training_path)
print(metadata_path)

print("\n" + "=" * 60)
print("PHASE 4 COMPLETE")
print("=" * 60)
