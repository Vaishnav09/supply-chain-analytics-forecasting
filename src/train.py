"""
Train XGBoost demand forecasting model on full clean dataset and persist to disk.
Run this script whenever new data is available to retrain the model.

Usage:
    python src/train.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime, timezone
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

ROOT      = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "clean_supply_chain.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

LAGS          = [1, 2, 4, 12, 52]
TEST_WEEKS    = 12
ARTIFACT_DATE = "2017-10-01"


def build_weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(df["order_date"].dt.to_period("W").dt.start_time)["order_item_quantity"]
        .sum()
        .reset_index()
        .rename(columns={"order_date": "ds", "order_item_quantity": "y"})
        .sort_values("ds")
        .reset_index(drop=True)
    )


def build_features(series: pd.Series, lags: list = LAGS, start_index: int = 0) -> pd.DataFrame:
    s = pd.DataFrame({"y": series.values})
    for lag in lags:
        s[f"lag_{lag}"] = s["y"].shift(lag)
    s["rolling_4"]    = s["y"].shift(1).rolling(4).mean()
    s["rolling_12"]   = s["y"].shift(1).rolling(12).mean()
    s["week_of_year"] = range(start_index, start_index + len(s))
    return s


def smape(a, p):
    return 100 * np.mean(2 * np.abs(a - p) / (np.abs(a) + np.abs(p)))


def evaluate_holdout(weekly: pd.DataFrame) -> dict:
    """Walk-forward holdout on last TEST_WEEKS to produce honest eval metrics."""
    train = weekly.iloc[:-TEST_WEEKS].copy()
    test  = weekly.iloc[-TEST_WEEKS:].copy()
    actual = test["y"].values

    all_y   = pd.concat([train["y"], test["y"]]).reset_index(drop=True)
    feat_df = build_features(all_y).dropna()
    usable  = len(train) - (len(all_y) - len(feat_df))
    fcols   = [c for c in feat_df.columns if c != "y"]

    eval_model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    eval_model.fit(feat_df[fcols].iloc[:usable], feat_df["y"].iloc[:usable], verbose=False)
    preds = eval_model.predict(feat_df[fcols].iloc[usable:])

    residuals = actual[-len(preds):] - preds
    return {
        "mae":   float(round(mean_absolute_error(actual[-len(preds):], preds), 2)),
        "rmse":  float(round(np.sqrt(mean_squared_error(actual[-len(preds):], preds)), 2)),
        "smape": float(round(smape(actual[-len(preds):], preds), 3)),
        "bias":  float(round(residuals.mean(), 2)),
        "residual_std": float(round(residuals.std(), 2)),
        "test_weeks": TEST_WEEKS,
    }


def train_production_model(weekly: pd.DataFrame):
    """Train on full dataset — more data = better production forecasts."""
    feat_df = build_features(weekly["y"]).dropna()
    fcols   = [c for c in feat_df.columns if c != "y"]

    model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(feat_df[fcols], feat_df["y"], verbose=False)
    return model, fcols


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["order_date"] = pd.to_datetime(df["order_date"])

    weekly = build_weekly_series(df)
    weekly = weekly[weekly["ds"] < ARTIFACT_DATE].copy().reset_index(drop=True)
    print(f"  {len(weekly)} clean weeks · {weekly['ds'].iloc[0].date()} → {weekly['ds'].iloc[-1].date()}")

    print("Evaluating on holdout...")
    metrics = evaluate_holdout(weekly)
    print(f"  MAE={metrics['mae']} · RMSE={metrics['rmse']} · SMAPE={metrics['smape']}%")

    print("Training production model on full dataset...")
    model, feature_cols = train_production_model(weekly)

    metadata = {
        "trained_at":       datetime.now(timezone.utc).isoformat(),
        "training_weeks":   len(weekly),
        "last_data_date":   str(weekly["ds"].iloc[-1].date()),
        "lags":             LAGS,
        "feature_cols":     feature_cols,
        "holdout_metrics":  metrics,
        "avg_weekly_demand": float(round(weekly["y"].mean(), 1)),
        "demand_std":        float(round(weekly["y"].std(), 1)),
    }

    joblib.dump(model, MODEL_DIR / "xgb_demand.pkl")
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    weekly.to_csv(MODEL_DIR / "training_series.csv", index=False)

    print(f"Saved model  → models/xgb_demand.pkl")
    print(f"Saved meta   → models/metadata.json")
    print(f"Saved series → models/training_series.csv")
    print("Done.")


if __name__ == "__main__":
    main()
