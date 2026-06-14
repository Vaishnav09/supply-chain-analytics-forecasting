"""
Generate forward demand forecasts using the persisted XGBoost model.
Uses recursive multi-step prediction — each future week's prediction
feeds into the lag features of the next week.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from scipy import stats

ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

SERVICE_LEVEL_Z = {
    0.90: 1.282,
    0.95: 1.645,
    0.99: 2.326,
}


def load_artifacts():
    model    = joblib.load(MODEL_DIR / "xgb_demand.pkl")
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
    series   = pd.read_csv(MODEL_DIR / "training_series.csv", parse_dates=["ds"])
    return model, metadata, series


def recursive_forecast(
    model,
    history: pd.Series,
    feature_cols: list,
    lags: list,
    n_weeks: int,
    series_start_index: int,
) -> np.ndarray:
    """
    Recursively predict n_weeks ahead.
    Each prediction is appended to history before generating the next step's features.
    """
    known = list(history.values)

    for step in range(n_weeks):
        row = {}
        for lag in lags:
            idx = len(known) - lag
            row[f"lag_{lag}"] = known[idx] if idx >= 0 else np.nan

        recent = pd.Series(known)
        row["rolling_4"]  = recent.shift(1).rolling(4).mean().iloc[-1] if len(known) >= 4 else np.nan
        row["rolling_12"] = recent.shift(1).rolling(12).mean().iloc[-1] if len(known) >= 12 else np.nan
        row["week_of_year"] = series_start_index + len(known)

        X_row  = pd.DataFrame([row])[feature_cols]
        pred   = float(model.predict(X_row)[0])
        known.append(max(pred, 0))

    return np.array(known[-n_weeks:])


def generate_forecast(n_weeks: int = 12, service_level: float = 0.95) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        week, forecast, lower_bound, upper_bound,
        safety_stock, recommended_order
    """
    model, metadata, series = load_artifacts()

    feature_cols = metadata["feature_cols"]
    lags         = metadata["lags"]
    rmse         = metadata["holdout_metrics"]["rmse"]
    z            = SERVICE_LEVEL_Z.get(service_level, 1.645)

    last_date   = series["ds"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=n_weeks,
        freq="W-MON",
    )

    preds = recursive_forecast(
        model         = model,
        history       = series["y"],
        feature_cols  = feature_cols,
        lags          = lags,
        n_weeks       = n_weeks,
        series_start_index = len(series),
    )

    safety_stock     = z * rmse
    lower            = np.maximum(preds - z * rmse, 0)
    upper            = preds + z * rmse
    recommended_order = np.ceil(preds + safety_stock).astype(int)

    return pd.DataFrame({
        "week":              future_dates,
        "forecast":          np.round(preds).astype(int),
        "lower_bound":       np.round(lower).astype(int),
        "upper_bound":       np.round(upper).astype(int),
        "safety_stock":      int(round(safety_stock)),
        "recommended_order": recommended_order,
    })


def get_recent_accuracy(n_weeks: int = 12) -> pd.DataFrame:
    """
    Returns last n_weeks of actual vs model prediction (from holdout).
    Used in the production app to show recent tracking performance.
    """
    model, metadata, series = load_artifacts()

    feature_cols = metadata["feature_cols"]
    lags         = metadata["lags"]

    test  = series.iloc[-n_weeks:].copy()
    train = series.iloc[:-n_weeks].copy()

    preds = recursive_forecast(
        model              = model,
        history            = train["y"],
        feature_cols       = feature_cols,
        lags               = lags,
        n_weeks            = n_weeks,
        series_start_index = len(train),
    )

    residuals = test["y"].values - preds
    return pd.DataFrame({
        "week":      test["ds"].values,
        "actual":    test["y"].values.astype(int),
        "forecast":  np.round(preds).astype(int),
        "error":     np.round(residuals).astype(int),
        "error_pct": np.round(residuals / test["y"].values * 100, 1),
    })
