"""
Retrains all forecasting models (ARIMA, Prophet, XGBoost, LSTM) for
USD/EUR/CNY against TZS, using real backtesting to measure accuracy
instead of just assuming the models work.

KEY DESIGN CHANGE from the original models:
  - XGBoost and LSTM used to forecast RECURSIVELY: each day's guess was fed
    back in as if it were real data, so small errors compounded into
    impossible numbers (this is the bug that caused a -173 TZS prediction).
  - Both are now DIRECT multi-step forecasters: they predict all 7 days at
    once, straight from real historical data, with no recursive feedback.
    This structurally removes the compounding-error problem.

Run this from the project folder that contains the `data/` and `models/`
directories:
    python retrain_models.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

DATA_DIR   = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

CURRENCIES = ["usd", "eur", "cny"]
HORIZON    = 30       # covers the app's longest option (30-day forecast);
                       # 7-day and 14-day views just slice the first N values
SEQ_LEN    = 60        # LSTM lookback window
TEST_DAYS  = 90        # held-out days for backtesting evaluation (needs to be
                       # a few multiples of HORIZON for a meaningful walk-forward test)


def load_series(cur):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{cur}_tzs_clean.csv"))
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------
def train_arima(train_series):
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    # Pick differencing order d via a stationarity test instead of guessing.
    d = 0
    series = train_series.copy()
    for _ in range(2):
        pval = adfuller(series.dropna())[1]
        if pval < 0.05:
            break
        series = series.diff()
        d += 1

    best_model, best_aic = None, np.inf
    for p in range(0, 3):
        for q in range(0, 3):
            try:
                m = ARIMA(train_series, order=(p, d, q)).fit()
                if m.aic < best_aic:
                    best_aic, best_model = m.aic, m
            except Exception:
                continue
    return best_model


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------
def train_prophet(train_df):
    from prophet import Prophet
    pdf = train_df[["date", "mean"]].rename(columns={"date": "ds", "mean": "y"})
    # Lower changepoint_prior_scale than Prophet's default (0.05) so it
    # doesn't overfit a dramatic trend from ordinary day-to-day noise —
    # this is what was causing the unrealistic -9% forecasts.
    m = Prophet(changepoint_prior_scale=0.01, daily_seasonality=False)
    m.fit(pdf)
    return m


# ---------------------------------------------------------------------------
# XGBoost — DIRECT multi-step (no recursion)
# ---------------------------------------------------------------------------
def make_features(df, i):
    """Build lag/rolling features using only real data up to index i."""
    vals = df["mean"].values
    l1  = vals[i]
    l3  = vals[i-2]  if i >= 2  else l1
    l7  = vals[i-6]  if i >= 6  else l1
    l14 = vals[i-13] if i >= 13 else l1
    l30 = vals[i-29] if i >= 29 else l1
    ma7  = vals[max(0, i-6):i+1].mean()
    ma30 = vals[max(0, i-29):i+1].mean()
    std7 = vals[max(0, i-6):i+1].std()
    d = df["date"].iloc[i]
    return [l1, l3, l7, l14, l30, ma7, ma30, std7,
            d.month, d.weekday(), (d.month - 1) // 3 + 1]


def build_direct_dataset(df, horizon):
    """For each valid day i, X = features at i, y = [value at i+1, ..., i+horizon]."""
    X, Y = [], []
    vals = df["mean"].values
    n = len(df)
    for i in range(29, n - horizon):   # need 30 days history + `horizon` days ahead
        X.append(make_features(df, i))
        Y.append(vals[i+1:i+1+horizon])
    return np.array(X), np.array(Y)


def train_xgboost_direct(train_df, horizon):
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor
    X, Y = build_direct_dataset(train_df, horizon)
    model = MultiOutputRegressor(
        XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=42)
    )
    model.fit(X, Y)
    return model


def predict_xgboost_direct(model, df, horizon):
    feat = np.array([make_features(df, len(df) - 1)])
    return model.predict(feat)[0]


# ---------------------------------------------------------------------------
# LSTM — DIRECT multi-step (no recursion)
# ---------------------------------------------------------------------------
def build_lstm_dataset(values, seq_len, horizon):
    X, Y = [], []
    for i in range(seq_len, len(values) - horizon):
        X.append(values[i-seq_len:i])
        Y.append(values[i:i+horizon])
    return np.array(X), np.array(Y)


def train_lstm_direct(train_series, seq_len, horizon):
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train_series.values.reshape(-1, 1)).flatten()
    X, Y = build_lstm_dataset(scaled, seq_len, horizon)
    X = X.reshape((X.shape[0], seq_len, 1))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, 1)),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(horizon)   # predicts all `horizon` days at once
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=40, batch_size=16, verbose=0)
    return model, scaler


def predict_lstm_direct(model, scaler, series, seq_len):
    scaled = scaler.transform(series.values[-seq_len:].reshape(-1, 1)).flatten()
    X = scaled.reshape(1, seq_len, 1)
    pred_scaled = model.predict(X, verbose=0)[0]
    return scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()


# ---------------------------------------------------------------------------
# Backtest harness — walk-forward evaluation on real held-out data
# ---------------------------------------------------------------------------
def mape_at_checkpoints(preds_list, actuals_list, checkpoints=(7, 14, 30)):
    """preds_list/actuals_list: list of (HORIZON,)-shaped arrays, one per
    walk-forward window. Returns MAPE at each checkpoint day across all
    windows, so we can see how accuracy degrades further into the future."""
    out = {}
    for cp in checkpoints:
        preds   = [p[cp-1] for p in preds_list if len(p) >= cp]
        actuals = [a[cp-1] for a in actuals_list if len(a) >= cp]
        if preds:
            out[f"day_{cp}"] = mape(actuals, preds)
    return out


def backtest(cur, df, skip_prophet=False):
    n = len(df)
    split = n - TEST_DAYS
    train_df = df.iloc[:split].reset_index(drop=True)
    results = {}

    print(f"\n=== {cur.upper()} — training on {split} days, backtesting on last {TEST_DAYS} days ===")

    # ---- ARIMA ----
    arima_model = train_arima(train_df["mean"])
    arima_fc = arima_model.forecast(steps=TEST_DAYS).values
    actual_all = df["mean"].iloc[split:split+TEST_DAYS].values
    results["ARIMA"] = mape_at_checkpoints([arima_fc], [actual_all])

    # ---- Prophet ----
    if not skip_prophet:
        prophet_model = train_prophet(train_df)
        future = prophet_model.make_future_dataframe(periods=TEST_DAYS)
        fc = prophet_model.predict(future)["yhat"].tail(TEST_DAYS).values
        results["Prophet"] = mape_at_checkpoints([fc], [actual_all])

    # ---- XGBoost (direct, walk-forward across multiple windows) ----
    xgb_model = train_xgboost_direct(train_df, HORIZON)
    xgb_preds_list, xgb_actuals_list = [], []
    for start in range(split, n - HORIZON, HORIZON):
        window_df = df.iloc[:start].reset_index(drop=True)
        p = predict_xgboost_direct(xgb_model, window_df, HORIZON)
        a = df["mean"].iloc[start:start+HORIZON].values
        xgb_preds_list.append(p)
        xgb_actuals_list.append(a)
    results["XGBoost"] = mape_at_checkpoints(xgb_preds_list, xgb_actuals_list)

    # ---- LSTM (direct, walk-forward across multiple windows) ----
    lstm_model, scaler = train_lstm_direct(train_df["mean"], SEQ_LEN, HORIZON)
    lstm_preds_list, lstm_actuals_list = [], []
    for start in range(split, n - HORIZON, HORIZON):
        window_series = df["mean"].iloc[:start].reset_index(drop=True)
        p = predict_lstm_direct(lstm_model, scaler, window_series, SEQ_LEN)
        a = df["mean"].iloc[start:start+HORIZON].values
        lstm_preds_list.append(p)
        lstm_actuals_list.append(a)
    results["LSTM"] = mape_at_checkpoints(lstm_preds_list, lstm_actuals_list)

    print(f"Backtest MAPE by horizon (lower = better; % avg error vs real prices):")
    print(f"  {'Model':10s} {'day 7':>8s} {'day 14':>8s} {'day 30':>8s}")
    for name, cps in results.items():
        print(f"  {name:10s} {cps.get('day_7', float('nan')):7.3f}% "
              f"{cps.get('day_14', float('nan')):7.3f}% "
              f"{cps.get('day_30', float('nan')):7.3f}%")

    return results


def train_final_and_save(cur, df, skip_prophet=False):
    """Train on ALL available data (not just the train split) for the
    models that actually get shipped, since backtesting already told us
    how accurate each one is."""
    print(f"\nTraining final {cur.upper()} models on full dataset...")

    arima_model = train_arima(df["mean"])
    joblib.dump(arima_model, os.path.join(MODELS_DIR, f"arima_{cur}_model.pkl"))

    if not skip_prophet:
        prophet_model = train_prophet(df)
        joblib.dump(prophet_model, os.path.join(MODELS_DIR, f"prophet_{cur}_model.pkl"))

    xgb_model = train_xgboost_direct(df, HORIZON)
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, f"xgboost_{cur}_model_direct.pkl"))

    lstm_model, scaler = train_lstm_direct(df["mean"], SEQ_LEN, HORIZON)
    lstm_model.save(os.path.join(MODELS_DIR, f"lstm_{cur}_model.h5"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"scaler_{cur}.pkl"))

    print(f"Saved models for {cur.upper()} to {MODELS_DIR}/")


if __name__ == "__main__":
    import sys
    skip_prophet = "--skip-prophet" in sys.argv  # pass this flag if Prophet/Stan isn't set up yet

    all_results = {}
    for cur in CURRENCIES:
        df = load_series(cur)
        all_results[cur] = backtest(cur, df, skip_prophet=skip_prophet)

    print("\n" + "="*60)
    print("SUMMARY — backtest MAPE (%) by horizon, per model per currency")
    print("="*60)
    rows = []
    for cur, model_results in all_results.items():
        for model_name, cps in model_results.items():
            rows.append({"currency": cur, "model": model_name, **cps})
    summary_df = pd.DataFrame(rows).set_index(["currency", "model"])
    print(summary_df.round(3))
    print("\nModels with high MAPE (e.g. >2-3%) for a stable currency like")
    print("this should be treated with suspicion before shipping.")
    print("Also expect day_30 error to be somewhat higher than day_7 — that's")
    print("normal, just make sure it isn't dramatically worse.")

    proceed = input("\nProceed to train final models on full data and save? [y/N] ").strip().lower()
    if proceed == "y":
        for cur in CURRENCIES:
            df = load_series(cur)
            train_final_and_save(cur, df, skip_prophet=skip_prophet)
        print("\nDone. New model files are in the models/ folder.")
    else:
        print("Skipped saving. Re-run and answer 'y' when ready.")