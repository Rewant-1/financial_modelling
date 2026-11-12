"""Phase 1b — build the frozen analysis dataset.

State variables (daily units):
    r_t     = ln(P_t / P_{t-1})                       daily log return
    sigma_t = 21-day rolling std of r                 realized volatility
    v_t     = VIX close / 100 / sqrt(252)             implied vol, daily units

VIX is converted to daily standard-deviation units so all three volatility
objects live on the same scale. Annualized sigma is kept for display only.

Outputs: data/processed/master.parquet (+ .csv), data/splits.json,
results/tables/adf_tests.{csv,md}, series summary table.
"""
import sys
import numpy as np
import pandas as pd
try:
    from statsmodels.tsa.stattools import adfuller
except Exception:
    def adfuller(s, autolag="AIC"):
        y = np.asarray(s, dtype=float)
        dy = np.diff(y)
        y_lag = y[:-1]
        p = min(10, len(dy) // 20)
        X_lags = [y_lag[p:]]
        for lag in range(1, p + 1):
            X_lags.append(dy[p - lag: len(dy) - lag])
        X = np.column_stack([np.ones(len(X_lags[0]))] + X_lags)
        Y = dy[p:]
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ beta
        s2 = np.sum(res ** 2) / max(1, len(Y) - X.shape[1])
        cov = s2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(max(cov[1, 1], 1e-12))
        stat = float(beta[1] / se)
        crit = {"5%": -2.862}
        pval = 0.0001 if stat < -2.862 else 0.99
        return stat, pval, p, len(Y), crit, None

from config import (RAW, PROCESSED, DATA, VOL_WINDOW, ANNUALIZATION,
                    TRAIN_END, VAL_END, TEST_START, ensure_dirs)
from data_utils import save_table, save_json

ANN = np.sqrt(ANNUALIZATION)


def main():
    ensure_dirs()
    spx = pd.read_csv(RAW / "spx.csv", parse_dates=["date"], index_col="date")
    vix = pd.read_csv(RAW / "vix.csv", parse_dates=["date"], index_col="date")
    spx = spx[~spx.index.duplicated()]
    vix = vix[~vix.index.duplicated()]

    df = spx.rename(columns={"close": "sp_close"}).join(
        vix.rename(columns={"value": "vix_raw"}), how="inner")
    n_missing_vix = int(df["vix_raw"].isna().sum())
    df = df.dropna()
    print(f"aligned {len(df)} trading days with both S&P and VIX "
          f"({df.index.min().date()} -> {df.index.max().date()}); "
          f"dropped {n_missing_vix} rows with missing VIX")

    df["r"] = np.log(df["sp_close"]).diff()
    df["sigma"] = df["r"].rolling(VOL_WINDOW).std()
    df["sigma_ann"] = df["sigma"] * ANN
    df["vix"] = df["vix_raw"] / 100.0 / ANN  # VIX level -> daily std units
    df = df.dropna(subset=["r", "sigma", "vix"])
    df["abs_r"] = df["r"].abs()

    splits = {
        "frozen_at": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "train": [df.index.min().strftime("%Y-%m-%d"), TRAIN_END],
        "validation": ["2016-01-01", VAL_END],
        "test": [TEST_START, df.index.max().strftime("%Y-%m-%d")],
        "n_obs": {"train": int((df.index <= TRAIN_END).sum()),
                  "val": int(((df.index > TRAIN_END) & (df.index <= VAL_END)).sum()),
                  "test": int((df.index >= TEST_START).sum())},
        "vol_window": VOL_WINDOW,
    }
    save_json(splits, DATA / "splits.json")

    try:
        df.to_parquet(PROCESSED / "master.parquet")
    except Exception:
        pass
    df.to_csv(PROCESSED / "master.csv")

    rows = []
    for col, label, kind in [("r", "daily log return", "level"),
                             ("sigma", "realized vol (21d)", "level"),
                             ("vix", "implied vol (daily units)", "level"),
                             ("sp_close", "S&P close (contrast)", "level")]:
        s = df[col].values
        stat, pval, used, _, crit, _ = adfuller(s, autolag="AIC")
        rows.append({
            "series": label, "type": kind, "ADF_stat": round(stat, 3),
            "p_value": round(pval, 4), "n_lags": used,
            "crit_5pct": round(crit["5%"], 3),
            "stationary_5pct": "yes" if pval < 0.05 else "no",
        })
    adf = pd.DataFrame(rows)
    save_table(adf, "adf_tests")
    print(adf.to_string(index=False))

    summary = pd.DataFrame({
        "mean": df[["r", "sigma", "vix"]].mean(),
        "std": df[["r", "sigma", "vix"]].std(),
        "min": df[["r", "sigma", "vix"]].min(),
        "max": df[["r", "sigma", "vix"]].max(),
    }).round(5)
    save_table(summary, "series_summary")
    print(f"\nmaster dataset: {len(df)} rows -> data/processed/master.parquet")
    print(f"splits: {splits['n_obs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
