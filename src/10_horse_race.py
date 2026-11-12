"""Phase 6 — the forecasting horse race (answers RQ3).

Protocol (strict, identical for every model):
- walk-forward, expanding window, quarterly refits over the frozen test
  window (2020 -> latest; includes COVID and the 2022 rate cycle on purpose);
- between refits, models roll daily with actual data (causal filters/lags);
- daily forecast origins; horizons h = 1, 5, 21 for realized vol sigma;
- metrics: RMSE, MAE, QLIKE (robust vol loss), Mincer-Zarnowitz regressions,
  Diebold-Mariano tests vs GARCH(1,1) and vs EWMA (HAC, overlap-aware);
- secondary honesty report: return-direction accuracy vs a coin flip.

STATA walk-forward forecasts, if produced via stata/05_walkforward_garch.do,
are folded in automatically from results/stata/stata_forecasts.csv.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats as sps

import matplotlib.pyplot as plt

from config import HORIZONS, FORECASTS, STATA_RESULTS, ensure_dirs
from models import build_zoo
from data_utils import load_master, save_table, save_fig, split_masks

FAST = os.environ.get("FAST", "0") == "1"


def qlike(rv, f):
    return float(np.mean(rv / f - np.log(rv / f) - 1.0))


def dm_test(loss_a, loss_b, lag):
    """Diebold-Mariano with Newey-West HAC variance (Bartlett weights)."""
    d = loss_a - loss_b
    n = len(d)
    dm = d - d.mean()
    g0 = float((dm @ dm) / n)
    lrv = g0
    for l in range(1, min(lag, n - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        gl = float((dm[l:] @ dm[:-l]) / n)
        lrv += 2.0 * w * gl
    stat = d.mean() / np.sqrt(max(lrv, 1e-18) / n)
    return float(stat), float(2 * (1 - sps.norm.cdf(abs(stat))))


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def mz_regression(rv, f, lag):
    """Mincer-Zarnowitz: regress realized on forecast, HAC standard errors."""
    try:
        import statsmodels.api as sm
        X = sm.add_constant(f)
        res = sm.OLS(rv, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        return float(res.params[0]), float(res.pvalues[0]), float(res.params[1]), \
            float(res.pvalues[1]), float(res.rsquared)
    except Exception:
        import sindy_core as sc
        X = np.column_stack([np.ones(len(f)), f])
        beta, se, tvalues, pvalues, r2 = sc._hac_ols(rv, X, maxlags=lag)
        return float(beta[0]), float(pvalues[0]), float(beta[1]), float(pvalues[1]), float(r2)


def main():
    ensure_dirs()
    df = load_master()
    train, val, test = split_masks(df)
    T = len(df)
    test_idx = np.where(test)[0]
    max_h = max(HORIZONS)
    sig = df["sigma"].to_numpy(float)
    ret = df["r"].to_numpy(float)
    dates = df.index

    quarters = pd.PeriodIndex(dates[test_idx], freq="Q")
    blocks = []
    for q in quarters.unique():
        blocks.append(test_idx[quarters == q])
    print(f"test window: {dates[test_idx[0]].date()} -> "
          f"{dates[test_idx[-1]].date()}  "
          f"({len(test_idx)} days, {len(blocks)} quarterly refit blocks)")

    zoo = build_zoo(df)
    recs = []
    r_recs = []
    reuse_fc = (os.environ.get("REUSE", "0") == "1" or "--reuse" in sys.argv) and (FORECASTS / "all_forecasts.csv").exists()
    if reuse_fc:
        print("reusing existing results/forecasts/all_forecasts.csv ...")
        fc = pd.read_csv(FORECASTS / "all_forecasts.csv", parse_dates=["date"])
    else:
        for b, block in enumerate(blocks):
            start = int(block[0])
            origins = np.array([t for t in block if t + max_h <= T - 1])
            if len(origins) == 0:
                continue
            t_block = time.time()
            for model in zoo:
                try:
                    model.fit(start)
                    fc = model.sigma_forecasts(origins, HORIZONS)
                    for h in HORIZONS:
                        for k, t in enumerate(origins):
                            recs.append({"model": model.name,
                                         "date": dates[t], "h": h,
                                         "forecast": fc[h][k]})
                    if model.produces_r:
                        rh = model.r_forecasts(origins)
                        for k, t in enumerate(origins):
                            r_recs.append({"model": model.name,
                                           "date": dates[t],
                                           "r_hat": rh[k], "r_actual": ret[t + 1]})
                except Exception as e:
                    print(f"  [warn] {model.name} failed in block "
                          f"{dates[start].date()}: {e}")
            print(f"  block {b + 1}/{len(blocks)} ({dates[start].date()}): "
                  f"{len(origins)} origins, {time.time() - t_block:.1f}s")

    if not reuse_fc:
        fc = pd.DataFrame(recs)
        fc.to_csv(FORECASTS / "all_forecasts.csv", index=False)

    # fold in STATA forecasts if the do-files were run
    stata_path = STATA_RESULTS / "stata_forecasts.csv"
    if stata_path.exists():
        st = pd.read_csv(stata_path, parse_dates=["date"])
        origin_set = set(zip(fc["date"], fc["h"]))
        extra = []
        for _, row in st.iterrows():
            key = (row["date"], int(row["h"]))
            if key in origin_set:
                extra.append({"model": "stata_garch", "date": row["date"],
                              "h": int(row["h"]), "forecast": row["forecast"]})
        if extra:
            fc = pd.concat([fc, pd.DataFrame(extra)], ignore_index=True)
            print(f"merged {len(extra)} STATA forecast rows")

    # realized values (sigma is daily units)
    date_to_pos = {d: i for i, d in enumerate(dates)}
    fc["realized"] = [sig[date_to_pos[d] + int(h)]
                      for d, h in zip(fc["date"], fc["h"])]

    # ---- metrics per model x horizon ----
    rows = []
    for (mname, h), g in fc.groupby(["model", "h"]):
        valid = np.isfinite(g["forecast"].to_numpy(float)) & np.isfinite(g["realized"].to_numpy(float))
        if not np.any(valid):
            continue
        gv = g[valid]
        f = np.clip(gv["forecast"].to_numpy(float), 1e-8, None)
        rv = gv["realized"].to_numpy(float)
        lag = min(h + 20, len(f) // 5)
        mz = mz_regression(rv, f, lag)
        rows.append({
            "model": mname, "h": h, "n": len(f),
            "RMSE": float(np.sqrt(np.mean((rv - f) ** 2))),
            "MAE": float(np.mean(np.abs(rv - f))),
            "QLIKE": qlike(rv, f),
            "MZ_intercept": mz[0], "MZ_intercept_p": mz[1],
            "MZ_slope": mz[2], "MZ_slope_p": mz[3], "MZ_R2": mz[4],
        })
    met = pd.DataFrame(rows)

    # DM tests vs benchmarks, per h
    for bench in ("garch11", "ewma"):
        col = f"DM_vs_{bench}"
        pcol = col + "_p"
        met[col] = np.nan
        met[pcol] = np.nan
        for h in HORIZONS:
            losses = {}
            for mname, g in fc[fc["h"] == h].groupby("model"):
                f = np.clip(g["forecast"].to_numpy(float), 1e-8, None)
                rv = g["realized"].to_numpy(float)
                losses[mname] = rv / f - np.log(rv / f)  # QLIKE loss
            if bench not in losses:
                continue
            for mname in losses:
                if mname == bench:
                    continue
                # align on common dates
                g1 = fc[(fc["h"] == h) & (fc["model"] == mname)]
                g2 = fc[(fc["h"] == h) & (fc["model"] == bench)]
                j = g1.merge(g2, on="date", suffixes=("_a", "_b")).dropna(
                    subset=["realized_a", "forecast_a", "realized_b", "forecast_b"])
                if len(j) == 0:
                    continue
                la = (j["realized_a"] / np.clip(j["forecast_a"], 1e-8, None)
                      - np.log(j["realized_a"] / np.clip(j["forecast_a"], 1e-8, None)))
                lb = (j["realized_b"] / np.clip(j["forecast_b"], 1e-8, None)
                      - np.log(j["realized_b"] / np.clip(j["forecast_b"], 1e-8, None)))
                stat, p = dm_test(la.to_numpy(float), lb.to_numpy(float),
                                  min(h + 20, len(j) // 5))
                mask = (met["model"] == mname) & (met["h"] == h)
                met.loc[mask, col] = stat
                met.loc[mask, pcol] = p

    met = met.round(5)
    met.to_csv(FORECASTS / "horse_race_metrics.csv", index=False)
    save_table(met[met["h"] == 1].sort_values("QLIKE"), "horse_race_h1")
    save_table(met[met["h"] == 21].sort_values("QLIKE"), "horse_race_h21")

    main_tab = met[met["h"] == 1].sort_values("QLIKE").copy()
    main_tab["DM_vs_garch"] = [
        f"{s:+.2f}{stars(p)}" if np.isfinite(s) else "-"
        for s, p in zip(main_tab["DM_vs_garch11"], main_tab["DM_vs_garch11_p"])]
    save_table(main_tab[["model", "n", "RMSE", "MAE", "QLIKE", "MZ_slope",
                         "MZ_R2", "DM_vs_garch"]],
               "horse_race_main")

    # ---- return-direction honesty report ----
    if r_recs:
        rr = pd.DataFrame(r_recs)
        rows = []
        n0 = (rr["model"] == rr["model"].iloc[0]).sum()
        for mname, g in rr.groupby("model"):
            hit = float(np.mean(np.sign(g["r_hat"]) == np.sign(g["r_actual"])))
            n = len(g)
            z = (hit - 0.5) / np.sqrt(0.25 / n)
            p = 2 * (1 - sps.norm.cdf(abs(z)))
            rows.append({"model": mname, "n": n, "direction_accuracy": hit,
                         "p_vs_coinflip": p})
        direction = pd.DataFrame(rows).round(4)
        save_table(direction, "return_direction_accuracy")
        print(direction.to_string(index=False))

    # ---- figures ----
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    for ax, (mname, color) in zip(axes, [("garch11", "steelblue"),
                                         ("sindy_stlsq", "firebrick"),
                                         ("random_forest", "seagreen")]):
        g = fc[(fc["h"] == 1) & (fc["model"] == mname)]
        if len(g) == 0:
            continue
        ax.scatter(g["realized"] * 100, g["forecast"] * 100, s=2, alpha=0.2,
                   color=color)
        lim = [0, max(g["realized"].max(), g["forecast"].max()) * 100 * 1.05]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("realized sigma (daily %)")
        ax.set_title(mname)
    axes[0].set_ylabel("forecast sigma (daily %)")
    fig.suptitle("Mincer-Zarnowitz calibration, h=1 (dots on diagonal = perfect)",
                 y=1.04)
    save_fig(fig, "fig14_calibration_h1")

    h1 = met[met["h"] == 1].sort_values("QLIKE")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(h1["model"], h1["QLIKE"],
           color=["firebrick" if "sindy" in m else "steelblue"
                  for m in h1["model"]])
    ax.set_ylabel("QLIKE (lower = better)")
    ax.set_title("One-day-ahead volatility forecast race")
    ax.tick_params(axis="x", rotation=45)
    save_fig(fig, "fig15_qlike_bars")

    ga = fc[(fc["h"] == 1) & (fc["model"] == "garch11")].set_index("date")
    sy = fc[(fc["h"] == 1) & (fc["model"] == "sindy_stlsq")].set_index("date")
    if len(ga) and len(sy):
        j = ga[["forecast", "realized"]].join(
            sy[["forecast"]], lsuffix="_garch", rsuffix="_sindy").dropna()
        la = j["realized"] / j["forecast_garch"] - np.log(
            j["realized"] / j["forecast_garch"])
        lb = j["realized"] / j["forecast_sindy"] - np.log(
            j["realized"] / j["forecast_sindy"])
        diff = (lb - la).rolling(63).mean()
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(diff.index, diff, color="firebrick", lw=1)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_ylabel("QLIKE(SINDy) - QLIKE(GARCH), 63-day mean")
        ax.set_title("When does SINDy beat GARCH? (below zero = SINDy better)")
        save_fig(fig, "fig16_rolling_qlike_diff")

    print("\n=== One-day-ahead results (sorted by QLIKE) ===")
    print(main_tab[["model", "RMSE", "MAE", "QLIKE", "DM_vs_garch"]]
          .to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
