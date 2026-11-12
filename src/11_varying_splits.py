"""Multi-split validation of SINDy dynamics and forecasting.

Evaluates whether SINDy discovered equations and out-of-sample forecast
performance remain stable across diverse train/val/test split configurations:
1. Baseline (70/11/19): Train 1990-2015, Val 2016-2019, Test 2020-2026 (COVID/rates)
2. Ratio 70/15/15: Chronological percentage split
3. Ratio 60/20/20: Chronological percentage split
4. Pre-GFC Test: Train 1990-2003, Val 2004-2005, Test 2006-2007 (Great Moderation)
5. Post-GFC Test: Train 1990-2007, Val 2008-2009, Test 2010-2015 (Post-Crisis QE)
6. Moderation Test: Train 1990-2012, Val 2013-2015, Test 2016-2019 (Low-vol cycle)

Outputs:
  results/tables/varying_splits_equations.{csv,md}
  results/tables/varying_splits_metrics.{csv,md}
  results/tables/varying_splits_sindy_vs_garch.{csv,md}
  results/figures/fig17_split_equation_stability.png
  results/figures/fig18_split_forecast_comparison.png
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import LIBRARY_TERMS, HORIZONS, ensure_dirs
from data_utils import load_master, save_table, save_fig, zscore, apply_zscore, undo_zscore
import sindy_core as sc
import garch as garch_mod


def qlike_loss(rv, f):
    f_safe = np.clip(f, 1e-8, None)
    rv_safe = np.clip(rv, 1e-8, None)
    return float(np.mean(rv_safe / f_safe - np.log(rv_safe / f_safe) - 1.0))


def compute_metrics(rv, f):
    f_safe = np.clip(f, 1e-8, None)
    rv_safe = np.clip(rv, 1e-8, None)
    rmse = float(np.sqrt(np.mean((rv_safe - f_safe) ** 2)))
    mae = float(np.mean(np.abs(rv_safe - f_safe)))
    ql = qlike_loss(rv_safe, f_safe)
    return rmse, mae, ql


def get_split_definitions(df: pd.DataFrame):
    """Return dictionary of split definitions."""
    dates = df.index
    T = len(df)
    
    # 70/15/15
    n_tr_70 = int(T * 0.70)
    n_va_15 = int(T * 0.15)
    
    # 60/20/20
    n_tr_60 = int(T * 0.60)
    n_va_20 = int(T * 0.20)
    
    splits = {
        "Baseline (1990-2015 | 2016-19 | 2020-26)": {
            "tr_mask": dates <= pd.Timestamp("2015-12-31"),
            "va_mask": (dates > pd.Timestamp("2015-12-31")) & (dates <= pd.Timestamp("2019-12-31")),
            "te_mask": dates >= pd.Timestamp("2020-01-01"),
            "desc": "Baseline test: COVID & 2022 rate cycle (2020-2026)"
        },
        "Split 70/15/15": {
            "tr_mask": np.arange(T) < n_tr_70,
            "va_mask": (np.arange(T) >= n_tr_70) & (np.arange(T) < n_tr_70 + n_va_15),
            "te_mask": np.arange(T) >= n_tr_70 + n_va_15,
            "desc": f"Fixed 70/15/15 ratio ({dates[n_tr_70+n_va_15].strftime('%Y-%m')} to end)"
        },
        "Split 60/20/20": {
            "tr_mask": np.arange(T) < n_tr_60,
            "va_mask": (np.arange(T) >= n_tr_60) & (np.arange(T) < n_tr_60 + n_va_20),
            "te_mask": np.arange(T) >= n_tr_60 + n_va_20,
            "desc": f"Fixed 60/20/20 ratio ({dates[n_tr_60+n_va_20].strftime('%Y-%m')} to end)"
        },
        "Pre-GFC (Test: 2006-2007)": {
            "tr_mask": dates <= pd.Timestamp("2003-12-31"),
            "va_mask": (dates > pd.Timestamp("2003-12-31")) & (dates <= pd.Timestamp("2005-12-31")),
            "te_mask": (dates > pd.Timestamp("2005-12-31")) & (dates <= pd.Timestamp("2007-12-31")),
            "desc": "Pre-GFC Great Moderation test (2006-2007)"
        },
        "GFC & Post-GFC (Test: 2008-2019)": {
            "tr_mask": dates <= pd.Timestamp("2005-12-31"),
            "va_mask": (dates > pd.Timestamp("2005-12-31")) & (dates <= pd.Timestamp("2007-12-31")),
            "te_mask": (dates > pd.Timestamp("2007-12-31")) & (dates <= pd.Timestamp("2019-12-31")),
            "desc": "Crisis & Post-GFC QE expansion test (2008-2019)"
        },
        "Low-Vol Moderation (Test: 2016-2019)": {
            "tr_mask": dates <= pd.Timestamp("2012-12-31"),
            "va_mask": (dates > pd.Timestamp("2012-12-31")) & (dates <= pd.Timestamp("2015-12-31")),
            "te_mask": (dates > pd.Timestamp("2015-12-31")) & (dates <= pd.Timestamp("2019-12-31")),
            "desc": "Mid-2010s low-vol expansion test (2016-2019)"
        },
    }
    return splits


def run_split_evaluation():
    ensure_dirs()
    df = load_master()
    dates = df.index
    T = len(df)
    
    r_all = df["r"].to_numpy(float)
    sig_all = df["sigma"].to_numpy(float)
    vix_all = df["vix"].to_numpy(float)
    X_full = np.column_stack([r_all, sig_all, vix_all])
    
    splits = get_split_definitions(df)
    
    eq_rows = []
    metric_rows = []
    
    print(f"=== Running Varying Splits Evaluation ({len(splits)} configurations) ===")
    
    for s_name, s_info in splits.items():
        tr_m = s_info["tr_mask"]
        va_m = s_info["va_mask"]
        te_m = s_info["te_mask"]
        
        n_tr = int(np.sum(tr_m))
        n_va = int(np.sum(va_m))
        n_te = int(np.sum(te_m))
        
        te_indices = np.where(te_m)[0]
        t_te_start = te_indices[0]
        t_te_end = te_indices[-1]
        te_dates = f"{dates[t_te_start].strftime('%Y-%m-%d')} to {dates[t_te_end].strftime('%Y-%m-%d')}"
        
        print(f"\nConfiguration: {s_name}")
        print(f"  Train: {n_tr} obs, Val: {n_va} obs, Test: {n_te} obs ({te_dates})")
        
        X_tr = X_full[tr_m]
        X_va = X_full[va_m]
        
        # Standardize strictly on Train statistics
        Z_tr, mu, sd = zscore(X_tr)
        Z_va = apply_zscore(X_va, mu, sd)
        Z_full = apply_zscore(X_full, mu, sd)
        
        # SINDy identification
        Th_tr = sc.build_library(Z_tr[:-1])
        Th_va = sc.build_library(Z_va[:-1])
        path = sc.sweep_lambda(Th_tr, Z_tr[1:], Th_va, Z_va[1:])
        lam_star = sc.select_lambda(path, n_val=len(Z_va) - 1)
        
        Xi_stlsq = sc.identify(Th_tr, Z_tr[1:], lam_star, solver="stlsq")
        Xi_sr3 = sc.identify(Th_tr, Z_tr[1:], lam_star, solver="sr3")
        
        # Record discovered equations for sigma_next
        for i, term in enumerate(LIBRARY_TERMS):
            eq_rows.append({
                "split": s_name,
                "term": term,
                "coef_stlsq": round(float(Xi_stlsq[i, 1]), 4),
                "active_stlsq": int(abs(Xi_stlsq[i, 1]) > 0),
                "coef_sr3": round(float(Xi_sr3[i, 1]), 4),
                "active_sr3": int(abs(Xi_sr3[i, 1]) > 0),
                "lambda_star": round(lam_star, 4)
            })
            
        # Fit benchmark GARCH(1,1) on Train returns
        r_tr = r_all[tr_m]
        try:
            garch_fit = garch_mod.fit_garch(r_tr, gjr=False)
        except Exception:
            garch_fit = garch_mod.fit_garch_qmle(r_tr, gjr=False)
            
        omega = float(garch_fit["omega"])
        alpha = float(garch_fit["alpha"])
        beta = float(garch_fit["beta"])
        pers = alpha + beta
        
        # Causal GARCH filter: s2_garch[t+1] is conditional variance at day t+1 given info up to day t
        s2_garch = garch_mod.filter_sigma2(r_all, garch_fit)
        
        # Causal EWMA filter (RiskMetrics lambda=0.94)
        lam_ewma = 0.94
        s2_ewma = np.empty(T)
        s2_ewma[0] = float(np.var(r_tr, ddof=1))
        e2 = (r_all - float(garch_fit.get("mu", r_tr.mean()))) ** 2
        for t in range(1, T):
            s2_ewma[t] = lam_ewma * s2_ewma[t - 1] + (1 - lam_ewma) * e2[t - 1]
            
        # Fit benchmark AR(p) on Train realized volatility with BIC order selection
        sig_tr = sig_all[tr_m]
        best_ar = (np.inf, 1, np.zeros(2))
        for p_lag in range(1, 6):
            if len(sig_tr) <= p_lag + 5:
                continue
            Y_ar = sig_tr[p_lag:]
            F_ar = np.column_stack([sig_tr[p_lag - 1 - i: len(sig_tr) - 1 - i] for i in range(p_lag)])
            F_ar = np.column_stack([np.ones(len(F_ar)), F_ar])
            b_ar, *_ = np.linalg.lstsq(F_ar, Y_ar, rcond=None)
            res_ar = Y_ar - F_ar @ b_ar
            bic_ar = len(Y_ar) * np.log(np.mean(res_ar ** 2)) + F_ar.shape[1] * np.log(len(Y_ar))
            if bic_ar < best_ar[0]:
                best_ar = (bic_ar, p_lag, b_ar)
        p_opt, beta_ar = best_ar[1], best_ar[2]

        def ar_forecast(origin, h_ahead):
            hist = list(sig_all[origin - p_opt + 1: origin + 1])
            for _ in range(h_ahead):
                lags = np.array(hist[-p_opt:][::-1])
                nxt = beta_ar[0] + beta_ar[1:] @ lags
                hist.append(nxt)
            return hist[-1]

        r2_all = r_all ** 2
        W_LEN = 21
            
        # Evaluate Out-of-Sample Forecasts on Test window
        max_h = max(HORIZONS)
        origins = [t for t in te_indices if t + max_h < T]
        n_origins = len(origins)
        
        for h in HORIZONS:
            realized_h = np.array([sig_all[t + h] for t in origins])
            
            # 1. SINDy STLSQ & SR3 forecasts (iterated map from origin)
            f_sindy_stlsq = []
            f_sindy_sr3 = []
            for t in origins:
                x0 = Z_full[t]
                path_stlsq = sc.iterate_map(x0, Xi_stlsq, h)
                f_sindy_stlsq.append(path_stlsq[h - 1, 1] * sd[1] + mu[1])
                path_sr3 = sc.iterate_map(x0, Xi_sr3, h)
                f_sindy_sr3.append(path_sr3[h - 1, 1] * sd[1] + mu[1])
                
            f_sindy_stlsq = np.array(f_sindy_stlsq)
            f_sindy_sr3 = np.array(f_sindy_sr3)
            
            # 2. GARCH(1,1) h-step variance forecasts mapped to 21-day rolling vol
            # Apples-to-apples mapping matching GARCHModel in src/models.py
            f_garch = []
            for t in origins:
                s2j = s2_garch[t + 1]              # sigma2_{t+1|t}
                fsum = 0.0
                for _ in range(h):
                    fsum += s2j
                    s2j = omega + pers * s2j
                lo = t + h - W_LEN
                real = r2_all[max(lo, 0): t + 1].sum()
                f_garch.append(np.sqrt(max(0.0, real + fsum) / W_LEN))
            f_garch = np.array(f_garch)
            
            # 3. EWMA h-step forecast mapped to 21-day rolling vol
            # Matches EWMA in src/models.py
            f_ewma = []
            for t in origins:
                s2n = s2_ewma[t + 1] if t + 1 < T else s2_ewma[t]
                lo = t + h - W_LEN
                real = r2_all[max(lo, 0): t + 1].sum()
                f_ewma.append(np.sqrt(max(0.0, real + h * s2n) / W_LEN))
            f_ewma = np.array(f_ewma)
            
            # 4. AR(sigma) forecast
            f_ar = np.array([ar_forecast(t, h) for t in origins])
            
            # 5. Naive Random Walk: tomorrow's vol = today's vol
            f_rw = np.array([sig_all[t] for t in origins])
            
            # Compute evaluation metrics
            models_dict = {
                "SINDy (STLSQ)": f_sindy_stlsq,
                "SINDy (SR3)": f_sindy_sr3,
                "GARCH(1,1)": f_garch,
                "EWMA (0.94)": f_ewma,
                "AR(sigma)": f_ar,
                "Naive RW": f_rw,
            }
            
            for m_name, f_vals in models_dict.items():
                rmse, mae, ql = compute_metrics(realized_h, f_vals)
                metric_rows.append({
                    "split": s_name,
                    "test_period": te_dates,
                    "model": m_name,
                    "h": h,
                    "n_eval": n_origins,
                    "RMSE": round(rmse, 6),
                    "MAE": round(mae, 6),
                    "QLIKE": round(ql, 6),
                })

    df_eq = pd.DataFrame(eq_rows)
    df_met = pd.DataFrame(metric_rows)
    
    save_table(df_eq, "varying_splits_equations")
    save_table(df_met, "varying_splits_metrics")
    
    # Compile Equation Term Stability Summary
    summary_terms = []
    for term in LIBRARY_TERMS:
        sub = df_eq[df_eq["term"] == term]
        stlsq_inc = float(sub["active_stlsq"].mean())
        sr3_inc = float(sub["active_sr3"].mean())
        c_stlsq = sub["coef_stlsq"]
        summary_terms.append({
            "term": term,
            "STLSQ_inclusion_pct": round(stlsq_inc * 100, 1),
            "SR3_inclusion_pct": round(sr3_inc * 100, 1),
            "mean_coef_STLSQ": round(float(c_stlsq.mean()), 4),
            "std_coef_STLSQ": round(float(c_stlsq.std()), 4),
            "min_coef_STLSQ": round(float(c_stlsq.min()), 4),
            "max_coef_STLSQ": round(float(c_stlsq.max()), 4),
        })
    df_term_sum = pd.DataFrame(summary_terms)
    save_table(df_term_sum, "varying_splits_term_summary")
    print("\n=== SINDy Sigma-Equation Stability Across All Splits ===")
    print(df_term_sum.to_string(index=False))
    
    # Compile Head-to-Head SINDy vs GARCH table
    h2h_rows = []
    for (split_name, h), g in df_met.groupby(["split", "h"]):
        sindy_row = g[g["model"] == "SINDy (STLSQ)"].iloc[0]
        garch_row = g[g["model"] == "GARCH(1,1)"].iloc[0]
        rw_row = g[g["model"] == "Naive RW"].iloc[0]
        
        ql_sindy = sindy_row["QLIKE"]
        ql_garch = garch_row["QLIKE"]
        ql_rw = rw_row["QLIKE"]
        
        rmse_sindy = sindy_row["RMSE"]
        rmse_garch = garch_row["RMSE"]
        
        sindy_wins_qlike = ql_sindy < ql_garch
        sindy_wins_rmse = rmse_sindy < rmse_garch
        
        h2h_rows.append({
            "split": split_name,
            "h": h,
            "SINDy_QLIKE": ql_sindy,
            "GARCH_QLIKE": ql_garch,
            "RW_QLIKE": ql_rw,
            "QLIKE_ratio (SINDy/GARCH)": round(ql_sindy / max(ql_garch, 1e-8), 4),
            "SINDy_RMSE": rmse_sindy,
            "GARCH_RMSE": rmse_garch,
            "RMSE_ratio (SINDy/GARCH)": round(rmse_sindy / max(rmse_garch, 1e-8), 4),
            "SINDy_beats_GARCH": "YES" if (sindy_wins_qlike and sindy_wins_rmse) else ("QLIKE only" if sindy_wins_qlike else "NO")
        })
    df_h2h = pd.DataFrame(h2h_rows)
    save_table(df_h2h, "varying_splits_sindy_vs_garch")
    print("\n=== Head-to-Head: SINDy vs GARCH(1,1) Across Splits ===")
    print(df_h2h[["split", "h", "QLIKE_ratio (SINDy/GARCH)", "RMSE_ratio (SINDy/GARCH)", "SINDy_beats_GARCH"]].to_string(index=False))
    
    # --- Generate Figures ---
    # Fig 17: Equation Term Stability Heatmap
    piv = df_eq.pivot(index="split", columns="term", values="coef_stlsq")[LIBRARY_TERMS]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cax = ax.imshow(piv.to_numpy(), cmap="coolwarm", aspect="auto", vmin=-0.1, vmax=1.0)
    ax.set_xticks(range(len(LIBRARY_TERMS)))
    ax.set_xticklabels(LIBRARY_TERMS, rotation=45, ha="right")
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=8)
    for r in range(len(piv)):
        for c in range(len(LIBRARY_TERMS)):
            v = piv.iloc[r, c]
            ax.text(c, r, f"{v:.2f}" if abs(v) > 0 else "-", ha="center", va="center",
                    fontsize=8, color="white" if abs(v) > 0.5 else "black")
    ax.grid(False)
    fig.colorbar(cax, label="Discovered Coefficient (Standardized)")
    ax.set_title("SINDy Sigma-Equation Coefficient Stability Across Splits (STLSQ)")
    save_fig(fig, "fig17_split_equation_stability")
    
    # Fig 18: Forecast Performance Comparison (h=1 QLIKE)
    h1_met = df_met[df_met["h"] == 1]
    split_names = list(splits.keys())
    models_to_plot = ["SINDy (STLSQ)", "GARCH(1,1)", "Naive RW"]
    colors = ["firebrick", "steelblue", "gray"]
    
    x = np.arange(len(split_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 4))
    for idx, (m_name, col) in enumerate(zip(models_to_plot, colors)):
        sub = h1_met[h1_met["model"] == m_name].set_index("split").reindex(split_names)
        ax.bar(x + idx * width, sub["QLIKE"], width, label=m_name, color=col, alpha=0.85)
        
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.split(":")[0].replace(" (Test", "\nTest") for s in split_names], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("QLIKE Loss (lower is better)")
    ax.set_title("Out-of-Sample Volatility Forecast Performance Across 6 Split Regimes (h=1)")
    ax.legend()
    save_fig(fig, "fig18_split_forecast_comparison")
    
    print("\nVarying splits evaluation complete. All tables and figures written.")
    return 0


if __name__ == "__main__":
    sys.exit(run_split_evaluation())
