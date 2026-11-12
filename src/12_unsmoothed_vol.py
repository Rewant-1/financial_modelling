"""Evaluation of SINDy on Unsmoothed Daily Volatility (No 21-day rolling smoothing).

Investigates how rolling-window smoothing dilutes single-day GARCH shocks ($r^2$)
and leverage terms ($r|r|$). Evaluates SINDy discovery across a window sweep
W in [1, 2, 3, 5, 10, 21]:
- W = 1: Unsmoothed daily instantaneous realized volatility (|r_t| * sqrt(pi/2) or Garman-Klass / Yang-Zhang if OHLC available)
- W = 2: 2-day realized volatility
- W = 3: 3-day realized volatility
- W = 5: 1-week realized volatility
- W = 10: 2-week realized volatility
- W = 21: Original 1-month rolling realized volatility

Demonstrates whether SINDy discovers the discrete GARCH shock ($r^2$) and
asymmetry ($r|r|$) when liberated from 21-day smoothing dilution.

Outputs:
  results/tables/unsmoothed_vol_equations.{csv,md}
  results/tables/unsmoothed_vol_metrics.{csv,md}
  results/tables/unsmoothed_vs_smoothed_comparison.{csv,md}
  results/figures/fig19_unsmoothed_term_evolution.png
  results/figures/fig20_unsmoothed_forecast_bars.png
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import LIBRARY_TERMS, HORIZONS, ensure_dirs
from data_utils import load_master, save_table, save_fig, split_masks, zscore, apply_zscore
import sindy_core as sc
import garch as garch_mod


VOL_FLOOR = 1e-4  # 1 bps daily vol floor to prevent asymptotic singularity in QLIKE


def qlike_loss(rv, f, floor=VOL_FLOOR):
    f_safe = np.clip(f, floor, None)
    rv_safe = np.clip(rv, floor, None)
    return float(np.mean(rv_safe / f_safe - np.log(rv_safe / f_safe) - 1.0))


def compute_metrics(rv, f, floor=VOL_FLOOR):
    f_safe = np.clip(f, floor, None)
    rv_safe = np.clip(rv, floor, None)
    rmse = float(np.sqrt(np.mean((rv_safe - f_safe) ** 2)))
    mae = float(np.mean(np.abs(rv_safe - f_safe)))
    ql = qlike_loss(rv_safe, f_safe, floor=floor)
    return rmse, mae, ql


# ==============================================================================
# Daily Volatility Estimators (Intraday OHLC and Close-to-Close)
# ==============================================================================

def garman_klass_vol(open_p: np.ndarray, high_p: np.ndarray,
                     low_p: np.ndarray, close_p: np.ndarray) -> np.ndarray:
    """Garman-Klass (1980) intraday volatility estimator.
    sigma_GK^2 = 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2
    Under continuous geometric Brownian motion, has ~8x lower variance than close-to-close.
    """
    hl = np.log(np.maximum(high_p, 1e-8) / np.maximum(low_p, 1e-8))
    co = np.log(np.maximum(close_p, 1e-8) / np.maximum(open_p, 1e-8))
    var = 0.5 * (hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (co ** 2)
    return np.sqrt(np.maximum(var, 1e-12))


def parkinson_vol(high_p: np.ndarray, low_p: np.ndarray) -> np.ndarray:
    """Parkinson (1980) extreme-value intraday volatility estimator.
    sigma_Parkinson^2 = (ln(H/L))^2 / (4 * ln(2))
    """
    hl = np.log(np.maximum(high_p, 1e-8) / np.maximum(low_p, 1e-8))
    var = (hl ** 2) / (4.0 * np.log(2.0))
    return np.sqrt(np.maximum(var, 1e-12))


def yang_zhang_vol(open_p: np.ndarray, high_p: np.ndarray,
                   low_p: np.ndarray, close_p: np.ndarray,
                   k: float = 0.34) -> np.ndarray:
    """Yang-Zhang (2000) drift-independent, jump-robust intraday volatility estimator.
    Combines overnight return variance (close-to-open), open-to-close return variance,
    and Rogers-Satchell drift-independent variance.
    """
    T = len(close_p)
    # Overnight return
    co_prev = np.concatenate([[close_p[0]], close_p[:-1]])
    r_overnight = np.log(np.maximum(open_p, 1e-8) / np.maximum(co_prev, 1e-8))
    r_oc = np.log(np.maximum(close_p, 1e-8) / np.maximum(open_p, 1e-8))
    
    # Rogers-Satchell component
    hc = np.log(np.maximum(high_p, 1e-8) / np.maximum(close_p, 1e-8))
    ho = np.log(np.maximum(high_p, 1e-8) / np.maximum(open_p, 1e-8))
    lc = np.log(np.maximum(low_p, 1e-8) / np.maximum(close_p, 1e-8))
    lo = np.log(np.maximum(low_p, 1e-8) / np.maximum(open_p, 1e-8))
    rs = hc * ho + lc * lo
    
    var = r_overnight ** 2 + k * (r_oc ** 2) + (1.0 - k) * rs
    return np.sqrt(np.maximum(var, 1e-12))


def daily_abs_return_vol(r: np.ndarray) -> np.ndarray:
    """Unsmoothed daily realized volatility proxy from daily returns.
    sigma_1d = |r_t| * sqrt(pi / 2).
    Unbiased for Gaussian returns: E[|r_t| * sqrt(pi/2)] = sigma_t.
    """
    scale = np.sqrt(np.pi / 2.0)  # ~1.2533
    return np.abs(r) * scale


def construct_vol_series(r: np.ndarray, window: int, df_raw: pd.DataFrame = None) -> np.ndarray:
    """Construct realized volatility series for a given window length.
    For window = 1:
      - Uses Garman-Klass or Yang-Zhang if OHLC columns exist in df_raw.
      - Falls back to daily absolute return |r_t| * sqrt(pi/2) if only close prices exist.
    For window > 1:
      - Rolling standard deviation over W trading days.
    """
    if window == 1:
        if df_raw is not None:
            cols = {c.lower(): c for c in df_raw.columns}
            if {"open", "high", "low", "close"}.issubset(cols.keys()):
                o = df_raw[cols["open"]].to_numpy(float)
                h = df_raw[cols["high"]].to_numpy(float)
                l = df_raw[cols["low"]].to_numpy(float)
                c = df_raw[cols["close"]].to_numpy(float)
                print("  [info] Computing intraday Garman-Klass realized volatility from OHLC quotes")
                return garman_klass_vol(o, h, l, c)
        return daily_abs_return_vol(r)
    else:
        return pd.Series(r).rolling(window).std().bfill().to_numpy(float)


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main():
    ensure_dirs()
    df = load_master()
    train, val, test = split_masks(df)
    dates = df.index
    T = len(df)
    
    r_all = df["r"].to_numpy(float)
    vix_all = df["vix"].to_numpy(float)
    r2_all = r_all ** 2
    
    windows = [1, 2, 3, 5, 10, 21]
    eq_rows = []
    comparison_rows = []
    forecast_rows = []
    
    print("=== Running Unsmoothed vs Smoothed Volatility Analysis ===")
    print(f"Sweeping smoothing windows: {windows} trading days ...")
    
    for W in windows:
        w_label = "Unsmoothed (W=1d)" if W == 1 else f"Smoothed (W={W}d)"
        sig_w = construct_vol_series(r_all, W, df_raw=df)
        X_w = np.column_stack([r_all, sig_w, vix_all])
        
        X_tr = X_w[train]
        X_va = X_w[val]
        X_te = X_w[test]
        
        # Standardize using Train statistics only
        Z_tr, mu, sd = zscore(X_tr)
        Z_va = apply_zscore(X_va, mu, sd)
        Z_full = apply_zscore(X_w, mu, sd)
        
        # When W=1 with |r|*sqrt(pi/2), sig_t is proportional to |r_t|.
        # In LIBRARY_TERMS: sig2 is collinear with r2, r_sig with r_absr, and sig_abssig with r2.
        # Excluding the redundant duplicates restores a well-conditioned design matrix (cond ~ 13-52).
        if W == 1:
            # Drop redundant duplicate terms: sig2 (idx 5), r_sig (idx 7), sig_abssig (idx 11)
            keep_idx = [0, 1, 2, 3, 4, 6, 8, 9, 10]
        else:
            keep_idx = list(range(len(LIBRARY_TERMS)))
            
        Th_tr = sc.build_library(Z_tr[:-1])[:, keep_idx]
        Th_va = sc.build_library(Z_va[:-1])[:, keep_idx]
        cond_tr = float(np.linalg.cond(Th_tr))
        
        path = sc.sweep_lambda(Th_tr, Z_tr[1:], Th_va, Z_va[1:])
        lam_star = sc.select_lambda(path, n_val=len(Z_va) - 1)
        
        Xi_stlsq_sub = sc.identify(Th_tr, Z_tr[1:], lam_star, solver="stlsq")
        Xi_sr3_sub = sc.identify(Th_tr, Z_tr[1:], lam_star, solver="sr3")
        
        # Map back to full LIBRARY_TERMS shape (12, 3)
        Xi_stlsq = np.zeros((len(LIBRARY_TERMS), 3))
        Xi_sr3 = np.zeros((len(LIBRARY_TERMS), 3))
        for sub_i, orig_i in enumerate(keep_idx):
            Xi_stlsq[orig_i] = Xi_stlsq_sub[sub_i]
            Xi_sr3[orig_i] = Xi_sr3_sub[sub_i]
            
        # Record equation coefficients for sigma_next
        sig_eq_stlsq = {LIBRARY_TERMS[i]: Xi_stlsq[i, 1] for i in range(len(LIBRARY_TERMS))}
        sig_eq_sr3 = {LIBRARY_TERMS[i]: Xi_sr3[i, 1] for i in range(len(LIBRARY_TERMS))}
        
        for i, term in enumerate(LIBRARY_TERMS):
            eq_rows.append({
                "window": W,
                "label": w_label,
                "term": term,
                "coef_stlsq": round(float(Xi_stlsq[i, 1]), 4),
                "active_stlsq": int(abs(Xi_stlsq[i, 1]) > 0),
                "coef_sr3": round(float(Xi_sr3[i, 1]), 4),
                "active_sr3": int(abs(Xi_sr3[i, 1]) > 0),
                "lambda_star": round(lam_star, 4),
            })
            
        comparison_rows.append({
            "window": W,
            "label": w_label,
            "lambda_star": round(lam_star, 4),
            "cond_number": round(cond_tr, 1),
            "sig_persistence (beta)": round(sig_eq_stlsq.get("sig", 0.0), 4),
            "r2_shock (alpha)": round(sig_eq_stlsq.get("r2", 0.0), 4),
            "r_absr_leverage (gamma)": round(sig_eq_stlsq.get("r_absr", 0.0), 4),
            "vix_implied": round(sig_eq_stlsq.get("vix", 0.0), 4),
            "const_omega": round(sig_eq_stlsq.get("1", 0.0), 4),
            "total_active_terms": int(sum(abs(v) > 0 for v in sig_eq_stlsq.values())),
            "GARCH_r2_discovered": "YES" if abs(sig_eq_stlsq.get("r2", 0.0)) > 0 else "NO",
            "Leverage_r_absr_discovered": "YES" if abs(sig_eq_stlsq.get("r_absr", 0.0)) > 0 else "NO",
        })
        
        # Test-set forecast evaluation for this window definition
        te_indices = np.where(test)[0]
        max_h = max(HORIZONS)
        origins = [t for t in te_indices if t + max_h < T]
        
        # Fit benchmark GARCH(1,1)
        r_tr = r_all[train]
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
            
        for h in HORIZONS:
            realized_h = np.array([sig_w[t + h] for t in origins])
            
            # SINDy STLSQ forecast (iterated forward map)
            f_sindy = []
            for t in origins:
                x0 = Z_full[t]
                p = sc.iterate_map(x0, Xi_stlsq, h)
                f_sindy.append(p[h - 1, 1] * sd[1] + mu[1])
            f_sindy = np.array(f_sindy)
            
            # GARCH forecast: causal rolling-window variance mapping
            # lo = t + h - W: window start day
            # real: sum of realized squared returns up to origin t
            # fsum: iterated forward conditional variance from t+1 to t+h
            f_garch = []
            for t in origins:
                s2j = s2_garch[t + 1]
                fsum = 0.0
                for _ in range(h):
                    fsum += s2j
                    s2j = omega + pers * s2j
                lo = t + h - W
                real = r2_all[max(lo, 0): t + 1].sum()
                f_garch.append(np.sqrt(max(0.0, real + fsum) / W))
            f_garch = np.array(f_garch)
            
            # EWMA forecast: causal rolling-window variance mapping
            f_ewma = []
            for t in origins:
                s2n = s2_ewma[t + 1] if t + 1 < T else s2_ewma[t]
                lo = t + h - W
                real = r2_all[max(lo, 0): t + 1].sum()
                f_ewma.append(np.sqrt(max(0.0, real + h * s2n) / W))
            f_ewma = np.array(f_ewma)
            
            # Naive RW forecast
            f_rw = np.array([sig_w[t] for t in origins])
            
            for m_name, f_vals in [("SINDy (STLSQ)", f_sindy),
                                   ("GARCH(1,1)", f_garch),
                                   ("EWMA (0.94)", f_ewma),
                                   ("Naive RW", f_rw)]:
                rmse, mae, ql = compute_metrics(realized_h, f_vals)
                forecast_rows.append({
                    "window": W,
                    "target_vol": w_label,
                    "model": m_name,
                    "h": h,
                    "RMSE": round(rmse, 6),
                    "MAE": round(mae, 6),
                    "QLIKE": round(ql, 6),
                })

    df_eq = pd.DataFrame(eq_rows)
    df_comp = pd.DataFrame(comparison_rows)
    df_fc = pd.DataFrame(forecast_rows)
    
    save_table(df_eq, "unsmoothed_vol_equations")
    save_table(df_comp, "unsmoothed_vs_smoothed_comparison")
    save_table(df_fc, "unsmoothed_vol_metrics")
    
    print("\n=== Mechanism Contrast: Unsmoothed (W=1) vs Smoothed (W=21) ===")
    print(df_comp[["window", "label", "cond_number", "sig_persistence (beta)",
                   "r2_shock (alpha)", "r_absr_leverage (gamma)", "vix_implied",
                   "total_active_terms", "GARCH_r2_discovered"]].to_string(index=False))
                   
    # --- Generate Figures ---
    # Fig 19: Term Evolution across Smoothing Windows
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    
    terms_to_plot = ["sig", "vix", "r2", "r_absr"]
    term_colors = {"sig": "darkblue", "vix": "steelblue", "r2": "firebrick", "r_absr": "darkgreen"}
    
    piv = df_eq.pivot(index="window", columns="term", values="coef_stlsq")
    for t in terms_to_plot:
        if t in piv.columns:
            ax1.plot(piv.index, piv[t], "o-", label=f"{t} coefficient",
                     color=term_colors.get(t, "gray"), lw=1.8, ms=5)
            
    ax1.set_xlabel("Volatility Smoothing Window W (trading days)")
    ax1.set_ylabel("Standardized SINDy Coefficient")
    ax1.set_title("How Window Smoothing Alters Discovered Dynamics")
    ax1.set_xticks(windows)
    ax1.axhline(0, color="gray", ls="--", lw=0.7)
    ax1.legend()
    
    # Active terms count vs W
    ax2.plot(df_comp["window"], df_comp["total_active_terms"], "s-", color="purple", lw=1.8, ms=6)
    ax2.set_xlabel("Volatility Smoothing Window W (trading days)")
    ax2.set_ylabel("Number of Active Terms in Sigma Equation")
    ax2.set_title("Equation Parsimony vs Smoothing Window")
    ax2.set_xticks(windows)
    save_fig(fig, "fig19_unsmoothed_term_evolution")
    
    # Fig 20: Forecast Metrics for Unsmoothed Volatility (W=1, h=1)
    fc_w1 = df_fc[(df_fc["window"] == 1) & (df_fc["h"] == 1)].sort_values("QLIKE")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(fc_w1["model"], fc_w1["QLIKE"],
           color=["firebrick" if "SINDy" in m else "steelblue" for m in fc_w1["model"]],
           alpha=0.85)
    ax.set_ylabel("QLIKE Loss (lower = better)")
    ax.set_title("One-Day-Ahead Forecast Performance on Unsmoothed Daily Volatility (W=1)")
    ax.tick_params(axis="x", rotation=25)
    save_fig(fig, "fig20_unsmoothed_forecast_bars")
    
    print("\nUnsmoothed volatility evaluation complete. All tables and figures written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
