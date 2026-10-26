"""Phase 7 — dynamics, simulation and stability: use the EQUATIONS in ways a
forecast table cannot.

1. Fixed point + Jacobian stability of the discovered map (stationarity).
2. Sigma-map phase portrait (the cobweb/mean-reversion picture).
3. SINDy vs DMD eigenvalues on the complex plane.
4. Simulated moments (kurtosis, ACF|r|, sigma distribution) vs actual — does
   the discovered system REPRODUCE the stylized facts?
5. COVID episode replay: simulate forward from 2020-02-21 vs what happened.
"""
import os
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.optimize import fsolve

import sindy_core as sc
from data_utils import load_master, save_table, save_fig, split_masks, zscore

FAST = os.environ.get("FAST", "0") == "1"


def moments(r, s):
    from scipy import stats as st
    return {
        "kurtosis_r": st.kurtosis(r),
        "skew_r": st.skew(r),
        "acf_absr_lag1": np.corrcoef(np.abs(r[:-1]), np.abs(r[1:]))[0, 1],
        "acf_absr_lag5": np.corrcoef(np.abs(r[:-5]), np.abs(r[5:]))[0, 1],
        "acf_absr_lag21": np.corrcoef(np.abs(r[:-21]), np.abs(r[21:]))[0, 1],
        "sigma_mean": np.mean(s), "sigma_p95": np.quantile(s, 0.95),
    }


def main():
    df = load_master()
    train, val, test = split_masks(df)
    X = df[["r", "sigma", "vix"]].to_numpy(float)
    n_tv = int((train | val).sum())
    Z, mu, sd = zscore(X[:n_tv])

    Theta = sc.build_library(Z[:-1])
    # lambda* from frozen validation (as in 06) — recompute quickly
    X_va = df.loc[val, ["r", "sigma", "vix"]].to_numpy(float)
    Z_va = (X_va - mu) / sd
    p2 = sc.sweep_lambda(Theta, Z[1:], sc.build_library(Z_va[:-1]), Z_va[1:])
    lam = sc.select_lambda(p2, n_val=len(Z_va) - 1)
    Xi = sc.identify(Theta, Z[1:], lam)
    print(f"discovered map at lambda*={lam:.4f}: "
          f"{int((np.abs(Xi) > 0).sum())} active terms")

    def build_row(z):
        return sc.build_library(np.atleast_2d(z))

    g = lambda z: (build_row(z) @ Xi - z)[0]

    x0 = np.zeros(3)
    xstar = fsolve(g, x0)
    jac = np.zeros((3, 3))
    eps = 1e-6
    for i in range(3):
        e = np.zeros(3); e[i] = eps
        jac[:, i] = ((build_row(xstar + e) @ Xi - (build_row(xstar - e) @ Xi))
                     / (2 * eps))[0]
    eig = np.linalg.eigvals(jac)
    xstar_raw = xstar * sd + mu

    stab = pd.DataFrame({
        "quantity": ["fixed point r", "fixed point sigma (daily)",
                     "fixed point sigma (annualized, %)",
                     "fixed point vix (daily)",
                     "max |eigenvalue| of Jacobian",
                     "verdict"],
        "value": [f"{xstar_raw[0]:.5f}", f"{xstar_raw[1]:.5f}",
                  f"{xstar_raw[1] * np.sqrt(252) * 100:.1f}%",
                  f"{xstar_raw[2]:.5f}",
                  f"{np.max(np.abs(eig)):.4f}",
                  "locally stable (mean-reverting)" if np.max(np.abs(eig)) < 1
                  else "UNSTABLE — investigate"],
    })
    save_table(stab, "fixed_point_stability")
    print(stab.to_string(index=False))

    # --- sigma-map phase portrait at r, vix = means ---
    grid = np.linspace(-1.5, 4.0, 200)
    zs = np.column_stack([np.zeros_like(grid), grid, np.zeros_like(grid)])
    sig_next = (build_row(zs) @ Xi)[:, 1]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(grid, sig_next, color="firebrick", lw=1.6, label="discovered sigma map")
    ax.plot(grid, grid, "k--", lw=0.8, label="45-degree line")
    fp = xstar[1]
    ax.plot(fp, fp, "o", ms=9, mfc="none", mec="blue", mew=1.5,
            label=f"fixed point (sigma = {xstar_raw[1]*np.sqrt(252)*100:.0f}% ann.)")
    s = 3.8
    ax.plot([s], [s], "o", ms=5, color="gray")
    ax.annotate("start: COVID-like vol", (s, s), (s - 1.4, s + 0.35),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.7))
    for _ in range(6):
        s_next = float((build_row(np.array([0, s, 0])) @ Xi)[0, 1])
        ax.plot([s, s], [s, s_next], color="gray", lw=1.1, alpha=0.9)
        ax.plot([s, s_next], [s_next, s_next], color="gray", lw=1.1, alpha=0.9)
        s = s_next
    ax.set_xlabel("sigma_t (standardized)")
    ax.set_ylabel("sigma_next (standardized)")
    ax.set_title("Volatility mean-reversion: cobweb back to the fixed point")
    ax.legend(fontsize=8)
    save_fig(fig, "fig11_phase_portrait")

    # --- eigenvalues: SINDy Jacobian vs DMD operator ---
    Zc = Z - Z.mean(0)
    A_dmd, *_ = np.linalg.lstsq(Zc[:-1], Zc[1:], rcond=None)
    eig_dmd = np.linalg.eigvals(A_dmd)
    fig, ax = plt.subplots(figsize=(5, 4))
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(th), np.sin(th), "k-", lw=0.8)
    ax.scatter(eig.real, eig.imag, s=45, color="firebrick",
               label="SINDy Jacobian at fixed point", zorder=3)
    ax.scatter(eig_dmd.real, eig_dmd.imag, s=30, color="steelblue",
               marker="s", label="DMD operator", zorder=3)
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Re"); ax.set_ylabel("Im")
    ax.set_title("Linearized dynamics: inside the unit circle = stationary")
    ax.legend(fontsize=8)
    save_fig(fig, "fig12_eigenvalues")

    # --- simulated moments: does the discovered system reproduce stylized facts?
    resid = Z[1:] - Theta @ Xi
    rng = np.random.default_rng(42)
    n_paths = 50 if FAST else 200
    burn = 500
    T = 20000
    stats_runs = []
    for _ in range(n_paths):
        x = Z[-1].copy()
        rs, ss = [], []
        blk = 21
        n_blk = T // blk + 2
        idx = rng.integers(0, len(resid) - blk, n_blk)
        noise = np.concatenate([resid[i: i + blk] for i in idx])[:T]
        for t in range(T):
            x = (build_row(x) @ Xi)[0] + noise[t]
            rs.append(x[0]); ss.append(x[1])
        rs, ss = np.array(rs[burn:]), np.array(ss[burn:])
        stats_runs.append(moments(rs * sd[0] + mu[0], ss * sd[1] + mu[1]))
    sim = pd.DataFrame(stats_runs).mean()

    act = moments(Z[burn:, 0] * sd[0] + mu[0], Z[burn:, 1] * sd[1] + mu[1])
    comp = pd.DataFrame({"moment": sim.index,
                         "simulated": sim.values.round(5),
                         "actual (fit window)": [round(act[k], 5)
                                                 for k in sim.index]})
    save_table(comp, "simulated_vs_actual_moments")
    print(comp.to_string(index=False))

    # --- COVID replay ---
    t0 = df.index.get_loc(pd.Timestamp("2020-02-21"))
    actual = df["sigma"].iloc[t0: t0 + 60].values * 100
    paths = []
    for _ in range(n_paths):
        x = (X[t0] - mu) / sd
        out = []
        for t in range(60):
            x = (build_row(x) @ Xi)[0] + resid[rng.integers(len(resid))]
            out.append(x[1] * sd[1] + mu[1])
        paths.append(out)
    paths = np.array(paths) * 100
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(actual, color="black", lw=1.4, label="actual sigma")
    ax.plot(np.median(paths, axis=0), color="firebrick", lw=1.4,
            label="simulated (median)")
    ax.fill_between(range(60), np.quantile(paths, 0.05, axis=0),
                    np.quantile(paths, 0.95, axis=0), color="firebrick",
                    alpha=0.2, label="5-95% band")
    ax.set_xlabel("trading days from 2020-02-21")
    ax.set_ylabel("sigma (daily %)")
    ax.set_title("Replaying COVID from the discovered equations")
    ax.legend()
    save_fig(fig, "fig13_covid_replay")
    return 0


if __name__ == "__main__":
    sys.exit(main())
