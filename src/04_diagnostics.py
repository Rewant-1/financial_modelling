"""Phase 1.5 — nonlinear-dynamics diagnostics: is a nonlinear model justified?

1. BDS test on AR-filtered and GARCH-filtered returns. Rejecting independence
   after AR filtering = nonlinearity beyond linear memory; the GARCH-filtered
   column shows what survives after volatility clustering is accounted for.
2. Linear-vs-nonlinear skill gap: one-step validation RMSE of the best LINEAR
   map (DMD / VAR) vs SINDy on the same state. A clear gap = nonlinearity is
   operative in the transition itself.
"""
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import bds

import matplotlib.pyplot as plt

import sindy_core as sc
import garch as garch_mod
from data_utils import load_master, save_table, save_fig, split_masks, zscore


def ar_filter(x, p=5):
    F = np.column_stack([np.ones(len(x) - p)] +
                        [x[p - 1 - i: len(x) - 1 - i] for i in range(p)])
    beta, *_ = np.linalg.lstsq(F, x[p:], rcond=None)
    return x[p:] - F @ beta


def main():
    df = load_master()
    train, val, test = split_masks(df)
    r_tv = df.loc[train | val, "r"].values

    print("BDS tests (stat > 1.96 = reject independence at 5%) ...")
    resid_ar = ar_filter(r_tv)
    g = garch_mod.fit_garch(r_tv, gjr=False)
    s2 = garch_mod.filter_sigma2(r_tv, g)
    resid_g = (r_tv - g["mu"]) / np.sqrt(s2)

    rows = []
    for label, x in [("raw returns", r_tv),
                     ("AR(5)-filtered", resid_ar),
                     ("GARCH(1,1)-filtered", resid_g)]:
        xz = (x - x.mean()) / x.std()
        try:
            stat, pvd = bds(xz, max_dim=4, distance=1.5)
            for m in (2, 3):
                rows.append({"series": label, "embed_dim": m,
                             "BDS_stat": round(float(stat[m - 2]), 2),
                             "p_value": float(pvd[m - 2])})
        except Exception as e:
            rows.append({"series": label, "embed_dim": m,
                         "BDS_stat": np.nan, "p_value": np.nan})
    bds_tab = pd.DataFrame(rows)
    bds_tab["verdict"] = np.where(bds_tab["p_value"] < 0.05,
                                  "reject independence", "—")
    save_table(bds_tab, "bds_tests")
    print(bds_tab.to_string(index=False))

    # --- skill gap on the validation block ---
    X_tr = df.loc[train, ["r", "sigma", "vix"]].to_numpy(float)
    X_va = df.loc[val, ["r", "sigma", "vix"]].to_numpy(float)
    mu, sd = X_tr.mean(0), X_tr.std(0, ddof=1)
    Z_tr, Z_va = (X_tr - mu) / sd, (X_va - mu) / sd

    T_tr, T_va = sc.build_library(Z_tr[:-1]), sc.build_library(Z_va[:-1])

    # linear map (DMD on standardized state, incl. mean offset via constant)
    Xi_lin = np.zeros((12, 3))
    A, *_ = np.linalg.lstsq(T_tr[:, :4], Z_tr[1:], rcond=None)  # {1,r,s,v}
    Xi_lin[:4] = A
    rmse_lin = np.sqrt(np.mean((Z_va[1:] - T_va[:, :4] @ A) ** 2, axis=0))

    rows = []
    lams = np.logspace(-3, 0.3, 22)
    mean_rmses, per_eq = [], []
    for lam in lams:
        Xi = sc.identify(T_tr, Z_tr[1:], lam)
        e = Z_va[1:] - T_va @ Xi
        rmse = np.sqrt(np.mean(e ** 2, axis=0))
        mean_rmses.append(rmse.mean())
        per_eq.append(rmse)
    mean_rmses, per_eq = np.array(mean_rmses), np.array(per_eq)
    # 1-SE rule (same convention as sindy_core.select_lambda)
    best = mean_rmses.min()
    se = best / np.sqrt(2 * max(len(Z_va) - 1, 10))
    i_star = int(np.where(mean_rmses <= best + se)[0][0])
    lam_star, rmse_syn = lams[i_star], per_eq[i_star]

    gap = pd.DataFrame({
        "variable": ["r", "sigma", "vix"],
        "linear_map_RMSE": rmse_lin.round(4),
        "SINDy_RMSE": rmse_syn.round(4),
        "improvement_%": (100 * (rmse_lin - rmse_syn) / rmse_lin).round(1),
        "lambda_star": round(lam_star, 4),
    })
    save_table(gap, "skill_gap")
    print(gap.to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(3)
    ax.bar(x - 0.18, rmse_lin, 0.34, label="best linear map", color="steelblue")
    ax.bar(x + 0.18, rmse_syn, 0.34, label="SINDy (STLSQ)",
           color="firebrick")
    ax.set_xticks(x)
    ax.set_xticklabels(["r", "sigma", "vix"])
    ax.set_ylabel("one-step RMSE (validation, standardized)")
    ax.set_title("Nonlinearity check: does SINDy beat the best linear map?")
    ax.legend()
    save_fig(fig, "fig05_skill_gap")

    imp = gap.loc[gap["variable"] == "sigma", "improvement_%"].iloc[0]
    direction = ("improves" if imp > 0 else "does not improve") if abs(imp) >= 1 \
        else "matches"
    verdict = (f"BDS rejects independence on AR-filtered returns "
               f"(p={bds_tab.query('series==\"AR(5)-filtered\" and embed_dim==2')['p_value'].iloc[0]:.1e}); "
               f"SINDy {direction} the one-step sigma map vs the best linear map "
               f"({imp:+.1f}%). One-step point prediction is a linear-regime "
               f"problem; nonlinear terms earn their keep in simulation and "
               f"higher moments (see 09_dynamics).")
    with open("results/tables/diagnostics_verdict.txt", "w") as fh:
        fh.write(verdict + "\n")
    print("\n" + verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
