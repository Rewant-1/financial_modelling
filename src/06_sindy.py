"""Phase 3 — first SINDy identification: THE core discovery step.

Discrete-time SINDy on the standardized state (r, sigma, vix):
    x_{t+1} = Theta(x_t) @ Xi

Sweep lambda -> sparsity path; select on the frozen validation block
(2016-2019, never test); write the three discovered equations; cross-check
with SR3; build the term-by-term comparison against GARCH(1,1)/GJR.
"""
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sindy_core as sc
from config import LIBRARY_TERMS
from data_utils import load_master, save_table, save_fig, split_masks, zscore


def main():
    df = load_master()
    train, val, test = split_masks(df)
    X_tr = df.loc[train, ["r", "sigma", "vix"]].to_numpy(float)
    X_va = df.loc[val, ["r", "sigma", "vix"]].to_numpy(float)
    Z_tr, mu, sd = zscore(X_tr)
    Z_va = (X_va - mu) / sd

    Theta_tr, Theta_va = sc.build_library(Z_tr[:-1]), sc.build_library(Z_va[:-1])
    path = sc.sweep_lambda(Theta_tr, Z_tr[1:], Theta_va, Z_va[1:])
    lam_star = sc.select_lambda(path, n_val=len(Z_va) - 1)
    print(f"lambda* = {lam_star:.4f} (validation one-step RMSE minimum)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.plot(path["lambda"], path["n_terms"], "o-", ms=3, color="steelblue")
    ax1.axvline(lam_star, color="firebrick", ls="--", lw=1,
                label=f"lambda* = {lam_star:.3f}")
    ax1.set_xscale("log")
    ax1.set_xlabel("threshold lambda")
    ax1.set_ylabel("active terms (all equations)")
    ax1.set_title("Sparsity path")
    ax1.legend()
    ax2.plot(path["lambda"], path["train_rmse"], "-", color="steelblue",
             label="train")
    ax2.plot(path["lambda"], path["val_rmse"], "-", color="firebrick",
             label="validation")
    ax2.set_xscale("log")
    ax2.set_xlabel("threshold lambda")
    ax2.set_ylabel("one-step RMSE (standardized)")
    ax2.set_title("Parsimony vs accuracy")
    ax2.legend()
    save_fig(fig, "fig06_sparsity_path")
    path.round(5).to_csv("results/tables/sparsity_path.csv", index=False)

    Xi = sc.identify(Theta_tr, Z_tr[1:], lam_star)
    eq = sc.equation_table(Xi)
    save_table(eq, "sindy_equations_stlsq")
    print("\n=== Discovered system (STLSQ + significance pruning, standardized space) ===")
    for line in sc.format_equations(Xi):
        print(" ", line)

    Xi_sr3 = sc.identify(Theta_tr, Z_tr[1:], lam_star, solver="sr3")
    eq_sr3 = sc.equation_table(Xi_sr3)
    save_table(eq_sr3, "sindy_equations_sr3")
    print("\n=== Same terms under SR3? ===")
    for line in sc.format_equations(Xi_sr3):
        print(" ", line)

    # term-by-term overlap between solvers
    overlap = pd.DataFrame({
        "term": LIBRARY_TERMS,
        "stlsq_sig_eq": Xi[:, 1].round(4),
        "sr3_sig_eq": Xi_sr3[:, 1].round(4),
    })
    save_table(overlap, "solver_agreement_sigma_eq")

    # --- comparison vs GARCH: does SINDy rediscover the econometrician's form?
    try:
        gp = pd.read_csv("results/tables/garch_params.csv")
        g11 = gp[gp["model"] == "GARCH(1,1)"].iloc[0]
        gjr = gp[gp["model"] == "GJR-GARCH(1,1)"].iloc[0]
    except Exception:
        g11 = gjr = None
    rows = []
    analogues = [
        ("1", "omega (intercept)",
         float(g11["omega"]) if g11 is not None else np.nan),
        ("r2", "alpha (shark shock term)",
         float(g11["alpha"]) if g11 is not None else np.nan),
        ("sig", "beta (vol memory)",
         float(g11["beta"]) if g11 is not None else np.nan),
        ("r_absr", "gamma (GJR asymmetry)",
         float(gjr["gamma"]) if gjr is not None else np.nan),
        ("vix", "(no GARCH analogue)", np.nan),
    ]
    for term, analogue, gval in analogues:
        i = LIBRARY_TERMS.index(term)
        rows.append({
            "SINDy_term": term,
            "SINDy_coef_std": round(Xi[i, 1], 4),
            "discovered": "yes" if abs(Xi[i, 1]) > 0 else "no",
            "GARCH_analogue": analogue,
            "GARCH_value": gval,
        })
    comp = pd.DataFrame(rows)
    save_table(comp, "sindy_vs_garch")
    print("\n=== SINDy sigma-equation vs GARCH analogues ===")
    print(comp.to_string(index=False))

    np.save("results/tables/Xi_stlsq.npy", Xi)
    np.save("results/tables/Xi_sr3.npy", Xi_sr3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
