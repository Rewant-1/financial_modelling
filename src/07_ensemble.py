"""Phase 4 — robust identification: which discovered terms are real?

1. Ensemble-SINDy (block bootstrap) -> term inclusion probabilities.
2. Regime subsamples: which terms persist across 1990s / dot-com / GFC /
   2010s / COVID+ (the last includes the test window: structural analysis
   only — never used to pick forecasting models).
3. Noise injection: re-identify under 0/5/10/20% added noise, record when the
   structure breaks.
"""
import os
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sindy_core as sc
from config import LIBRARY_TERMS, REGIMES
from data_utils import load_master, save_table, save_fig, split_masks, zscore

FAST = os.environ.get("FAST", "0") == "1"


def main():
    df = load_master()
    train, val, test = split_masks(df)
    X_tr = df.loc[train, ["r", "sigma", "vix"]].to_numpy(float)
    X_va = df.loc[val, ["r", "sigma", "vix"]].to_numpy(float)
    Z_tr, mu, sd = zscore(X_tr)
    Z_va = (X_va - mu) / sd

    Theta_tr, Theta_va = sc.build_library(Z_tr[:-1]), sc.build_library(Z_va[:-1])
    path = sc.sweep_lambda(Theta_tr, Z_tr[1:], Theta_va, Z_va[1:])
    lam = sc.select_lambda(path, n_val=len(Z_va) - 1)
    n_boot = 200 if FAST else 500
    print(f"ensemble-SINDy at lambda*={lam:.4f}, {n_boot} block-bootstrap fits ...")
    prob, med = sc.ensemble_inclusion(Z_tr, lam, n_boot=n_boot, seed=42)

    eq_names = ["r_eq", "sig_eq", "vix_eq"]
    prob_df = pd.DataFrame(prob, index=LIBRARY_TERMS, columns=eq_names).round(3)
    med_df = pd.DataFrame(med, index=LIBRARY_TERMS, columns=eq_names).round(4)
    save_table(prob_df, "inclusion_probabilities")
    save_table(med_df, "ensemble_median_coefs")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(prob, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3), eq_names)
    ax.set_yticks(range(len(LIBRARY_TERMS)), LIBRARY_TERMS)
    for i in range(len(LIBRARY_TERMS)):
        for j in range(3):
            ax.text(j, i, f"{prob[i, j]:.2f}", ha="center", va="center",
                    fontsize=8)
    ax.grid(False)
    fig.colorbar(im, label="P(term included)")
    ax.set_title(f"Ensemble-SINDy inclusion probabilities "
                 f"({n_boot} block-bootstrap fits)")
    save_fig(fig, "fig07_inclusion_heatmap")

    # --- regime stability ---
    rows = []
    for name, (a, b) in REGIMES.items():
        sub = df.loc[(df.index >= a) & (df.index <= b),
                     ["r", "sigma", "vix"]].to_numpy(float)
        if len(sub) < 300:
            continue
        Zr, _, _ = zscore(sub)
        Th = sc.build_library(Zr[:-1])
        Xi = sc.identify(Th, Zr[1:], lam)
        rows.append({"term": LIBRARY_TERMS, "regime": name,
                     "coef": Xi[:, 1].round(3), "active_sig_eq":
                     (np.abs(Xi[:, 1]) > 0).astype(int)})
    reg = pd.DataFrame({
        "term": LIBRARY_TERMS,
        **{r["regime"]: r["active_sig_eq"] for r in rows},
    })
    reg["n_regimes_active"] = reg[[r["regime"] for r in rows]].sum(axis=1)
    reg = reg.sort_values("n_regimes_active", ascending=False)
    save_table(reg, "regime_stability")
    print(reg.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    M = reg[[r["regime"] for r in rows]].to_numpy(float)
    im = ax.imshow(M.T, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(rows)), [r["regime"] for r in rows])
    ax.set_xticks(range(len(LIBRARY_TERMS)), LIBRARY_TERMS, rotation=60)
    for j in range(M.shape[0]):
        for i in range(M.shape[1]):
            ax.text(i, j, int(M[j, i]), ha="center", va="center", fontsize=8)
    ax.grid(False)
    ax.set_title("Sigma-equation term stability across regimes "
                 "(1 = term active)")
    save_fig(fig, "fig08_regime_stability")

    # --- noise injection ---
    rng = np.random.default_rng(7)
    noise_rows = []
    for pct in (0, 5, 10, 20):
        Zn = Z_tr + rng.normal(0, pct / 100.0, Z_tr.shape)
        Thn = sc.build_library(Zn[:-1])
        Xi = sc.identify(Thn, Zn[1:], lam)
        act = np.abs(Xi[:, 1]) > 0
        noise_rows.append({
            "noise_pct": pct,
            "n_terms_sig_eq": int(act.sum()),
            **{f"coef_{t}": round(Xi[i, 1], 4)
               for i, t in enumerate(LIBRARY_TERMS) if t in
               ("1", "r2", "sig", "r_absr", "vix", "sig2")},
        })
    noise = pd.DataFrame(noise_rows)
    save_table(noise, "noise_robustness")
    print(noise.to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(noise["noise_pct"], noise["n_terms_sig_eq"], "o-",
            color="firebrick")
    ax.set_xlabel("added noise (% of std)")
    ax.set_ylabel("active terms in sigma equation")
    ax.set_title("Structure degradation under noise injection")
    save_fig(fig, "fig09_noise_robustness")
    return 0


if __name__ == "__main__":
    sys.exit(main())
