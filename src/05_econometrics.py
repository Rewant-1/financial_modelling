"""Phase 2 — econometric baselines (Python side; mirrors the STATA do-files).

Fits GARCH(1,1), GJR-GARCH(1,1), AR on sigma, VAR on the state, all on the
train+validation window only. Exports coefficient tables for the report and
saves GARCH parameters where 06_sindy reads them for the term-by-term
'SINDy vs GARCH' comparison. STATA equivalents live in stata/*.do.
"""
import sys
import numpy as np
import pandas as pd

import garch as garch_mod
from data_utils import load_master, save_table, split_masks
from config import TABLES


def main():
    df = load_master()
    train, val, test = split_masks(df)
    r_tv = df.loc[train | val, "r"].values

    print("GARCH(1,1) on returns (train+val) ...")
    g11 = garch_mod.fit_garch(r_tv, gjr=False)
    print(f"  mu={g11['mu']:.5f} omega={g11['omega']:.3e} "
          f"alpha={g11['alpha']:.4f} beta={g11['beta']:.4f} "
          f"persistence={g11['persistence']:.4f}")
    print("GJR-GARCH(1,1) on returns (train+val) ...")
    gjr = garch_mod.fit_garch(r_tv, gjr=True)
    print(f"  alpha={gjr['alpha']:.4f} gamma={gjr['gamma']:.4f} "
          f"beta={gjr['beta']:.4f} persistence={gjr['persistence']:.4f}")

    params = pd.DataFrame([
        {"model": "GARCH(1,1)", **{k: g11[k] for k in
          ("mu", "omega", "alpha", "beta", "gamma", "persistence")}},
        {"model": "GJR-GARCH(1,1)", **{k: gjr[k] for k in
          ("mu", "omega", "alpha", "beta", "gamma", "persistence")}},
    ]).round(6)
    save_table(params, "garch_params")

    # half-life of a volatility shock under each model
    hl = pd.DataFrame({
        "model": ["GARCH(1,1)", "GJR-GARCH(1,1)"],
        "persistence": [g11["persistence"], gjr["persistence"]],
        "half_life_days": [np.log(0.5) / np.log(g11["persistence"]),
                           np.log(0.5) / np.log(gjr["persistence"])],
    }).round(2)
    save_table(hl, "garch_halflife")

    # AR on sigma and VAR: report selections (models reused in horse race)
    from models import ARSigma, VARModel
    n_tv = int((train | val).sum())
    ar = ARSigma(df).fit(n_tv)
    var = VARModel(df).fit(n_tv)
    sel = pd.DataFrame([
        {"model": "AR(sigma)", "selection": f"p = {ar.p} (BIC)"},
        {"model": "VAR(r, sigma, vix)", "selection": f"p = {var.p} (BIC)"},
    ])
    save_table(sel, "econometric_selections")
    print(sel.to_string(index=False))
    print("GARCH coefficient tables -> results/tables/garch_params.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
