"""Phase 5 — ML baselines: analysis pass on train+validation.

The walk-forward forecasting lives in 10_horse_race (same model classes);
this script answers 'which factors does ML think matter?' and contrasts it
with the terms SINDy kept (RQ5 bridge).
"""
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sindy_core as sc
from models import RidgeLib, LassoLib, RandomForestModel, XGBModel, HAS_SKLEARN, HAS_XGB
from config import LIBRARY_TERMS
from data_utils import load_master, save_table, save_fig, split_masks

RAW_FEATURES = ["r_t", "sig_t", "vix_t", "|r_t|", "r_t^2",
                "r_lag1", "sig_lag1", "vix_lag1"]


def main():
    if not HAS_SKLEARN:
        print("sklearn unavailable — skipping ML analysis")
        return 0
    df = load_master()
    train, val, test = split_masks(df)
    n_tv = int((train | val).sum())

    ridge = RidgeLib(df).fit(n_tv)
    lasso = LassoLib(df).fit(n_tv)
    rf = RandomForestModel(df).fit(n_tv)

    tab = pd.DataFrame({"term": LIBRARY_TERMS,
                        "lasso_coef_h1": lasso.models[1].coef_,
                        "ridge_coef_h1": ridge.models[1].coef_,
                        }).round(5)
    try:
        sindy_eq = pd.read_csv("results/tables/sindy_equations_stlsq.csv")
        tab["sindy_sig_eq"] = sindy_eq["sig_next"].values
    except Exception:
        pass
    save_table(tab, "ml_vs_sindy_terms")
    print(tab.to_string(index=False))

    rows = []
    for i in range(len(LIBRARY_TERMS)):
        rows.append({"model": "lasso (library)", "kind": "library term",
                     "feature": LIBRARY_TERMS[i],
                     "importance": abs(lasso.models[1].coef_[i])})
    for i, f in enumerate(RAW_FEATURES):
        rows.append({"model": "random forest", "kind": "raw feature",
                     "feature": f,
                     "importance": rf.models[1].feature_importances_[i]})
    if HAS_XGB:
        xgb = XGBModel(df).fit(n_tv)
        for i, f in enumerate(RAW_FEATURES):
            rows.append({"model": "xgboost", "kind": "raw feature",
                         "feature": f,
                         "importance": float(xgb.models[1].feature_importances_[i])})
    imp = pd.DataFrame(rows).sort_values("importance", ascending=False)
    save_table(imp.head(20), "ml_feature_importance")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    sub = imp.head(12)
    colors = {"library term": "firebrick", "raw feature": "steelblue"}
    ax.barh(range(len(sub)), sub["importance"],
            color=[colors[k] for k in sub["kind"]])
    ax.set_yticks(range(len(sub)), sub["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("importance (|coef| for lasso; impurity gain for trees)")
    ax.set_title("Which factors does ML think matter? (h=1 sigma forecast)")
    save_fig(fig, "fig10_ml_feature_importance")
    print("ML importance table + figure written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
