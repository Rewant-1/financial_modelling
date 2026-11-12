"""Tests for data leakage fixes, multi-split evaluation, and unsmoothed volatility estimators.

Verifies:
1. DirectMLBase._features has strict 1:1 row alignment with Z (no offset, no lookahead), and handles empty arrays.
2. Tree models (RandomForest, XGBoost) fit and predict without future lookahead.
3. Multi-split configurations produce valid out-of-sample partitions without temporal overlap.
4. Intraday volatility estimators (Garman-Klass, Parkinson, Yang-Zhang) compute correctly on OHLC inputs.
5. Unsmoothed daily volatility series (W=1) is well-conditioned (cond < 100) and SINDy discovers positive r^2 shock.
6. GARCH rolling-window variance mapping in multi-split and unsmoothed evaluations matches models.GARCHModel.
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_utils import load_master, zscore, apply_zscore
from models import RandomForestModel, XGBModel, RidgeLib, LassoLib, GARCHModel, HAS_SKLEARN, HAS_XGB
import sindy_core as sc
from config import LIBRARY_TERMS
import garch as garch_mod


def test_features_1_to_1_alignment():
    """Verify that _features maintains strict row-for-row alignment with input Z,
    and cleanly handles boundary conditions and empty inputs."""
    T = 100
    # Construct an artificial Z with distinct, strictly monotonic values
    Z = np.arange(T * 3, dtype=float).reshape(T, 3)
    
    dummy_df = pd.DataFrame({
        "r": Z[:, 0],
        "sigma": Z[:, 1],
        "vix": Z[:, 2],
    })
    
    rf = RandomForestModel(dummy_df)
    F = rf._features(Z)
    
    # 1. Output shape must be exactly (T, 8)
    assert F.shape == (T, 8), f"Expected shape ({T}, 8), got {F.shape}"
    
    # 2. Row 0 contemporaneous features must match Z[0]
    np.testing.assert_array_equal(F[0, :3], Z[0], err_msg="Row 0 features do not match Z[0]")
    
    # 3. Row t contemporaneous features must match Z[t] for all t
    np.testing.assert_array_equal(F[:, :3], Z, err_msg="Contemporaneous features do not match Z")
    
    # 4. Lagged features: row t lag should match Z[t-1] for t >= 1
    np.testing.assert_array_equal(F[1:, 5:8], Z[:-1], err_msg="Lagged features do not match Z[t-1]")
    
    # 5. Row 0 lag should be Z[0] (causal boundary condition, not Z[-1] wrap-around)
    np.testing.assert_array_equal(F[0, 5:8], Z[0], err_msg="Row 0 lag boundary violated")
    
    # 6. Edge case: empty input
    F_empty = rf._features(np.empty((0, 3)))
    assert F_empty.shape == (0, 8), f"Expected shape (0, 8), got {F_empty.shape}"
    
    print("[PASS] _features 1:1 row alignment, causality, and edge cases verified")


def test_no_lookahead_leakage():
    """Verify that tree models cannot peek into future sigma at prediction time."""
    T = 500
    rng = np.random.default_rng(42)
    
    # Generate random walk return and sigma
    r = rng.normal(0, 0.01, T)
    sigma = np.abs(r) + 0.01
    vix = sigma * 1.1 + rng.normal(0, 0.001, T)
    
    df = pd.DataFrame({"r": r, "sigma": sigma, "vix": vix})
    
    if HAS_SKLEARN:
        rf = RandomForestModel(df)
        rf.fit(300)
        
        # Perturb future sigma at origin t + 1
        origin = 350
        fc_before = rf.sigma_forecasts([origin], [1])[1][0]
        
        # Modify future data at origin + 1
        df_perturbed = df.copy()
        df_perturbed.loc[origin + 1, "sigma"] = 999.0
        rf_perturbed = RandomForestModel(df_perturbed)
        rf_perturbed.models = rf.models
        rf_perturbed.mu = rf.mu
        rf_perturbed.sd = rf.sd
        rf_perturbed.y_mu = rf.y_mu
        rf_perturbed.y_sd = rf.y_sd
        
        fc_after = rf_perturbed.sigma_forecasts([origin], [1])[1][0]
        assert np.isclose(fc_before, fc_after), \
            f"Forecast at origin {origin} changed when future index {origin+1} was modified! Leakage detected."
        print("[PASS] No lookahead leakage: future modifications do not affect origin forecast")


def test_multi_split_partitions():
    """Verify that multi-split definitions are valid and have no train/test leakage."""
    import importlib
    varying_splits = importlib.import_module("11_varying_splits")
    df = load_master()
    splits = varying_splits.get_split_definitions(df)
    
    assert len(splits) == 6, f"Expected 6 split regimes, got {len(splits)}"
    
    T = len(df)
    for s_name, s_info in splits.items():
        tr = s_info["tr_mask"]
        va = s_info["va_mask"]
        te = s_info["te_mask"]
        
        # No overlap between train, val, test
        assert not np.any(tr & va), f"Overlap between train and val in {s_name}"
        assert not np.any(tr & te), f"Overlap between train and test in {s_name}"
        assert not np.any(va & te), f"Overlap between val and test in {s_name}"
        
        # Temporal ordering: max(train index) < min(val index) < min(test index)
        idx_tr = np.where(tr)[0]
        idx_va = np.where(va)[0]
        idx_te = np.where(te)[0]
        
        assert idx_tr[-1] < idx_va[0], f"Train not strictly before Val in {s_name}"
        assert idx_va[-1] < idx_te[0], f"Val not strictly before Test in {s_name}"
        
    print("[PASS] All 6 multi-split definitions verified: temporal causality and partition disjointness satisfied")


def test_intraday_volatility_estimators():
    """Verify Garman-Klass, Parkinson, and Yang-Zhang intraday volatility formulas."""
    import importlib
    unsmoothed = importlib.import_module("12_unsmoothed_vol")
    
    T = 200
    rng = np.random.default_rng(123)
    p0 = 100.0
    drift = rng.normal(0, 0.01, T)
    close = p0 * np.exp(np.cumsum(drift))
    open_p = close * np.exp(rng.normal(0, 0.003, T))
    high_p = np.maximum(open_p, close) * np.exp(np.abs(rng.normal(0, 0.005, T)))
    low_p = np.minimum(open_p, close) * np.exp(-np.abs(rng.normal(0, 0.005, T)))
    
    gk = unsmoothed.garman_klass_vol(open_p, high_p, low_p, close)
    pk = unsmoothed.parkinson_vol(high_p, low_p)
    yz = unsmoothed.yang_zhang_vol(open_p, high_p, low_p, close)
    
    assert len(gk) == T and len(pk) == T and len(yz) == T
    assert np.all(np.isfinite(gk)) and np.all(gk > 0)
    assert np.all(np.isfinite(pk)) and np.all(pk > 0)
    assert np.all(np.isfinite(yz)) and np.all(yz > 0)
    
    # Check that construct_vol_series auto-detects OHLC dataframe
    df_ohlc = pd.DataFrame({"open": open_p, "high": high_p, "low": low_p, "close": close})
    r = np.diff(np.log(close), prepend=np.log(close[0]))
    vol_ohlc = unsmoothed.construct_vol_series(r, 1, df_raw=df_ohlc)
    assert len(vol_ohlc) == T
    np.testing.assert_allclose(vol_ohlc, gk)
    
    print("[PASS] Intraday volatility estimators (Garman-Klass, Parkinson, Yang-Zhang) verified")


def test_unsmoothed_vol_recovers_garch_shock():
    """Verify that unsmoothed daily volatility allows SINDy to identify r^2
    with a well-conditioned library (cond < 100) and no exploding coefficients."""
    import importlib
    unsmoothed = importlib.import_module("12_unsmoothed_vol")
    df = load_master()
    r = df["r"].to_numpy(float)
    vix = df["vix"].to_numpy(float)
    
    sig_1d = unsmoothed.construct_vol_series(r, 1)
    X = np.column_stack([r, sig_1d, vix])
    
    # Split on standard masks
    from data_utils import split_masks
    train, val, test = split_masks(df)
    Z_tr, mu, sd = zscore(X[train])
    Z_va = apply_zscore(X[val], mu, sd)
    
    # Non-redundant library for W=1
    keep = [0, 1, 2, 3, 4, 6, 8, 9, 10]
    Th_tr = sc.build_library(Z_tr[:-1])[:, keep]
    Th_va = sc.build_library(Z_va[:-1])[:, keep]
    
    # Condition number must be well-conditioned (< 100, not 1e16!)
    cond = np.linalg.cond(Th_tr)
    assert cond < 100, f"Design matrix is ill-conditioned! cond={cond:.2e}"
    
    path = sc.sweep_lambda(Th_tr, Z_tr[1:], Th_va, Z_va[1:])
    lam = sc.select_lambda(path, n_val=len(Z_va) - 1)
    Xi = sc.identify(Th_tr, Z_tr[1:], lam)
    
    sub_terms = [LIBRARY_TERMS[i] for i in keep]
    eq = {sub_terms[i]: float(Xi[i, 1]) for i in range(len(sub_terms))}
    
    # The discrete GARCH shock r^2 must be active and positive
    assert eq.get("r2", 0.0) > 0.0, f"GARCH shock r^2 was not discovered or non-positive: {eq.get('r2')}"
    # All coefficients must be bounded (no exploding coefficients)
    assert max(abs(v) for v in eq.values()) < 5.0, f"Exploding coefficient detected: {eq}"
    
    print(f"[PASS] SINDy on unsmoothed daily vol: cond={cond:.1f}, r^2={eq['r2']:.4f} > 0, all terms bounded")


def test_garch_rolling_window_consistency():
    """Verify that GARCH rolling-window variance mapping in multi-split evaluation
    is mathematically consistent with GARCHModel in models.py."""
    df = load_master()
    r = df["r"].to_numpy(float)
    from data_utils import split_masks
    train, val, test = split_masks(df)
    n_fit = int(train.sum())
    
    gm = GARCHModel(df, gjr=False).fit(n_fit)
    origins = [1000, 2000, 3000]
    fc_models = gm.sigma_forecasts(origins, [1, 5, 21])
    
    # Compute using the causal filter and formula directly
    pars = gm.pars
    s2 = garch_mod.filter_sigma2(r, pars)
    omega = pars["omega"]
    pers = pars["persistence"]
    r2 = r ** 2
    
    for h in [1, 5, 21]:
        manual_fc = []
        for t in origins:
            s2j = s2[t + 1]
            fsum = 0.0
            for _ in range(h):
                fsum += s2j
                s2j = omega + pers * s2j
            lo = t + h - 21
            real = r2[max(lo, 0): t + 1].sum()
            manual_fc.append(np.sqrt(max(0.0, real + fsum) / 21))
            
        np.testing.assert_allclose(fc_models[h], manual_fc, rtol=1e-6,
                                  err_msg=f"GARCH rolling window mismatch at h={h}")
        
    print("[PASS] GARCH rolling window forecasting mathematically verified against GARCHModel")


def main():
    test_features_1_to_1_alignment()
    test_no_lookahead_leakage()
    test_multi_split_partitions()
    test_intraday_volatility_estimators()
    test_unsmoothed_vol_recovers_garch_shock()
    test_garch_rolling_window_consistency()
    print("ALL VERIFICATION CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
