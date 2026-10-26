"""Phase 0 reproduction check — run before trusting anything on real data.

Generates returns from a KNOWN GARCH(1,1) process, constructs the same
state variables as the real pipeline (r, 21-day rolling sigma, and a
VIX-like proxy correlated with sigma), and verifies that discrete-time
SINDy rediscovers the volatility equation structure on its own:
  - sigma_next ~ const + alpha*r^2 + beta*sigma   (the GARCH form)
  - the return equation comes out essentially EMPTY (EMH by construction)

Also checks the numpy QMLE GARCH fallback recovers (alpha, beta) accurately.

Run:  python tests/test_sindy_recovers_garch.py   (exit 0 = pass)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import sindy_core as sc
import garch as garch_mod

OMEGA, ALPHA, BETA = 1e-6, 0.08, 0.90


def simulate_garch(T=40000, seed=7):
    rng = np.random.default_rng(seed)
    r = np.empty(T)
    s2 = OMEGA / (1 - ALPHA - BETA)
    for t in range(T):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = OMEGA + ALPHA * r[t] ** 2 + BETA * s2
    return r


def main():
    import pandas as pd
    r = simulate_garch()
    sigma = pd.Series(r).rolling(21).std().to_numpy()
    vix_like = 1.15 * sigma + 0.25 * np.nanstd(sigma) * \
        np.random.default_rng(1).standard_normal(len(r))
    X = np.column_stack([r, sigma, vix_like])
    X = X[1000:]  # burn-in (also drops the rolling-window NaN head)
    assert np.isfinite(X).all()
    Z = (X - X.mean(0)) / X.std(0, ddof=1)

    Theta, Y = sc.build_library(Z[:-1]), Z[1:]
    n_val = 2000
    path = sc.sweep_lambda(Theta[:-n_val], Y[:-n_val],
                           Theta[-n_val:], Y[-n_val:])
    lam = sc.select_lambda(path, n_val=n_val)
    Xi = sc.identify(Theta[:-n_val], Y[:-n_val], lam)
    terms = sc.LIBRARY_TERMS if hasattr(sc, "LIBRARY_TERMS") else None
    from config import LIBRARY_TERMS as TERMS
    sig_eq = dict(zip(TERMS, Xi[:, 1].round(4)))
    r_eq = dict(zip(TERMS, Xi[:, 0].round(4)))

    print(f"lambda* = {lam:.4f}")
    print("discovered sigma equation:", {k: v for k, v in sig_eq.items() if v})
    print("discovered r equation:    ", {k: v for k, v in r_eq.items() if v})

    # --- assertions: the GARCH form re-emerges ---
    assert sig_eq.get("sig", 0) > 0.30, "sigma persistence term missing"
    assert sig_eq.get("sig", 0) < 0.999, "sigma term degenerate"
    assert sig_eq.get("r2", 0) > 0, "shock term r^2 must be active and positive"
    assert max(abs(v) for k, v in r_eq.items() if k != "1") < 0.15, \
        "return equation should be ~empty (returns unpredictable by construction)"
    print("[PASS] STLSQ rediscovered the GARCH(1,1) sigma-equation structure")

    Xi_sr3 = sc.identify(Theta[:-n_val], Y[:-n_val], lam, solver="sr3")
    sig_sr3 = dict(zip(TERMS, Xi_sr3[:, 1].round(4)))
    assert sig_sr3.get("sig", 0) > 0.30 and sig_sr3.get("r2", 0) > 0, \
        "SR3 must agree on the core terms"
    print("[PASS] SR3 agrees on the core structure")

    pars = garch_mod.fit_garch_qmle(r)
    print(f"QMLE GARCH recovery: alpha={pars['alpha']:.3f} (true {ALPHA}), "
          f"beta={pars['beta']:.3f} (true {BETA})")
    assert abs(pars["alpha"] - ALPHA) < 0.03 and abs(pars["beta"] - BETA) < 0.05, \
        "QMLE fallback must recover true parameters"
    print("[PASS] QMLE GARCH fallback recovers parameters")
    print("ALL CHECKS PASSED — pipeline core is sound on known ground truth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
