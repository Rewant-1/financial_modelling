"""Discrete-time SINDy core: candidate library, STLSQ and SR3 solvers,
lambda selection, map iteration, and ensemble inclusion probabilities.

Everything here operates on the STANDARDIZED state x = (r, sig, vix) so that a
single threshold lambda is meaningful across equations. The identified map is
x_{t+1} = Theta(x_t) @ Xi, one column of Xi per state variable.
"""
import numpy as np
import pandas as pd

from config import LIBRARY_TERMS


def build_library(Z: np.ndarray) -> np.ndarray:
    """Theta (T, 12): candidate terms evaluated on standardized state (T, 3)."""
    r, s, v = Z[:, 0], Z[:, 1], Z[:, 2]
    cols = {
        "1": np.ones_like(r), "r": r, "sig": s, "vix": v,
        "r2": r * r, "sig2": s * s, "vix2": v * v,
        "r_sig": r * s, "r_vix": r * v, "sig_vix": s * v,
        "r_absr": r * np.abs(r), "sig_abssig": s * np.abs(s),
    }
    return np.column_stack([cols[t] for t in LIBRARY_TERMS])


def stlsq(Theta: np.ndarray, Y: np.ndarray, lam: float, n_iter: int = 20) -> np.ndarray:
    """Sequential thresholded least squares: fit, zero |coef| < lam, refit."""
    Xi = np.linalg.lstsq(Theta, Y, rcond=None)[0]
    for _ in range(n_iter):
        active = np.abs(Xi) >= lam
        Xi_new = np.zeros_like(Xi)
        for j in range(Y.shape[1]):
            idx = active[:, j]
            if idx.any():
                Xi_new[idx, j] = np.linalg.lstsq(Theta[:, idx], Y[:, j], rcond=None)[0]
        if np.allclose(Xi_new, Xi):
            break
        Xi = Xi_new
    Xi[np.abs(Xi) < lam] = 0.0
    return Xi


def sr3(Theta: np.ndarray, Y: np.ndarray, lam: float, nu: float = 1.0,
        n_iter: int = 500, tol: float = 1e-10) -> np.ndarray:
    """Relaxed regularized regression (Champion et al. 2020). More stable than
    STLSQ when library terms are correlated (sigma vs vix are ~0.8)."""
    p = Theta.shape[1]
    AtA = Theta.T @ Theta
    AtY = Theta.T @ Y
    M = AtA + nu * np.eye(p)
    w = np.zeros((p, Y.shape[1]))
    Xi = w.copy()
    for _ in range(n_iter):
        Xi = np.linalg.solve(M, AtY + nu * w)
        w_new = np.sign(Xi) * np.maximum(np.abs(Xi) - lam / (2 * nu), 0.0)
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    # final refit on surviving support, unregularized (same convention as STLSQ)
    Xi[np.abs(Xi) < lam / (2 * nu)] = 0.0
    out = np.zeros_like(Xi)
    for j in range(Y.shape[1]):
        idx = np.abs(Xi[:, j]) > 0
        if idx.any():
            out[idx, j] = np.linalg.lstsq(Theta[:, idx], Y[:, j], rcond=None)[0]
        out[np.abs(out[:, j]) < lam, j] = 0.0
    return out


def one_step_rmse(Theta: np.ndarray, X_next: np.ndarray, Xi: np.ndarray) -> np.ndarray:
    """Per-equation one-step RMSE of map x_{t+1} ~= Theta(x_t) Xi."""
    err = X_next - Theta @ Xi
    return np.sqrt(np.mean(err ** 2, axis=0))


def prune_significance(Xi, Theta, Y, tmin=2.5, hac=10):
    """Post-selection significance pruning: refit OLS on the active support of
    each equation with HAC standard errors and drop terms with |t| < tmin.

    Rationale: one-step RMSE is nearly flat in lambda for small terms (a real
    r^2 term and pure noise cost about the same in average prediction error),
    so thresholding alone cannot separate them. A small coefficient on a REAL
    term comes with a large t-statistic; noise does not. This is the bridge
    to classical econometric practice ('which terms are significant?')."""
    import statsmodels.api as sm
    out = np.zeros_like(Xi)
    for j in range(Y.shape[1]):
        idx = np.where(np.abs(Xi[:, j]) > 0)[0]
        if len(idx) == 0:
            continue
        res = sm.OLS(Y[:, j], Theta[:, idx]).fit(
            cov_type="HAC", cov_kwds={"maxlags": hac})
        keep = np.abs(res.tvalues) >= tmin
        if keep.any():
            out[idx[keep], j] = res.params[keep]
    return out


def identify(Theta, Y, lam, solver="stlsq", tmin=2.5):
    """Full identification step used across the project:
    thresholded sparse regression + significance pruning."""
    Xi = stlsq(Theta, Y, lam) if solver == "stlsq" else sr3(Theta, Y, lam)
    return prune_significance(Xi, Theta, Y, tmin=tmin)


def sweep_lambda(Theta_tr, Xn_tr, Theta_val, Xn_val, lams=None) -> pd.DataFrame:
    """Sparsity path: lambda vs #terms vs train/val one-step RMSE (mean over
    the three equations). This drives the parsimony-elbow selection."""
    if lams is None:
        lams = np.logspace(-3, 0.3, 22)
    rows = []
    for lam in lams:
        Xi = stlsq(Theta_tr, Xn_tr, lam)
        rows.append({
            "lambda": lam,
            "n_terms": int((np.abs(Xi) > 0).sum()),
            "train_rmse": one_step_rmse(Theta_tr, Xn_tr, Xi).mean(),
            "val_rmse": one_step_rmse(Theta_val, Xn_val, Xi).mean(),
        })
    return pd.DataFrame(rows)


def select_lambda(path: pd.DataFrame, n_val: int = 500) -> float:
    """One-standard-error rule, discovery-oriented side.

    The RMSE-optimal lambda can threshold away small but real terms (e.g. the
    r^2 shock term, whose one-step contribution to a smoothed vol proxy is
    below the noise floor). Among models statistically tied with the optimum
    (val RMSE <= best + 1 SE), we keep the RICHEST (smallest lambda). SE of
    an RMSE is approx RMSE / sqrt(2n). The selected model is documented as
    'statistically indistinguishable from the RMSE-optimal one'."""
    best = float(path["val_rmse"].min())
    se = best / np.sqrt(2 * max(n_val, 10))
    ok = path[path["val_rmse"] <= best + se]
    return float(ok["lambda"].min())


def iterate_map(x0: np.ndarray, Xi: np.ndarray, h: int) -> np.ndarray:
    """Iterate the discovered map h steps from standardized state x0.
    Returns (h, 3); row k is x at k+1 steps ahead."""
    out = np.empty((h, Xi.shape[1]))
    x = x0.astype(float).copy()
    for k in range(h):
        theta = build_library(x[None, :])[0]
        x = theta @ Xi
        out[k] = x
    return out


def equation_table(Xi: np.ndarray) -> pd.DataFrame:
    """Discovered equations as a readable (term x equation) coefficient table."""
    eq_names = ["r_next", "sig_next", "vix_next"]
    df = pd.DataFrame(Xi, index=LIBRARY_TERMS, columns=eq_names)
    return df.round(4)


def format_equations(Xi: np.ndarray) -> list:
    """Human-readable one-line-per-equation strings."""
    eqs = []
    names = LIBRARY_TERMS
    for j, target in enumerate(["r'", "sig'", "vix'"]):
        terms = [f"{Xi[i, j]:+.4f}*{names[i]}" for i in range(len(names))
                 if abs(Xi[i, j]) > 0]
        eqs.append(f"{target} = " + (" ".join(terms) if terms else "0"))
    return eqs


def ensemble_inclusion(Z, lam, n_boot=500, block_frac=0.5, seed=42):
    """Bagged SINDy over contiguous time blocks (Fasel et al. 2022), using the
    full identification step (STLSQ + significance pruning). Inclusion
    probability = P(term survives selection on a random half-history block).

    Returns (prob, med_coef): per-term inclusion probability per equation and
    the median nonzero coefficient — the uncertainty quantification for
    discovered terms."""
    rng = np.random.default_rng(seed)
    Theta = build_library(Z[:-1])   # row t holds candidate terms of state t
    Y = Z[1:]                       # row t holds the next state x_{t+1}
    T = len(Y)
    blk = int(T * block_frac)
    counts = np.zeros((Theta.shape[1], 3))
    coefs = []
    for _ in range(n_boot):
        start = rng.integers(0, T - blk)
        idx = np.arange(start, start + blk)
        Xi = identify(Theta[idx], Y[idx], lam)
        counts += (np.abs(Xi) > 0)
        coefs.append(Xi)
    prob = counts / n_boot
    stacked = np.stack(coefs)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(np.where(np.abs(stacked) > 0, stacked, np.nan),
                           axis=0)
    return prob, np.nan_to_num(med)
