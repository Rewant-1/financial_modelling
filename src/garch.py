"""GARCH(1,1) and GJR-GARCH(1,1): fitting (arch package with a numpy QMLE
fallback) plus causal filtering and h-step variance forecasts.

Parameterization (daily units, variance in return^2):
    eps_t = r_t - mu
    GARCH:      sigma2_t = omega + alpha*eps_{t-1}^2 + beta*sigma2_{t-1}
    GJR:        sigma2_t = omega + (alpha + gamma*I(eps_{t-1}<0))*eps_{t-1}^2
                            + beta*sigma2_{t-1}

Returns are fitted in percent (x100) for numerical conditioning and converted
back here, so everything leaving this module is in daily units.
"""
import warnings
import numpy as np

try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:  # pragma: no cover - fallback when arch unavailable
    HAS_ARCH = False


def fit_garch_arch(r: np.ndarray, gjr: bool = False):
    """Fit via arch (QML, normal). Returns dict of daily-unit params + summary."""
    x = r * 100.0
    model = arch_model(x, mean="Constant",
                       vol="GARCH", p=1, o=1 if gjr else 0, q=1,
                       dist="normal", rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp="off", show_warning=False)
    p = res.params
    out = {
        "mu": p["mu"] / 100.0,
        "omega": p["omega"] / 1e4,
        "alpha": p["alpha[1]"],
        "beta": p["beta[1]"],
        "gamma": p.get("gamma[1]", 0.0),
        "loglik": float(res.loglikelihood),
        "n_obs": int(res.nobs),
        "converged": bool(res.convergence_flag == 0),
    }
    out["persistence"] = out["alpha"] + out["beta"] + 0.5 * out["gamma"]
    out["se"] = {k: (float(res.std_err[k]) if k in res.std_err else None)
                 for k in ("mu", "omega", "alpha[1]", "beta[1]", "gamma[1]")}
    return out, res


def fit_garch_qmle(r: np.ndarray, gjr: bool = False):
    """Self-contained fallback: Gaussian QML with variance targeting,
    scipy L-BFGS-B. Used by tests and when arch is not installed."""
    from scipy.optimize import minimize
    x = r - r.mean()
    target = float(x.var(ddof=1))

    def filter_sigma2(pars):
        a, b = pars[0], pars[1]
        g = pars[2] if gjr else 0.0
        pers = a + b + 0.5 * g
        if pers >= 0.999 or min(a, b) < 1e-8:
            return None
        omega = target * (1.0 - pers)
        s2 = np.empty(len(x))
        s2[0] = target
        for t in range(1, len(x)):
            s2[t] = omega + b * s2[t - 1] + (a + g * (x[t - 1] < 0)) * x[t - 1] ** 2
        return s2

    def nll(pars):
        s2 = filter_sigma2(pars)
        if s2 is None:
            return 1e10
        return 0.5 * float(np.mean(np.log(s2) + x ** 2 / s2))

    x0 = np.array([0.08, 0.90, 0.10]) if gjr else np.array([0.08, 0.90])
    bounds = [(1e-6, 0.5), (1e-6, 0.999)] + ([(1e-6, 0.5)] if gjr else [])
    res = minimize(nll, x0[:3 if gjr else 2], method="L-BFGS-B", bounds=bounds)
    a, b = float(res.x[0]), float(res.x[1])
    g = float(res.x[2]) if gjr else 0.0
    pers = a + b + 0.5 * g
    return {
        "mu": float(r.mean()), "omega": target * (1.0 - pers),
        "alpha": a, "beta": b, "gamma": g, "persistence": pers,
        "loglik": -float(res.fun) * len(r) / 2, "n_obs": len(r),
        "converged": bool(res.success),
    }


def fit_garch(r: np.ndarray, gjr: bool = False):
    if HAS_ARCH:
        try:
            out, res = fit_garch_arch(r, gjr=gjr)
            return out
        except Exception:
            pass
    return fit_garch_qmle(r, gjr=gjr)


def filter_sigma2(r: np.ndarray, pars: dict) -> np.ndarray:
    """Causal conditional-variance path sigma2_t given parameters. Element t
    uses only information through t-1's return, so reading sigma2[t+1] at
    origin t is a legitimate one-step forecast."""
    eps = r - pars["mu"]
    omega, a, b, g = pars["omega"], pars["alpha"], pars["beta"], pars["gamma"]
    s2 = np.empty(len(r))
    s2[0] = omega / max(1e-12, 1.0 - (a + b + 0.5 * g))
    for t in range(1, len(r)):
        s2[t] = omega + b * s2[t - 1] + (a + g * (eps[t - 1] < 0)) * eps[t - 1] ** 2
    return s2


def hstep_variance_forecast(s2_next: float, pars: dict, h: int) -> float:
    """sigma2 at horizon h (>=1) given the one-step forecast sigma2_{t+1|t},
    using the standard symmetric-persistence recursion."""
    pers = pars["alpha"] + pars["beta"] + 0.5 * pars["gamma"]
    omega = pars["omega"]
    if h == 1:
        return s2_next
    return omega + pers ** (h - 1) * (s2_next - omega)
