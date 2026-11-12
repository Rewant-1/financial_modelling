"""The model zoo for the forecasting horse race.

Protocol: construct with the FULL master frame; call fit(n_fit) to estimate on
data[:n_fit] only; call sigma_forecasts(origins, horizons) where every origin
index is >= n_fit. Forecast at origin t may use information through t only —
filters are causal, lags come from actual history, maps iterate from x_t.

All sigma forecasts are in DAILY units (not annualized) to match the target.
"""
import numpy as np
import pandas as pd

import garch as garch_mod
import sindy_core as sc
from data_utils import state_matrix, zscore, apply_zscore, undo_zscore

try:
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from statsmodels.tsa.api import VAR as smVAR
    HAS_SM = True
except Exception:
    HAS_SM = False


class BaseModel:
    name = "base"
    produces_r = False  # whether the model also forecasts the return equation

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.X = state_matrix(df)          # (T, 3): r, sigma, vix
        self.r = self.X[:, 0]
        self.sig = self.X[:, 1]
        self.n_fit = None

    def fit(self, n_fit: int):
        self.n_fit = n_fit
        return self

    def sigma_forecasts(self, origins, horizons):
        raise NotImplementedError

    def r_forecasts(self, origins):
        raise NotImplementedError


class SigmaRW(BaseModel):
    """Naive floor: tomorrow's vol is today's vol, at every horizon."""
    name = "naive_rw"

    def sigma_forecasts(self, origins, horizons):
        return {h: self.sig[origins].copy() for h in horizons}


class EWMA(BaseModel):
    """RiskMetrics EWMA (lambda = 0.94), mapped onto the realized-vol target:
    the 21-day rolling variance at t+h = 21-h realized days + h days of
    E[r^2] = one-step EWMA variance (flat forecast, IGARCH property)."""
    name = "ewma"

    def __init__(self, df):
        super().__init__(df)
        self.lam = 0.94
        self.w_len = 21

    def _filter(self):
        eps = self.r - self.r[: self.n_fit].mean()
        w = self.lam
        s2 = np.empty(len(eps))
        s2[0] = eps[: min(len(eps), 50)].var()
        for t in range(1, len(eps)):
            s2[t] = w * s2[t - 1] + (1 - w) * eps[t - 1] ** 2
        return s2  # s2[t] forecasts variance at t from info at t-1

    def sigma_forecasts(self, origins, horizons):
        s2 = self._filter()
        r2 = self.r ** 2  # means are ~1e-4 of r^2; second moment approx OK
        out = {}
        for h in horizons:
            fs = []
            for t in origins:
                s2n = s2[t + 1]                       # one-step E[r_{t+1}^2]
                lo = t + h - self.w_len               # window start day
                real = r2[max(lo, 0): t + 1].sum()    # realized part of window
                fs.append(np.sqrt((real + h * s2n) / self.w_len))
            out[h] = np.clip(np.array(fs), 1e-8, None)
        return out


class ARSigma(BaseModel):
    """AR(p) on sigma with p chosen by BIC (p <= 10); iterate h steps."""
    name = "ar_sigma"

    def fit(self, n_fit):
        super().fit(n_fit)
        y = self.sig[:n_fit]
        best = (np.inf, None, None)
        for p in range(1, 11):
            Y, F = y[p:], np.column_stack(
                [y[p - 1 - i: n_fit - 1 - i] for i in range(p)])
            F = np.column_stack([np.ones(len(F)), F])
            beta, *_ = np.linalg.lstsq(F, Y, rcond=None)
            resid = Y - F @ beta
            n, k = len(Y), F.shape[1]
            bic = n * np.log(resid @ resid / n) + k * np.log(n)
            if bic < best[0]:
                best = (bic, p, beta)
        self.p, self.beta = best[1], best[2]
        return self

    def _path(self, origin, h):
        hist = list(self.sig[origin - self.p + 1: origin + 1])
        out = []
        for _ in range(h):
            lags = np.array(hist[-self.p:][::-1])
            nxt = self.beta[0] + self.beta[1:] @ lags
            out.append(nxt)
            hist.append(nxt)
        return np.array(out)

    def sigma_forecasts(self, origins, horizons):
        hmax = max(horizons)
        paths = np.array([self._path(t, hmax) for t in origins])
        return {h: np.clip(paths[:, h - 1], 1e-8, None) for h in horizons}


class GARCHModel(BaseModel):
    """GARCH(1,1) (gjr=False) or GJR-GARCH(1,1) (gjr=True) on returns,
    mapped onto the realized-vol target: the 21-day rolling variance at t+h
    combines realized squared returns with the iterated conditional-variance
    path sigma2_{t+k|t} — the apples-to-apples benchmark for a rolling-vol
    target. Fitted on the estimation window; between refits the filter rolls
    daily with actual returns."""
    W_LEN = 21

    def __init__(self, df, gjr=False):
        super().__init__(df)
        self.gjr = gjr
        self.name = "gjr_garch" if gjr else "garch11"

    def fit(self, n_fit):
        super().fit(n_fit)
        self.pars = garch_mod.fit_garch(self.r[:n_fit], gjr=self.gjr)
        self.s2 = garch_mod.filter_sigma2(self.r, self.pars)  # causal
        return self

    def sigma_forecasts(self, origins, horizons):
        omega = self.pars["omega"]
        pers = self.pars["alpha"] + self.pars["beta"] + 0.5 * self.pars["gamma"]
        r2 = self.r ** 2
        out = {}
        for h in horizons:
            fs = []
            for t in origins:
                s2j = self.s2[t + 1]              # sigma2_{t+1|t}
                fsum = 0.0
                for _ in range(h):
                    fsum += s2j
                    s2j = omega + pers * s2j
                lo = t + h - self.W_LEN
                real = r2[max(lo, 0): t + 1].sum()
                fs.append(np.sqrt(max(0.0, real + fsum) / self.W_LEN))
            out[h] = np.clip(np.array(fs), 1e-8, None)
        return out


class VARModel(BaseModel):
    """VAR(p) on (r, sigma, vix), p by BIC (statsmodels). Iterates all three
    equations jointly, so sigma forecasts embed the return/vix feedback."""
    name = "var"
    produces_r = True

    def fit(self, n_fit):
        super().fit(n_fit)
        if HAS_SM:
            self.res = smVAR(self.X[:n_fit]).fit(maxlags=10, ic="bic")
            self.p = self.res.k_ar
        else:
            X_data = self.X[:n_fit]
            best = (np.inf, 1, None)
            for p in range(1, 11):
                if len(X_data) <= p + 10:
                    continue
                Y = X_data[p:]
                F = np.column_stack([np.ones(len(Y))] +
                                    [X_data[p - 1 - i: len(X_data) - 1 - i] for i in range(p)])
                B, *_ = np.linalg.lstsq(F, Y, rcond=None)
                res = Y - F @ B
                sigma_u = (res.T @ res) / len(Y)
                sign, logdet = np.linalg.slogdet(sigma_u)
                bic = logdet + (F.shape[1] * 3) * np.log(len(Y)) / len(Y)
                if bic < best[0]:
                    best = (bic, p, B)
            self.p = best[1]
            self.B = best[2]
            self.res = None
        return self

    def _paths(self, origins, hmax):
        paths = np.empty((len(origins), hmax, 3))
        if HAS_SM and self.res is not None:
            for i, t in enumerate(origins):
                f = self.res.forecast(self.X[t - self.p + 1: t + 1], hmax)
                paths[i] = f
        else:
            for i, t in enumerate(origins):
                hist = list(self.X[t - self.p + 1: t + 1])
                f = []
                for _ in range(hmax):
                    lags = np.array(hist[-self.p:][::-1]).reshape(-1)
                    F_vec = np.concatenate([[1.0], lags])
                    nxt = F_vec @ self.B
                    f.append(nxt)
                    hist.append(nxt)
                paths[i] = np.array(f)
        return paths

    def sigma_forecasts(self, origins, horizons):
        paths = self._paths(origins, max(horizons))
        return {h: np.clip(paths[:, h - 1, 1], 1e-8, None) for h in horizons}

    def r_forecasts(self, origins):
        return self._paths(origins, 1)[:, 0, 0]


class DMDModel(BaseModel):
    """Dynamic mode decomposition: best linear map on the standardized,
    demeaned state. The 'what if everything is linear' control."""
    name = "dmd"
    produces_r = True

    def fit(self, n_fit):
        super().fit(n_fit)
        Z, self.mu, self.sd = zscore(self.X[:n_fit])
        self.m = Z.mean(axis=0)
        Zc = Z - self.m
        A, *_ = np.linalg.lstsq(Zc[:-1], Zc[1:], rcond=None)  # (3,3)
        self.A = A
        return self

    def _paths(self, origins, hmax):
        paths = np.empty((len(origins), hmax, 3))
        for i, t in enumerate(origins):
            z = apply_zscore(self.X[t], self.mu, self.sd) - self.m
            traj = []
            for _ in range(hmax):
                z = z @ self.A
                traj.append(z)
            paths[i] = undo_zscore(np.array(traj) + self.m, self.mu, self.sd)
        return paths

    def sigma_forecasts(self, origins, horizons):
        paths = self._paths(origins, max(horizons))
        return {h: np.clip(paths[:, h - 1, 1], 1e-8, None) for h in horizons}

    def r_forecasts(self, origins):
        return self._paths(origins, 1)[:, 0, 0]


class SINDyModel(BaseModel):
    """Discrete-time SINDy map (STLSQ or SR3) on the standardized state.
    Lambda selected per refit by one-step validation error on the final
    VAL_TAIL observations of the estimation window (never on test data)."""
    VAL_TAIL = 500
    LAMBDAS = np.logspace(-3, 0.3, 22)

    def __init__(self, df, solver="stlsq"):
        super().__init__(df)
        self.solver = solver
        self.name = f"sindy_{solver}"

    produces_r = True

    def fit(self, n_fit):
        super().fit(n_fit)
        Z, self.mu, self.sd = zscore(self.X[:n_fit])
        n_val = min(self.VAL_TAIL, len(Z) // 4)
        tr, va = Z[:-n_val], Z[-n_val:]
        Theta_tr, Theta_va = sc.build_library(tr[:-1]), sc.build_library(va[:-1])
        raw = (lambda Th, Y, lam: sc.sr3(Th, Y, lam)) if self.solver == "sr3" \
            else sc.stlsq
        lams, rmses = [], []
        for lam in self.LAMBDAS:  # sweep raw (fast); pruning is post-processing
            Xi = raw(Theta_tr, tr[1:], lam)
            rmse = np.sqrt(np.mean((va[1:] - Theta_va @ Xi) ** 2))
            lams.append(lam)
            rmses.append(rmse)
        # 1-SE rule (discovery side): richest model statistically tied with
        # the RMSE optimum — same convention as sindy_core.select_lambda.
        rmses, lams = np.array(rmses), np.array(lams)
        best = rmses.min()
        se = best / np.sqrt(2 * max(len(va) - 1, 10))
        self.lam = float(lams[rmses <= best + se].min())
        self.Xi = sc.identify(Theta_tr, tr[1:], self.lam, solver=self.solver)
        Theta_full = sc.build_library(Z[:-1])
        self.Xi_full = sc.identify(Theta_full, Z[1:], self.lam,
                                   solver=self.solver)
        return self

    def _solve(self, Theta, Y, lam):
        return sc.identify(Theta, Y, lam, solver=self.solver)

    def _paths(self, origins, hmax):
        paths = np.empty((len(origins), hmax, 3))
        for i, t in enumerate(origins):
            z0 = apply_zscore(self.X[t], self.mu, self.sd)
            paths[i] = undo_zscore(sc.iterate_map(z0, self.Xi_full, hmax),
                                   self.mu, self.sd)
        return paths

    def sigma_forecasts(self, origins, horizons):
        paths = self._paths(origins, max(horizons))
        return {h: np.clip(paths[:, h - 1, 1], 1e-8, None) for h in horizons}

    def r_forecasts(self, origins):
        return self._paths(origins, 1)[:, 0, 0]


class DirectMLBase(BaseModel):
    """Direct h-step regressors: predict sigma_{t+h} from features at t.
    Ridge/Lasso use the SAME library matrix as SINDy (apples-to-apples);
    tree models use raw features. Regularization strength chosen on the
    final 500 observations of the estimation window."""
    VAL_TAIL = 500
    feature_kind = "library"

    def _features(self, Z, lag=0):
        if self.feature_kind == "library":
            return sc.build_library(Z)
        if len(Z) == 0:
            return np.empty((0, 8), dtype=float)
        r, s, v = Z[:, 0], Z[:, 1], Z[:, 2]
        rL = np.concatenate([[r[0]], r[:-1]])
        sL = np.concatenate([[s[0]], s[:-1]])
        vL = np.concatenate([[v[0]], v[:-1]])
        return np.column_stack([r, s, v, np.abs(r), r * r, rL, sL, vL])

    def _make_model(self, alpha=None):
        raise NotImplementedError

    def _alpha_grid(self):
        raise NotImplementedError

    def fit(self, n_fit):
        super().fit(n_fit)
        Xf = self.X[:n_fit]
        Z, self.mu, self.sd = zscore(Xf)
        F_all = self._features(Z)
        # standardized target so coefficients are comparable to SINDy's
        self.y_mu = float(self.sig[:n_fit].mean())
        self.y_sd = float(self.sig[:n_fit].std(ddof=1))
        n_val = min(self.VAL_TAIL, len(Z) // 4)
        self.models = {}
        for h in self.HORIZONS_SET:
            y = (self.sig[h:n_fit] - self.y_mu) / self.y_sd
            F, yh = F_all[: n_fit - h], y
            F_tr, y_tr = F[:-n_val], yh[:-n_val]
            F_va, y_va = F[-n_val:], yh[-n_val:]
            best = (np.inf, None)
            for a in self._alpha_grid():
                m = self._make_model(a)
                m.fit(F_tr, y_tr)
                rmse = np.sqrt(np.mean((m.predict(F_va) - y_va) ** 2))
                if rmse < best[0]:
                    best = (rmse, a)
            m = self._make_model(best[1])
            m.fit(F, yh)
            self.models[h] = m
        return self

    def sigma_forecasts(self, origins, horizons):
        out = {}
        Z_all = apply_zscore(self.X, self.mu, self.sd)
        F_all = self._features(Z_all)
        for h in horizons:
            pred = self.models[h].predict(F_all[origins]) * self.y_sd + self.y_mu
            out[h] = np.clip(pred, 1e-8, None)
        return out

    HORIZONS_SET = (1, 5, 21)


class RidgeLib(DirectMLBase):
    """Ridge on the SINDy library — deliberately dense: the anti-sparsity
    contrast (same features, no thresholding)."""
    name = "ridge_lib"
    feature_kind = "library"

    def _make_model(self, alpha=None):
        return Ridge(alpha=alpha)

    def _alpha_grid(self):
        return np.logspace(-2, 4, 13)


class LassoLib(DirectMLBase):
    """Lasso on the SINDy library: sparsity WITHOUT the dynamics reading."""
    name = "lasso_lib"
    feature_kind = "library"

    def _make_model(self, alpha=None):
        return Lasso(alpha=alpha, max_iter=20000)

    def _alpha_grid(self):
        return np.logspace(-6, -2, 13)


class RandomForestModel(DirectMLBase):
    """Random forest on raw features (r, sigma, vix, |r|, r^2, one lag each).
    Single fixed hyperparameter set (documented, no tuning on test)."""
    name = "random_forest"
    feature_kind = "raw"
    ALPHA = 0  # no grid; keep one configuration

    def _make_model(self, alpha=None):
        return RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            random_state=42, n_jobs=-1)

    def _alpha_grid(self):
        return [None]


class XGBModel(DirectMLBase):
    name = "xgboost"
    feature_kind = "raw"

    def _make_model(self, alpha=None):
        return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42, n_jobs=-1, verbosity=0)

    def _alpha_grid(self):
        return [None]


def build_zoo(df, include_optional=True):
    """Instantiate the full Python model zoo for the horse race."""
    zoo = [SigmaRW(df), EWMA(df), ARSigma(df),
           GARCHModel(df, gjr=False), GARCHModel(df, gjr=True)]
    if HAS_SM:
        zoo.append(VARModel(df))
    zoo.append(DMDModel(df))
    if HAS_SKLEARN:
        zoo += [RidgeLib(df), LassoLib(df), RandomForestModel(df)]
    if HAS_SKLEARN and HAS_XGB and include_optional:
        zoo.append(XGBModel(df))
    zoo += [SINDyModel(df, "stlsq"), SINDyModel(df, "sr3")]
    return zoo
