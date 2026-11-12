# SINDy for Financial Modeling — `sindy-vol`

**Discovering the governing equations of market volatility:** sparse
identification of nonlinear dynamics (SINDy) on the US equity market,
benchmarked against econometric (GARCH family), machine-learning, and
linear-dynamics (DMD) approaches under a strict walk-forward protocol.

- **State:** `x_t = [r_t, σ_t, v_t]` — S&P 500 daily log return, 21-day
  realized volatility, VIX (implied vol, daily units), 1990–present.
- **Data:** all free — Yahoo Finance (^GSPC), CBOE (VIX history; FRED
  fallback), optional FRED macro controls (DFF, T10Y3M).
- **Splits (frozen, in `data/splits.json`):** train 1990–2015 · validation
  2016–2019 · test 2020–present (COVID crash + 2022 rate cycle included on
  purpose — a stress test, not an accident).

## Quick start

```bash
# one-time setup (Python 3.12 venv; see Environment notes)
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS/Linux

# full pipeline (phases 0–8 in dependency order)
.venv/Scripts/python src/run_all.py

# fast smoke run (reduced bootstrap counts)
FAST=1 .venv/Scripts/python src/run_all.py
```

Every step can also be run individually: `src/01_download.py`,
`src/02_construct.py`, …, `src/10_horse_race.py` (numbered in dependency
order). The synthetic validation gate runs first:
`tests/test_sindy_recovers_garch.py` — SINDy must rediscover the GARCH form
on data where the truth is known before anything touches real data.

## What each step does

| Script | Phase | Output |
|---|---|---|
| `01_download.py` | 1 | raw CSVs + `data/manifest.json` (provenance) |
| `02_construct.py` | 1 | `data/processed/master.parquet`, frozen splits, ADF table |
| `03_eda.py` | 1 | 4 stylized-fact figures + `stylized_facts` table |
| `04_diagnostics.py` | 1.5 | BDS tests, linear-vs-nonlinear skill gap |
| `05_econometrics.py` | 2 | GARCH(1,1)/GJR params, half-lives, AR/VAR selections |
| `06_sindy.py` | 3 | **the discovery**: sparsity path, equations, SINDy-vs-GARCH table |
| `07_ensemble.py` | 4 | inclusion-probability heatmap, regime stability, noise robustness |
| `08_ml.py` | 5 | Ridge/Lasso/RF/XGBoost feature importances vs SINDy terms |
| `09_dynamics.py` | 7 | fixed point, eigenvalues, phase portrait, simulated moments, COVID replay |
| `10_horse_race.py` | 6 | walk-forward forecast race: RMSE/MAE/QLIKE, MZ, DM tests |
| `11_varying_splits.py` | Robustness | multi-split validation: 6 regime splits, equation stability & forecast race |
| `12_unsmoothed_vol.py` | Discovery | unsmoothed vol ($W \in [1, 2, 3, 5, 10, 21]$): GARCH $r^2$ shock and leverage recovery |

## Methodology notes (the decisions a referee will ask about)

- **Discrete-time SINDy.** `x_{t+1} = Θ(x_t)·Ξ` on the standardized state —
  no derivatives of noisy data, and it makes the GARCH analogy exact:
  putting today's return shock *in the state* means the library term `r²`
  plays the role of GARCH's α·ε² and `σ` plays β.
- **Library (12 terms):** `1, r, σ, v, r², σ², v², rσ, rv, σv, r|r|, σ|σ|`.
- **Identification = STLSQ + significance pruning.** One-step RMSE is nearly
  flat in the threshold λ for small terms (a real r² term and pure noise cost
  about the same in average prediction error), so thresholding alone cannot
  separate them. After thresholded selection we refit OLS on the active
  support with HAC standard errors and drop terms with |t| < 2.5 — a small
  coefficient on a real term carries a large t-statistic; noise does not.
- **λ selection = one-standard-error rule (discovery side).** Among models
  statistically tied with the RMSE optimum (val RMSE ≤ best + 1 SE), keep the
  richest. The selected model is reported as "statistically
  indistinguishable from the RMSE-optimal one".
- **Stochastic reality.** SINDy identifies the one-step conditional mean
  (drift); diffusion is handled by resampling residuals (block bootstrap) in
  `09_dynamics.py`. Stated openly in the report.
- **Forecast protocol.** Walk-forward, expanding window, quarterly refits,
  daily origins, horizons 1/5/21 days; QLIKE as primary metric (robust to
  vol spikes); Diebold–Mariano with Newey–West HAC (lag h+20 for the
  overlapping 21-day realized-vol target); Mincer–Zarnowitz regressions;
  return-direction accuracy as the honesty report vs a coin flip.
- **Fair benchmarking of conditional-vol models.** The target is the 21-day
  rolling σ, so GARCH/GJR/EWMA forecasts are mapped onto that object: the
  rolling variance at t+h = (21−h) realized squared returns + h steps of the
  iterated conditional-variance path. Comparing raw next-day conditional
  vol to a smoothed rolling target would handicap the econometric family —
  the mapping makes the race apples-to-apples.
- **No leakage.** Standardization statistics, λ, and ML regularization are
  computed inside each fit window only; the test window (2020+) is touched
  exactly once, by `10_horse_race.py`. The one exception is the regime
  table in `07_ensemble.py` (COVID+ row), which is structural analysis,
  never model selection — flagged in the table itself.

## The model zoo (13 Python models + STATA)

| Family | Models |
|---|---|
| Naive floor | σ random walk, EWMA-RiskMetrics (λ=0.94) |
| Econometric | AR(p) on σ (BIC), GARCH(1,1), GJR-GARCH(1,1) (QML via `arch`), VAR(p) (BIC) |
| ML (direct) | Ridge & Lasso **on the SINDy library** (apples-to-apples), Random Forest, XGBoost on raw features |
| Dynamics | DMD (best linear map), **SINDy-STLSQ**, **SINDy-SR3** |

If the STATA do-files have been run, their walk-forward GARCH forecasts
(`results/stata/stata_forecasts.csv`) are merged into the race automatically.

## STATA side (Phase 2 in the licensed software)

```stata
do stata/00_master.do
```

Runs ADF tests, AR/BIC, GARCH(1,1), EGARCH, GJR, VAR, and the quarterly
walk-forward export. Written to mirror the Python protocol exactly; verify
the `predict, variance(dynamic())` behavior on your STATA version (noted
inline). Requires `estout` (`ssc install estout`) for `esttab` exports.

## Environment notes

Developed and executed on Windows with a **Python 3.12.10 per-user install**
and a project venv (`.venv/`). Package versions in `requirements.txt` are
deliberately pinned to well-aged releases: on machines with Windows Smart
App Control / WDAC (cloud-reputation Application Control), freshly released
wheel builds can be blocked at import ("An Application Control policy has
blocked this file") while established builds load fine. If you hit that,
stay on the pins. Everything is pure-Python-on-numpy except `arch`,
`scikit-learn`, `xgboost`, `pyarrow` (compiled but mature).

Core math (STLSQ, SR3, GARCH QMLE filter/forecasts, DMD, ensemble SINDy) is
implemented in-house in `src/sindy_core.py` / `src/garch.py` — readable,
dependency-light, and unit-validated on synthetic ground truth. `pysindy`
can be installed to cross-check (optional).

## Results map (what to look at first)

- `results/tables/horse_race_main.md` — the one-table answer to RQ3 (forecasting horse race)
- `results/tables/varying_splits_term_summary.md` — multi-split stability across 6 regimes
- `results/tables/varying_splits_sindy_vs_garch.md` — out-of-sample head-to-head across splits
- `results/tables/unsmoothed_vs_smoothed_comparison.md` — recovery of GARCH r^2 shock across smoothing windows
- `results/tables/sindy_equations_stlsq.md` — the discovered system
- `results/tables/sindy_vs_garch.md` — term-by-term GARCH comparison
- `results/tables/inclusion_probabilities.csv` — which terms are real (UQ)
- `results/tables/regime_stability.md` — which terms survive 35 years
- `results/figures/fig17_split_equation_stability.png`, `fig18_split_forecast_comparison.png` — multi-split figures
- `results/figures/fig19_unsmoothed_term_evolution.png`, `fig20_unsmoothed_forecast_bars.png` — unsmoothed volatility figures
- `results/figures/fig07_inclusion_heatmap.png`, `fig11_phase_portrait.png`, `fig13_covid_replay.png` — the presentation centerpieces
- `results/tables/return_direction_accuracy.md` — the EMH honesty report

## Known limitations (write these in the report, don't hide them)

1. **Memoryless state.** The map has no shock memory beyond what the state
   carries; simulated |r|-autocorrelation decays faster than the real thing
   at lag 21 (visible in `simulated_vs_actual_moments`). Delay embedding
   (HAVOK) is the natural extension.
2. **Drift only.** Diffusion is resampled, not identified (Kramers–Moyal
   extension would close this).
3. **r² channel under window smoothing.** Against the smoothed 21-day σ proxy,
   the single-day shock term is diluted and pruned (ensemble inclusion ≈ 0);
   the 21d σ-equation is a persistence + VIX-coupling equation. However, as
   demonstrated in `12_unsmoothed_vol.py`, as the smoothing window W decreases
   towards daily (W ≤ 5), dilution vanishes and SINDy recovers the discrete
   GARCH shock: r² emerges at +0.0208 (W=5), +0.0567 (W=3), +0.1471 (W=2),
   and +0.0675 (W=1).
4. **Single market, daily frequency.** Cross-country systems and weekly
   aggregation are robustness extensions, not included.
5. **STATA walk-forward not executed here** (no STATA in this environment);
   the do-files mirror the Python protocol and export to the merge point.
