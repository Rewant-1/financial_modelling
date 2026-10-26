"""Project-wide paths, dates, and constants. Single source of truth."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
FORECASTS = RESULTS / "forecasts"
STATA = ROOT / "stata"
STATA_RESULTS = RESULTS / "stata"

START_DATE = "1990-01-01"   # VIX begins 1990-01; alignment starts here
TRAIN_END = "2015-12-31"    # frozen split, documented in data/splits.json
VAL_END = "2019-12-31"
TEST_START = "2020-01-01"   # deliberately includes COVID crash + 2022 rate cycle

HORIZONS = [1, 5, 21]       # forecast horizons in trading days
VOL_WINDOW = 21             # rolling window for realized vol (one trading month)
ANNUALIZATION = 252
EWMA_LAMBDA = 0.94          # RiskMetrics convention

SEED = 42

STATE_COLS = ["r", "sigma", "vix"]        # the three state variables
TARGET = "sigma"                          # primary forecast target

# Candidate library on the standardized state (Appendix B of the plan).
LIBRARY_TERMS = ["1", "r", "sig", "vix", "r2", "sig2", "vix2",
                 "r_sig", "r_vix", "sig_vix", "r_absr", "sig_abssig"]

REGIMES = {                  # subsample identification windows (Phase 4)
    "1990s": ("1990-01-01", "1999-12-31"),
    "dot-com": ("2000-01-01", "2007-12-31"),
    "GFC": ("2008-01-01", "2012-12-31"),
    "2010s": ("2013-01-01", "2019-12-31"),
    "COVID+": ("2020-01-01", "2026-12-31"),  # includes test window: structural analysis only
}


def ensure_dirs():
    for p in (RAW, PROCESSED, FIGURES, TABLES, FORECASTS, STATA_RESULTS):
        p.mkdir(parents=True, exist_ok=True)
