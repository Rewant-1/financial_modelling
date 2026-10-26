* ============================================================================
* SINDy for Financial Modeling - STATA side (Phase 2 + walk-forward export)
* Run this master file; it executes all steps in order.
*   stata> do stata/00_master.do
* Requires: data/processed/master.csv produced by src/02_construct.py
* Outputs:  results/stata/*.txt logs, results/stata/stata_forecasts.csv
*           (auto-merged into the Python horse race by src/10_horse_race.py)
* ============================================================================
clear all
set more off

* Adjust ROOT to the repo location if running from elsewhere
global ROOT "."
global DATA "$ROOT/data/processed"
global OUT  "$ROOT/results/stata"
capture mkdir "$ROOT/results"
capture mkdir "$OUT"

do "$ROOT/stata/01_import_adf.do"
do "$ROOT/stata/02_ar.do"
do "$ROOT/stata/03_garch.do"
do "$ROOT/stata/04_var.do"
do "$ROOT/stata/05_walkforward_garch.do"

display as result "STATA pipeline complete. See $OUT/"
