* ----------------------------------------------------------------------------
* 01 - import the frozen master dataset; unit-root tests (the econometric
*      hygiene step; mirrors results/tables/adf_tests.csv from Python)
* ----------------------------------------------------------------------------
clear
import delimited "$DATA/master.csv", varnames(1) case(preserve) stringcols(1)
* master.csv: first column = ISO date string
rename v1 date_str
gen date = date(date_str, "YMD")
format date %td
drop date_str
tsset date

* the state variables (VIX already converted to daily units in Python)
* r  = daily log return, sigma = 21-day realized vol, vix = implied (daily)

log using "$OUT/01_adf.log", replace text
foreach v in r sigma vix sp_close {
    dfuller `v', lags(aic)
}
log close
