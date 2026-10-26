* ----------------------------------------------------------------------------
* 05 - QUARTERLY walk-forward GARCH(1,1) forecasts over the frozen test
*      window, exported in the format src/10_horse_race.py auto-merges:
*      results/stata/stata_forecasts.csv  with columns: date,h,forecast
*      (forecast = conditional volatility, DAILY units, sqrt of variance)
*
* Protocol matches the Python side: expanding window, refit each quarter,
* daily origins within the quarter, horizons h = 1, 5, 21.
*
* NOTE: mirrors the Python protocol; verify on your STATA version
* (`predict, variance(dynamic())` gives the iterated sigma2 path from the
* origin, so element i+h is the h-step variance forecast).
* ----------------------------------------------------------------------------
clear
import delimited "$DATA/master.csv", varnames(1) case(preserve) stringcols(1)
rename v1 date_str
gen date = date(date_str, "YMD")
format date %td
drop date_str
tsset date
gen t = _n

* extend the calendar 21 trading days past the sample so h-step dynamic
* predictions have somewhere to land
tsappend, add(21)
quietly count
local T = r(N)

tempname results
postfile `results' str10 origin_date int h double(forecast) ///
    using "$OUT/wf_tmp", replace

local prev_q = -1
quietly levelsof t if date >= date("2020-01-01", "YMD"), local(testts)
foreach i of local testts {
    quietly summarize date if t == `i'
    local d = r(mean)
    local q = quarter(`d') + 10 * year(`d')
    if `q' != `prev_q' {
        * new quarter: refit on all data strictly before this origin
        local nfit = `i' - 1
        quietly arch r in 1/`nfit', arch(1) garch(1)
        local prev_q = `q'
    }
    if `i' + 21 <= `T' {
        tempvar vf
        quietly predict double `vf' = variance, dynamic(`i')
        foreach h in 1 5 21 {
            local j = `i' + `h'
            post `results' (string(`d', "%tdCCYY-NN-DD")) (`h') (sqrt(`vf'[`j']))
        }
    }
}
postclose `results'

preserve
use "$OUT/wf_tmp", clear
export delimited using "$OUT/stata_forecasts.csv", replace
erase "$OUT/wf_tmp.dta"
restore
display as result "walk-forward GARCH forecasts -> $OUT/stata_forecasts.csv"
