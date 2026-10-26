* ----------------------------------------------------------------------------
* 04 - VAR(p) on the state (r, sigma, vix); lag order by BIC
*      (mirrors VARModel in src/models.py)
* ----------------------------------------------------------------------------
log using "$OUT/04_var.log", replace text

varsoc r sigma vix, maxlag(10)
* read the BIC-optimal order from the table above; example assumes 6:
var r sigma vix, lags(1/6)
estimates store var6
esttab var6 using "$OUT/var_coefs.csv", replace csv se star(* 0.10 ** 0.05 *** 0.01)

log close
