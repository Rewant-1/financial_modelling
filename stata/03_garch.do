* ----------------------------------------------------------------------------
* 03 - GARCH family on returns: GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1)
*      (mirrors src/05_econometrics.py and the garch.py QMLE fits)
* ----------------------------------------------------------------------------
log using "$OUT/03_garch.log", replace text

* GARCH(1,1)
arch r, arch(1) garch(1)
estimates store garch11
esttab garch11 using "$OUT/garch11_coefs.csv", replace csv se star(* 0.10 ** 0.05 *** 0.01)

* EGARCH(1,1) - asymmetry by specification
arch r, earch(1) egarch(1)
estimates store egarch
esttab egarch using "$OUT/egarch_coefs.csv", replace csv se star(* 0.10 ** 0.05 *** 0.01)

* GJR-GARCH(1,1) - simple asymmetric term ('sa')
arch r, arch(1) garch(1) sa
estimates store gjr
esttab gjr using "$OUT/gjr_coefs.csv", replace csv se star(* 0.10 ** 0.05 *** 0.01)

log close
