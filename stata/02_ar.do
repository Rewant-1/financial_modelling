* ----------------------------------------------------------------------------
* 02 - AR(p) on realized vol, lag order by BIC (mirrors ARSigma in Python)
* ----------------------------------------------------------------------------
log using "$OUT/02_ar.log", replace text

estimates clear
forvalues p = 1/10 {
    quietly regress sigma L(1/`p').sigma
    estimates store ar`p'
}
estimates stats ar1-ar10
* choose min BIC manually from the table above; example assumes p=6:
quietly regress sigma L(1/6).sigma
estimates store AR_final

* export coefficient table
esttab AR_final using "$OUT/ar_sigma_coefs.csv", replace csv se star(* 0.10 ** 0.05 *** 0.01)
log close
