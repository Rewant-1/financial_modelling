"""Phase 1c — exploratory analysis: the four stylized-fact figures the report
reuses, plus a quantitative stylized-facts table.

(i)   price + realized-vol overlay  -> volatility clustering
(ii)  return histogram vs normal    -> fat tails
(iii) r_t vs next-day sigma change  -> leverage effect; ACF of |r|
(iv)  sigma vs VIX co-movement      -> two distinct volatility objects
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt

from data_utils import load_master, save_table, save_fig, split_masks


def main():
    df = load_master()
    train, val, test = split_masks(df)
    tv = df[train | val]  # stylized facts described on train+val only

    # (i) price + vol overlay
    fig, ax1 = plt.subplots(figsize=(9, 3.5))
    ax1.plot(tv.index, tv["sp_close"], lw=0.6, color="steelblue")
    ax1.set_yscale("log")
    ax1.set_ylabel("S&P 500 (log scale)")
    ax2 = ax1.twinx()
    ax2.plot(tv.index, tv["sigma_ann"], lw=0.5, color="firebrick", alpha=0.85)
    ax2.set_ylabel("realized vol (ann.)", color="firebrick")
    ax2.grid(False)
    for d, lab in [("2008-09-15", "GFC"), ("2020-03-16", "COVID")]:
        if pd.Timestamp(d) in tv.index:
            ax1.axvline(pd.Timestamp(d), color="gray", ls=":", lw=0.8)
    ax1.set_title("S&P 500 and 21-day realized volatility: clustering is visible")
    save_fig(fig, "fig01_price_vol_overlay")

    # (ii) fat tails
    r = tv["r"].values
    jb, jb_p = stats.jarque_bera(r)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    xs = np.linspace(r.min(), r.max(), 400)
    ax.hist(r, bins=120, density=True, alpha=0.65, color="steelblue")
    ax.plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), "k--", lw=1.2,
            label="normal with same mean/var")
    ax.set_yscale("log")
    ax.set_xlabel("daily log return")
    ax.set_title(f"Fat tails: excess kurtosis = {stats.kurtosis(r):.1f}, "
                 f"JB p = {jb_p:.1e}")
    ax.legend()
    save_fig(fig, "fig02_return_histogram")

    # (iii) leverage effect + vol clustering ACF
    d_sig = tv["sigma"].shift(-1) - tv["sigma"]
    valid = (~d_sig.isna()).values
    lev_corr = np.corrcoef(tv["r"][valid], d_sig[valid])[0, 1]
    acf_absr = [tv["abs_r"].autocorr(lag=k) for k in range(1, 22)]
    acf_r = [tv["r"].autocorr(lag=k) for k in range(1, 22)]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9, 3.5))
    axa.scatter(tv["r"][valid], d_sig[valid] * 100, s=2, alpha=0.15,
                color="steelblue")
    binned = pd.DataFrame({"r": tv["r"][valid], "ds": d_sig[valid] * 100})
    binned["bucket"] = pd.qcut(binned["r"], 11)
    agg = binned.groupby("bucket", observed=True).mean()
    axa.plot(agg["r"], agg["ds"], "o-", color="firebrick", ms=4, lw=1.2,
             label="decile means")
    axa.axhline(0, color="gray", lw=0.6)
    axa.set_xlabel("return today  r_t")
    axa.set_ylabel("vol change tomorrow (%)")
    axa.set_title(f"Leverage effect: corr = {lev_corr:.3f}")
    axa.legend()
    axb.bar(range(1, 22), acf_absr, color="firebrick", alpha=0.8,
            label="ACF |r|")
    axb.plot(range(1, 22), acf_r, "s", ms=3, color="steelblue",
             label="ACF r")
    axb.axhline(0, color="gray", lw=0.6)
    axb.set_xlabel("lag (days)")
    axb.set_title("Volatility clustering (|r|) vs no return memory (r)")
    axb.legend()
    save_fig(fig, "fig03_leverage_clustering")

    # (iv) sigma vs VIX
    corr_sv = tv["sigma"].corr(tv["vix"])
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9, 3.5))
    axa.plot(tv.index, tv["sigma"], lw=0.5, color="steelblue", label="realized")
    axa.plot(tv.index, tv["vix"], lw=0.5, color="darkorange", alpha=0.8,
             label="implied (VIX, daily units)")
    axa.legend()
    axa.set_title(f"Realized vs implied volatility (corr = {corr_sv:.3f})")
    axb.scatter(tv["vix"], tv["sigma"], s=2, alpha=0.15, color="steelblue")
    axb.set_xlabel("implied vol (daily units)")
    axb.set_ylabel("realized vol (daily units)")
    lo, hi = tv["vix"].quantile([0.01, 0.99])
    grid = np.linspace(lo, hi, 100)
    b = pd.DataFrame({"v": tv["vix"], "s": tv["sigma"]})
    b["bucket"] = pd.cut(b["v"], 20)
    agg = b.groupby("bucket", observed=True).mean()
    axb.plot(agg["v"], agg["s"], "o-", color="firebrick", ms=3, lw=1)
    axb.plot(grid, grid, "k--", lw=0.8, label="45°")
    axb.legend()
    axb.set_title("VIX usually above realized vol (variance premium)")
    save_fig(fig, "fig04_sigma_vix")

    facts = pd.DataFrame([
        {"stylized_fact": "fat tails (excess kurtosis of r)",
         "value": f"{stats.kurtosis(r):.1f}", "interpretation": ">> 0"},
        {"stylized_fact": "Jarque-Bera p-value", "value": f"{jb_p:.2e}",
         "interpretation": "normality rejected"},
        {"stylized_fact": "ACF |r| at lag 1 / 21",
         "value": f"{acf_absr[0]:.3f} / {acf_absr[20]:.3f}",
         "interpretation": "persistent vol clustering"},
        {"stylized_fact": "ACF r at lag 1", "value": f"{acf_r[0]:.3f}",
         "interpretation": "returns ~unforecastable (EMH)"},
        {"stylized_fact": "corr(r_t, sigma_{t+1}-sigma_t)",
         "value": f"{lev_corr:.3f}",
         "interpretation": "negative: leverage effect"},
        {"stylized_fact": "corr(sigma, vix)", "value": f"{corr_sv:.3f}",
         "interpretation": "related but distinct objects"},
        {"stylized_fact": "mean VIX - mean sigma (premium)",
         "value": f"{(tv['vix'] - tv['sigma']).mean():.5f}",
         "interpretation": "variance risk premium > 0"},
    ])
    save_table(facts, "stylized_facts")
    print(facts.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
