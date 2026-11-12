"""Phase 1a — download raw data (all free sources).

S&P 500 : Yahoo Finance chart API (fallback: yfinance package)
VIX     : FRED VIXCLS (CSV, no API key)
Extras  : FRED DFF (fed funds, for the optional SINDYc control), T10Y3M

Downloads once into data/raw/, records provenance in data/manifest.json.
"""
import json
import sys
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from config import RAW, DATA, START_DATE, ensure_dirs

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def http_get(url: str) -> bytes:
    """Fetch with retries; falls back to curl (FRED sometimes refuses
    Python's default TLS fingerprint)."""
    import time
    last = None
    for attempt in range(1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.read()
        except Exception as e:
            last = e
    import subprocess
    r = subprocess.run(["curl", "-sL", "--max-time", "3", "-A", UA["User-Agent"], url],
                       capture_output=True)
    if r.returncode == 0 and r.stdout:
        return r.stdout
    raise RuntimeError(f"could not fetch {url}: {last}")


def fetch_yahoo_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Daily closes from the Yahoo chart API (no dependencies)."""
    t0 = int(pd.Timestamp(start).timestamp())
    t1 = int(pd.Timestamp(end).timestamp() + 86400 * 2)
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?period1={t0}&period2={t1}&interval=1d&events=history")
    raw = json.loads(http_get(url))
    res = raw["chart"]["result"][0]
    ts = res["timestamp"]
    quote = res["indicators"]["quote"][0]
    close = quote["close"]
    rows = [(datetime.fromtimestamp(t, tz=timezone.utc).date(), c)
            for t, c in zip(ts, close) if c is not None]
    df = pd.DataFrame(rows, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def fetch_cboe_vix() -> pd.DataFrame:
    """Official CBOE VIX history (primary source; FRED merely republishes it)."""
    raw = http_get("https://cdn.cboe.com/api/global/us_indices/"
                   "daily_prices/VIX_History.csv").decode("utf-8")
    from io import StringIO
    df = pd.read_csv(StringIO(raw))
    df["date"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    return df.set_index("date")[["CLOSE"]].rename(columns={"CLOSE": "value"})


def fetch_fred(series: str, start: str) -> pd.DataFrame:
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
           f"&cosd={start}")
    text = http_get(url).decode("utf-8")
    from io import StringIO
    df = pd.read_csv(StringIO(text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
    return df.set_index("date").sort_index()


def main():
    ensure_dirs()
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    manifest = {"downloaded_at": datetime.now(timezone.utc).isoformat(),
                "sources": {}}

    print("downloading S&P 500 (^GSPC) from Yahoo ...")
    spx = None
    try:
        spx = fetch_yahoo_daily("^GSPC", START_DATE, end)
    except Exception as e:
        print(f"  chart API failed ({e}); trying yfinance package ...")
        try:
            import yfinance as yf
            spx = yf.download("^GSPC", start=START_DATE, end=end,
                              auto_adjust=True, progress=False)["Close"]
            spx = spx.rename("close").to_frame()
        except Exception as e2:
            print(f"  yfinance failed ({e2})")
            if (RAW / "spx.csv").exists():
                print("  [offline fallback] reusing existing data/raw/spx.csv")
                spx = pd.read_csv(RAW / "spx.csv", parse_dates=["date"], index_col="date")
            else:
                raise
    spx.to_csv(RAW / "spx.csv")
    manifest["sources"]["spx"] = {
        "provider": "Yahoo Finance", "symbol": "^GSPC",
        "rows": len(spx), "start": str(spx.index.min().date()),
        "end": str(spx.index.max().date())}
    print(f"  {len(spx)} rows: {spx.index.min().date()} -> {spx.index.max().date()}")

    print("downloading VIX from CBOE (FRED fallback) ...")
    vix = None
    try:
        vix = fetch_cboe_vix()
        vix_src = "CBOE VIX_History.csv"
    except Exception as e:
        print(f"  CBOE failed ({e}); falling back to FRED VIXCLS ...")
        try:
            vix = fetch_fred("VIXCLS", START_DATE)
            vix_src = "FRED VIXCLS"
        except Exception as e2:
            print(f"  FRED failed ({e2})")
            if (RAW / "vix.csv").exists():
                print("  [offline fallback] reusing existing data/raw/vix.csv")
                vix = pd.read_csv(RAW / "vix.csv", parse_dates=["date"], index_col="date")
                vix_src = "Existing raw file (offline)"
            else:
                raise
    vix.to_csv(RAW / "vix.csv")
    manifest["sources"]["vix"] = {
        "provider": vix_src, "rows": len(vix),
        "start": str(vix.index.min().date()), "end": str(vix.index.max().date())}
    print(f"  {len(vix)} rows via {vix_src}")

    # optional macro controls (only used by the optional SINDYc extension)
    for series, fname in [("DFF", "dff"), ("T10Y3M", "t10y3m")]:
        try:
            df = fetch_fred(series, START_DATE)
            df.to_csv(RAW / f"{fname}.csv")
            manifest["sources"][fname] = {"provider": "FRED", "series": series,
                                          "rows": int(df["value"].notna().sum())}
            print(f"  {series}: {int(df['value'].notna().sum())} rows")
        except Exception as e:
            print(f"  [warn] {series} unavailable ({e}) — optional, continuing")

    with open(DATA / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("manifest -> data/manifest.json")


if __name__ == "__main__":
    sys.exit(main())
