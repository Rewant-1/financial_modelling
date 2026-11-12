"""Shared data loading, standardization, and table/figure output helpers."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PROCESSED, TABLES, FIGURES, STATE_COLS

plt.rcParams.update({
    "figure.dpi": 110, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.3, "font.size": 9, "axes.titlesize": 10,
})


def load_master() -> pd.DataFrame:
    """Load the frozen analysis dataset built by 02_construct."""
    path = PROCESSED / "master.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(PROCESSED / "master.csv", parse_dates=["date"], index_col="date")
    return df


def state_matrix(df: pd.DataFrame) -> np.ndarray:
    """(T, 3) array of [r, sigma, vix] in raw daily units."""
    return df[STATE_COLS].to_numpy(float)


def zscore(X: np.ndarray):
    """Return (Z, mu, sd). Standardization stats are always computed on the
    fit window only and reused everywhere else — never leak test statistics."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    return (X - mu) / sd, mu, sd


def apply_zscore(X, mu, sd):
    return (X - mu) / sd


def undo_zscore(Z, mu, sd):
    return Z * sd + mu


def _df_to_markdown(df: pd.DataFrame, index: bool = True) -> str:
    try:
        return df.to_markdown(index=index)
    except Exception:
        cols = list(df.columns)
        if index:
            idx_name = df.index.name if df.index.name else ""
            headers = [str(idx_name)] + [str(c) for c in cols]
            rows = [[str(idx)] + [str(val) for val in row] for idx, row in zip(df.index, df.values)]
        else:
            headers = [str(c) for c in cols]
            rows = [[str(val) for val in row] for row in df.values]
        sep = ["---"] * len(headers)
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
        for r in rows:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, name: str) -> None:
    """Write a results table as CSV plus a Markdown rendering for the report."""
    df.to_csv(TABLES / f"{name}.csv")
    with open(TABLES / f"{name}.md", "w", encoding="utf-8") as fh:
        fh.write(_df_to_markdown(df, index=bool(df.index.name)))


def save_fig(fig, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)
    print(f"  figure -> results/figures/{name}.png")


def save_json(obj, path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def split_masks(df: pd.DataFrame):
    """Frozen train / validation / test boolean masks by date."""
    idx = df.index
    return (
        idx <= pd.Timestamp("2015-12-31"),
        (idx > pd.Timestamp("2015-12-31")) & (idx <= pd.Timestamp("2019-12-31")),
        idx >= pd.Timestamp("2020-01-01"),
    )
