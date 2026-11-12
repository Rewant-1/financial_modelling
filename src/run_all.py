"""One-command pipeline runner: phases 0-8 in dependency order.

Usage (from repo root, with the project venv active):
    python src/run_all.py            # full run
    FAST=1 python src/run_all.py     # reduced bootstraps/paths for a smoke run
"""
import os
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent

STEPS = [
    ("tests/test_sindy_recovers_garch.py", "Phase 0: synthetic validation"),
    ("01_download.py", "Phase 1a: download raw data"),
    ("02_construct.py", "Phase 1b: construct state dataset"),
    ("03_eda.py", "Phase 1c: exploratory analysis"),
    ("04_diagnostics.py", "Phase 1.5: nonlinearity diagnostics"),
    ("05_econometrics.py", "Phase 2: econometric baselines"),
    ("06_sindy.py", "Phase 3: SINDy identification"),
    ("07_ensemble.py", "Phase 4: robust identification"),
    ("08_ml.py", "Phase 5: ML baselines analysis"),
    ("09_dynamics.py", "Phase 7: dynamics & simulation"),
    ("10_horse_race.py", "Phase 6: forecasting horse race"),
    ("11_varying_splits.py", "Phase 8: multi-split validation"),
    ("12_unsmoothed_vol.py", "Phase 9: unsmoothed volatility analysis"),
]


def main():
    fast = os.environ.get("FAST", "0") == "1"
    env = dict(os.environ)
    if fast:
        env["FAST"] = "1"
    t0 = time.time()
    for script, label in STEPS:
        path = (SRC / script)
        if script.startswith("tests/"):
            path = SRC.parent / script
        print(f"\n{'=' * 64}\n{label}  [{script}]\n{'=' * 64}")
        t = time.time()
        r = subprocess.run([sys.executable, str(path)], env=env)
        if r.returncode != 0:
            print(f"FAILED at {script} (exit {r.returncode}) — stopping")
            return r.returncode
        print(f"-- done in {time.time() - t:.1f}s")
    print(f"\nPipeline complete in {(time.time() - t0) / 60:.1f} min. "
          f"Outputs: results/figures, results/tables, results/forecasts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
