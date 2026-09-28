"""Reproduce every number, table and figure of the paper from the logged experiment data (no model calls).

usage:  python analysis/run_all.py [--jobs 4] [--splits 200]

Steps: pooled policy analysis for each model x evidence version x alpha (analysis/pooled.py), formal tests and
task-level conformal calibration (analysis/stats.py), then the per-run table and figures (analysis/figures.py).
Everything is written to runs/analysis/. The full analysis (200 splits) takes well under an hour on a laptop.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ["gpt-5-mini", "gpt-5", "claude-haiku-4-5"]


def run(args):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run([sys.executable, *args], cwd=HERE, env=env, capture_output=True, text=True)
    print(" ".join(args), "->", "ok" if r.returncode == 0 else "FAILED", flush=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-4000:])
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--splits", type=int, default=200)
    a = ap.parse_args()
    jobs = [["pooled.py", m, ev, "--alpha", str(al), "--tag", tag, "--splits", str(a.splits)]
            for ev in ("v2", "v1") for al, tag in ((0.02, "main"), (0.01, "a01")) for m in MODELS]
    jobs += [["stats.py", "v2"], ["stats.py", "v1"]]
    with ThreadPoolExecutor(a.jobs) as ex:
        codes = list(ex.map(run, jobs))
    if any(codes):
        sys.exit("some analysis steps failed; see the messages above")
    sys.exit(run(["figures.py"]))


if __name__ == "__main__":
    main()
