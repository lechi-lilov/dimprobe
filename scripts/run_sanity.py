"""Sanity check: critical-n vs d on LIMIT-style all-k-subsets qrels, for each configured solver.

Usage:
  .venv/Scripts/python scripts/run_sanity.py --config configs/pilot_limit_small.yaml
  (--dims 2 3 4 --n-max 60 --solvers infonce   for a quick smoke test)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dimprobe import data, runlog, search, solvers  # noqa: E402


def critical_n(solver: str, kw: dict, d: int, k: int, n_max: int, seeds, margin, device, log):
    def pred(n: int) -> bool:
        rel = data.all_subsets_qrels(n, k, device=device)
        for s in seeds:
            r = solvers.fit(rel, d, s, solver=solver, margin=margin, **kw)
            log.append({"solver": solver, "d": d, "n": n, "seed": s, **r.to_dict()})
            print(f"  {solver:8s} d={d:2d} n={n:4d} seed={s} ok={r.success!s:5s} "
                  f"frac_ok={r.frac_ok:.4f} steps={r.steps:5d} {r.seconds:6.1f}s ({r.stop_reason})",
                  flush=True)
            if r.success:
                return True
        return False

    # n <= d is trivially solvable for k=2 (orthonormal docs, query = e_i + e_j): use lo = max(k+1, d).
    lo = max(k + 1, d)
    return search.largest_true(pred, lo, n_max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_limit_small.yaml")
    ap.add_argument("--dims", type=int, nargs="*")
    ap.add_argument("--n-max", type=int)
    ap.add_argument("--solvers", nargs="*")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.dims:
        cfg["dims"] = args.dims
    if args.n_max:
        cfg["n_max"] = args.n_max
    if args.solvers:
        cfg["solvers"] = {s: cfg["solvers"][s] for s in args.solvers}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    log, table = [], {}
    for solver, kw in cfg["solvers"].items():
        table[solver] = {}
        for d in cfg["dims"]:
            n, censored, _ = critical_n(solver, kw, d, cfg["k"], cfg["n_max"], cfg["seeds"],
                                        cfg["margin"], device, log)
            table[solver][d] = {"critical_n": n, "censored": censored}
            print(f"[{solver}] d={d}: critical-n = {'>=' if censored else ''}{n}", flush=True)

    print("\nsummary (critical-n; '>=' means hit n_max):")
    print("d    " + "  ".join(f"{s:>10s}" for s in table))
    for d in cfg["dims"]:
        row = []
        for s in table:
            e = table[s][d]
            row.append(f"{('>=' if e['censored'] else '') + str(e['critical_n']):>10s}")
        print(f"{d:<4d} " + "  ".join(row))
    path = runlog.write("sanity", cfg, {"table": table, "log": log})
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
