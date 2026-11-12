# dimprobe

How large an embedding dimension do real retrieval benchmarks need, and how far are trained
models from it?

All numbers produced here are **optimizer-reachable dimensions** (upper bounds), not minimum
dimensions.

## Setup (Windows, Python 3.12)

```
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install numpy requests matplotlib pyyaml
```

Use `python -m pip`, not `pip.exe` (the latter is blocked by Application Control on the dev
machine). Verified: Python 3.12.10, torch 2.11.0+cu128, sm_120.

## Layout

```
configs/          experiment configs (YAML)
src/dimprobe/     library code
  solvers.py      fit(rel, d, seed) -> FitResult   (infonce | hinge | centroid)
  search.py       monotone integer search; does not know which solver it calls
  metrics.py      exact / relaxed separation checks
  runlog.py       seed + git sha + GPU + config hash -> JSON
scripts/          entry points (run_sanity.py first)
results/          gitignored, except MANIFEST.md
paper/            LaTeX sources
```

## Sanity check (run first)

```
.venv/Scripts/python scripts/run_sanity.py --config configs/pilot_limit_small.yaml
```

Reproduces, on small cases: (a) the LIMIT k=2 free-embedding critical-n for d <= 16 and
(b) the R2k centroid setting on the same problem.

## Credits

- LIMIT free-embedding protocol: Weller et al., "On the Theoretical Limitations of
  Embedding-Based Retrieval" (github.com/google-deepmind/limit).
- Centroid setting and hinge-loss optimizer: Wang et al., "R^{2k} is Theoretically Large Enough
  for Embedding-based Top-k Retrieval". This repository reimplements them; see the
  docstrings in `src/dimprobe/solvers.py` for every point where the reimplementation may differ.
