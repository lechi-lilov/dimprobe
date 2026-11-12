"""Separation metrics on a score matrix.

Definitions (keep in sync with the paper):
- per-query margin  = min score over relevant docs - max score over non-relevant docs.
- exact success at margin g: every query has margin >= g (for g == 0: margin > 0, strictly).
- relaxed success: Recall@|R_q| == 1 for every query, ties counted as failures.
  With ties counted as failures this coincides with exact success at g == 0; the two only
  differ when g > 0. Recall at a fixed cut-off (e.g. Recall@10) is a separate, weaker metric.
Queries with no relevant docs, or with every doc relevant, are ignored (margin = +inf).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def query_margins(scores: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
    inf = torch.tensor(float("inf"), device=scores.device, dtype=scores.dtype)
    min_rel = torch.where(rel, scores, inf).amin(dim=1)
    max_non = torch.where(rel, -inf, scores).amax(dim=1)
    m = min_rel - max_non
    degenerate = ~rel.any(dim=1) | rel.all(dim=1)
    return torch.where(degenerate, inf, m)


def ok_mask(margins: torch.Tensor, g: float) -> torch.Tensor:
    return margins > 0 if g == 0 else margins >= g


def summarize(margins: torch.Tensor, g: float) -> dict:
    ok = ok_mask(margins, g)
    finite = margins[torch.isfinite(margins)]
    return {
        "success_exact": bool(ok.all()),
        "success_relaxed": bool((margins > 0).all()),
        "frac_ok": float(ok.float().mean()),
        "min_margin": float(finite.min()) if finite.numel() else float("inf"),
    }
