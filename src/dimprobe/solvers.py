"""Free-embedding solvers. One interface: fit(rel, d, seed, ...) -> FitResult.

search.py must not know which solver it calls; pass a solver name + kwargs.

All solvers: full batch, Adam, unit-norm vectors kept by projection (renormalise after every
step), early stop after `patience` steps without loss improvement, hard cap `max_steps`.
Success is checked on the scores of every forward pass (before the update), so the reported
configuration is one that was actually reached.

REIMPLEMENTATION CAVEATS -- verify against the original papers/code before citing numbers:
- infonce  (LIMIT, Weller et al.): multi-positive InfoNCE where each positive competes only
  against the non-relevant docs. Temperature, lr and the exact multi-positive form used by
  LIMIT are NOT verified; check github.com/google-deepmind/limit.
- hinge    (R2k-style, Wang et al.): my own vectorised "hardest counterpart" hinge
  (every negative vs the hardest positive + every positive vs the hardest negative). The R2k
  appendix code may use a different pairing; compare before claiming equivalence.
- centroid (R2k, Wang et al.): query = L2-normalised mean of its relevant doc vectors; only doc
  vectors are optimised; trained with the hinge above. Normalising the query does not change
  rankings but does change the margin scale -- state this in the paper.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from . import metrics

SOLVERS = ("infonce", "hinge", "centroid")


@dataclass
class FitResult:
    success: bool  # exact success at the evaluation margin
    success_relaxed: bool
    frac_ok: float
    min_margin: float
    steps: int
    seconds: float
    stop_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _infonce_loss(S, rel, tau):
    # per positive (q, p): -log( e^{s_qp/t} / (e^{s_qp/t} + sum_{neg} e^{s_qn/t}) )
    logits = S / tau
    lse_neg = torch.where(rel, torch.full_like(logits, float("-inf")), logits).logsumexp(dim=1)
    per_pos = F.softplus(lse_neg[:, None] - logits)
    return per_pos[rel].mean()


def _hinge_loss(S, rel, gamma):
    big = torch.finfo(S.dtype).max / 4
    min_pos = torch.where(rel, S, torch.full_like(S, big)).amin(dim=1, keepdim=True)
    max_neg = torch.where(rel, torch.full_like(S, -big), S).amax(dim=1, keepdim=True)
    neg_term = torch.where(rel, torch.zeros_like(S), F.relu(gamma + S - min_pos))
    pos_term = torch.where(rel, F.relu(gamma + max_neg - S), torch.zeros_like(S))
    return (neg_term.sum() + pos_term.sum()) / S.shape[0]


def fit(
    rel: torch.Tensor,
    d: int,
    seed: int,
    solver: str = "infonce",
    margin: float = 0.0,
    lr: float = 0.01,
    max_steps: int = 20000,
    patience: int = 1000,
    tau: float = 0.1,
    train_margin: float | None = None,
    min_improve: float = 1e-6,
) -> FitResult:
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}; choose from {SOLVERS}")
    if train_margin is None:
        train_margin = margin + 0.01
    device = rel.device
    m, n = rel.shape
    gen = torch.Generator(device=device).manual_seed(seed)

    D = F.normalize(torch.randn(n, d, generator=gen, device=device), dim=1).requires_grad_(True)
    params = [D]
    if solver == "centroid":
        relf = rel.float()
        Q = None
    else:
        Q = F.normalize(torch.randn(m, d, generator=gen, device=device), dim=1).requires_grad_(True)
        params.append(Q)
    opt = torch.optim.Adam(params, lr=lr)

    t0 = time.perf_counter()
    best_loss, since_best = float("inf"), 0
    summary, stop_reason, step = None, "max_steps", 0
    for step in range(max_steps):
        q = F.normalize(relf @ D, dim=1) if solver == "centroid" else Q
        S = q @ D.T
        summary = metrics.summarize(metrics.query_margins(S.detach(), rel), margin)
        if summary["success_exact"]:
            stop_reason = "success"
            break
        if solver == "infonce":
            loss = _infonce_loss(S, rel, tau)
        else:
            loss = _hinge_loss(S, rel, train_margin)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p in params:
                p.copy_(F.normalize(p, dim=1))
        lv = loss.item()
        if lv < best_loss - min_improve:
            best_loss, since_best = lv, 0
        else:
            since_best += 1
            if since_best >= patience:
                stop_reason = "patience"
                break
    if stop_reason != "success":  # evaluate the final parameters too
        with torch.no_grad():
            q = F.normalize(relf @ D, dim=1) if solver == "centroid" else Q
            summary = metrics.summarize(metrics.query_margins(q @ D.T, rel), margin)
            if summary["success_exact"]:
                stop_reason = "success"
    if device.type == "cuda":
        torch.cuda.synchronize()
    return FitResult(
        success=summary["success_exact"],
        success_relaxed=summary["success_relaxed"],
        frac_ok=summary["frac_ok"],
        min_margin=summary["min_margin"],
        steps=step + 1,
        seconds=time.perf_counter() - t0,
        stop_reason=stop_reason,
    )
