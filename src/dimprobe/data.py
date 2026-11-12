"""Relevance matrices.

A relevance matrix is a dense bool tensor of shape (num_queries, num_docs); True = relevant.
BEIR loaders (official zips via requests + json, no pyarrow) will be added after the sanity check.
"""
from __future__ import annotations

import itertools

import torch


def all_subsets_qrels(n: int, k: int = 2, device: str | torch.device = "cpu") -> torch.Tensor:
    """LIMIT-style synthetic qrels: one query per k-subset of n docs, relevant set = that subset.

    Returns a bool tensor of shape (C(n, k), n).
    """
    if k == 2:
        idx = torch.combinations(torch.arange(n, device=device), r=2)
    else:
        idx = torch.tensor(list(itertools.combinations(range(n), k)), device=device)
    rel = torch.zeros(idx.shape[0], n, dtype=torch.bool, device=device)
    rel.scatter_(1, idx, True)
    return rel
