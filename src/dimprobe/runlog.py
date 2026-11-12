"""Run provenance: seed(s), git SHA (+dirty flag), GPU, torch version, config hash -> JSON."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def provenance(cfg: dict) -> dict:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": _git("rev-parse", "HEAD") or "no-commit",
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": gpu,
        "config_hash": config_hash(cfg),
        "config": cfg,
    }


def write(name: str, cfg: dict, results) -> Path:
    out_dir = REPO / "results"
    out_dir.mkdir(exist_ok=True)
    rec = provenance(cfg)
    rec["results"] = results
    path = out_dir / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}_{rec['config_hash']}.json"
    path.write_text(json.dumps(rec, indent=2))
    return path
