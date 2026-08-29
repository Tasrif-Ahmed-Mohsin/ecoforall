"""
Reproducibility, Seed Management & Artifact Checksum Module
=============================================================
Enforces deterministic random seeds across NumPy, PyTorch, Python random,
and generates SHA-256 integrity hashes for data files and benchmark artifacts.
"""

from __future__ import annotations
import os
import random
import hashlib
from pathlib import Path
from typing import Union, Dict, Any
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set seeds globally across all execution environments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file for data provenance and versioning."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    sha256 = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_manifest(files: Dict[str, Union[str, Path]]) -> Dict[str, Dict[str, Any]]:
    """Generate a structured manifest with file sizes and SHA-256 checksums."""
    manifest = {}
    for name, path in files.items():
        p = Path(path)
        if p.exists():
            manifest[name] = {
                "path": str(p),
                "size_bytes": p.stat().st_size,
                "sha256": compute_file_sha256(p),
            }
        else:
            manifest[name] = {"path": str(p), "status": "MISSING"}
    return manifest
