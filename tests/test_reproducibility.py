"""
Test Suite: Reproducibility & Integrity Module
"""

import tempfile
from pathlib import Path
import numpy as np
from src.utils.reproducibility import seed_everything, compute_file_sha256, generate_manifest


def test_seed_everything_deterministic():
    seed_everything(42)
    val1 = np.random.normal(0, 1, 100)
    
    seed_everything(42)
    val2 = np.random.normal(0, 1, 100)
    
    np.testing.assert_array_equal(val1, val2)


def test_compute_file_sha256():
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        f.write("Deterministic Data Provenance Test")
        tmp_path = Path(f.name)
    
    try:
        h1 = compute_file_sha256(tmp_path)
        h2 = compute_file_sha256(tmp_path)
        assert h1 == h2
        assert len(h1) == 64
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_generate_manifest():
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        f.write("Panel Version 1.0")
        tmp_path = Path(f.name)
        
    try:
        manifest = generate_manifest({"test_panel": tmp_path})
        assert "test_panel" in manifest
        assert manifest["test_panel"]["size_bytes"] > 0
        assert "sha256" in manifest["test_panel"]
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
