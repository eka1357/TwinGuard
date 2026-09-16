"""Tests for Intel Anomalib-aligned visual anomaly detector in TwinGuard."""

import pytest
import numpy as np
import torch

from safety.visual_guard import VisualAnomalyDetector, VisualGuard


def test_visual_anomaly_detector_forward():
    """Verify PyTorch model shape and L2 normalization."""
    model = VisualAnomalyDetector(embedding_dim=64)
    model.eval()

    dummy_input = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        embeddings = model(dummy_input)

    assert embeddings.shape == (2, 64)
    # Check L2 norm is ~1.0
    norms = torch.norm(embeddings, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)


def test_visual_guard_check():
    """Verify visual anomaly score computation."""
    guard = VisualGuard(threshold=0.40)

    # Initial reference frame (e.g. clean table)
    ref_img = np.ones((128, 128, 3), dtype=np.uint8) * 128
    guard.set_reference_state(ref_img)

    # Identical frame should have low anomaly score
    res_clean = guard.check_visual_anomaly(ref_img)
    assert not res_clean["anomaly_detected"]
    assert res_clean["score"] < 0.05

    # Completely different frame (e.g. scene disrupted)
    noisy_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    res_noisy = guard.check_visual_anomaly(noisy_img)
    assert "anomaly_detected" in res_noisy
    assert "score" in res_noisy
