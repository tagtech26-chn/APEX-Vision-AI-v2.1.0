from __future__ import annotations

import os

import numpy as np

from app.ai.geometry.metric_floor import MetricFloorEstimator


def test_metric_floor_recovers_planar_surface() -> None:
    h, w = 120, 180
    yy, xx = np.mgrid[0:h, 0:w]
    z = 2.0 + 0.0015 * yy + 0.0007 * xx
    depth = z.astype(np.float32)
    mask = np.full((h, w), 255, dtype=np.uint8)
    mask[:10] = 0
    mask[-10:] = 0

    result = MetricFloorEstimator(sample_size=5000).estimate(mask, depth)

    assert result.inlier_ratio > 0.9
    assert result.residual_p95 < 0.01
    assert np.isfinite(result.equation).all()
    assert np.isclose(np.linalg.norm(result.normal), 1.0, atol=1e-5)


def test_metric3d_provider_executes_on_cpu_without_optional_mmcv() -> None:
    """CPU development must not fail the render pipeline when MMCV is absent."""
    os.environ["APEX_V22_DEPTH_FALLBACK"] = "1"
    from app.ai.depth.metric3d import Metric3DProvider

    provider = Metric3DProvider(device="cpu")
    image = np.zeros((96, 144, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(20, 180, 144, dtype=np.uint8)[None, :]
    depth = provider.predict(image)

    assert depth.shape == image.shape[:2]
    assert depth.dtype == np.float32
    assert np.isfinite(depth).all()
    assert float(depth.min()) > 0.0
    assert float(depth.max()) > float(depth.min())
    assert provider.last_metric_depth is not None
    assert provider.last_metric_depth.shape == image.shape[:2]
