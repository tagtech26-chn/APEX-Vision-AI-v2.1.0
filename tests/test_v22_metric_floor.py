from __future__ import annotations

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
