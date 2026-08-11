"""Metric 3D floor geometry for APEX Vision AI v2.2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class MetricFloorResult:
    normal: np.ndarray
    centroid: np.ndarray
    equation: np.ndarray
    pitch: float
    roll: float
    residual_p95: float
    inlier_ratio: float


class MetricFloorEstimator:
    """Fits a robust plane in camera coordinates from metric depth or 3D points."""

    def __init__(self, sample_size: int = 12000, iterations: int = 4) -> None:
        self.sample_size = sample_size
        self.iterations = iterations

    @staticmethod
    def _points_from_depth(depth: np.ndarray, focal: float | None = None) -> np.ndarray:
        h, w = depth.shape[:2]
        focal = float(focal or (0.9 * max(w, h)))
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        yy, xx = np.mgrid[0:h, 0:w]
        z = depth.astype(np.float32)
        x = (xx.astype(np.float32) - cx) * z / focal
        y = (yy.astype(np.float32) - cy) * z / focal
        return np.stack((x, y, z), axis=-1)

    def estimate(
        self,
        floor_mask: np.ndarray,
        metric_depth: np.ndarray,
        points: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
    ) -> MetricFloorResult:
        if metric_depth is None:
            raise ValueError("Metric floor estimation requires metric depth.")
        xyz = points if points is not None else self._points_from_depth(metric_depth, self._focal(intrinsics))
        if xyz.ndim == 4:
            xyz = xyz.squeeze(0)
        if xyz.shape[:2] != floor_mask.shape[:2]:
            raise ValueError("3D points and floor mask dimensions differ.")

        mask = (floor_mask > 0) & np.isfinite(xyz).all(axis=2) & (xyz[:, :, 2] > 0.05)
        ys, xs = np.where(mask)
        if len(xs) < 100:
            raise ValueError("Too few metric floor points for plane fitting.")
        if len(xs) > self.sample_size:
            rng = np.random.default_rng(42)
            keep = rng.choice(len(xs), self.sample_size, replace=False)
            xs, ys = xs[keep], ys[keep]
        sample = xyz[ys, xs].astype(np.float32)

        inliers = np.ones(len(sample), dtype=bool)
        for _ in range(self.iterations):
            fit = sample[inliers]
            centroid = fit.mean(axis=0)
            _, _, vh = np.linalg.svd(fit - centroid, full_matrices=False)
            normal = vh[-1]
            normal /= max(np.linalg.norm(normal), 1e-8)
            distances = np.abs((sample - centroid) @ normal)
            threshold = max(float(np.percentile(distances[inliers], 85)) * 1.8, 0.008)
            inliers = distances <= threshold

        fit = sample[inliers]
        centroid = fit.mean(axis=0)
        _, _, vh = np.linalg.svd(fit - centroid, full_matrices=False)
        normal = vh[-1]
        normal /= max(np.linalg.norm(normal), 1e-8)
        if normal[1] > 0:
            normal = -normal
        d = -float(normal @ centroid)
        residuals = np.abs((sample - centroid) @ normal)
        p95 = float(np.percentile(residuals, 95))
        pitch = float(np.degrees(np.arctan2(normal[1], np.sqrt(normal[0] ** 2 + normal[2] ** 2))))
        roll = float(np.degrees(np.arctan2(normal[0], normal[2])))
        return MetricFloorResult(
            normal=normal.astype(np.float32),
            centroid=centroid.astype(np.float32),
            equation=np.array([*normal, d], dtype=np.float32),
            pitch=pitch,
            roll=roll,
            residual_p95=p95,
            inlier_ratio=float(inliers.mean()),
        )

    @staticmethod
    def _focal(intrinsics: np.ndarray | None) -> float | None:
        if intrinsics is None:
            return None
        k = np.asarray(intrinsics)
        if k.shape == (3, 3):
            return float((k[0, 0] + k[1, 1]) * 0.5)
        if k.size >= 4:
            return float((k.flat[0] + k.flat[1]) * 0.5)
        return None
