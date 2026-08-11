"""V2.2 scene analyzer: semantic segmentation + metric 3D geometry."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.ai.geometry.metric_floor import MetricFloorEstimator
from app.ai.scene.analyzer import SceneAnalyzer

logger = logging.getLogger("apex.ai")


class V22SceneAnalyzer(SceneAnalyzer):
    """Refine the existing Heavy-AI scene with metric floor geometry.

    Metric depth and normals are geometric evidence only. They never create
    floor pixels outside the semantic floor segmentation, preventing walls,
    cabinets and furniture from becoming synthetic floor regions.
    """

    def __init__(self, detector, segmenter, depth) -> None:
        super().__init__(detector=detector, segmenter=segmenter, depth=depth)
        self.metric_floor = MetricFloorEstimator()

    @staticmethod
    def _normal_consistency_mask(normals: np.ndarray, seed_mask: np.ndarray, minimum_cosine: float = 0.82) -> np.ndarray:
        if normals is None or normals.shape[:2] != seed_mask.shape[:2]:
            return np.ones(seed_mask.shape, dtype=bool)
        valid = (seed_mask > 0) & np.isfinite(normals).all(axis=2)
        if int(valid.sum()) < 100:
            return np.ones(seed_mask.shape, dtype=bool)
        samples = normals[valid].astype(np.float32)
        lengths = np.linalg.norm(samples, axis=1)
        samples = samples[lengths > 1e-5]
        if len(samples) < 100:
            return np.ones(seed_mask.shape, dtype=bool)
        target = np.median(samples, axis=0)
        target /= max(float(np.linalg.norm(target)), 1e-6)
        cosine = np.sum(normals.astype(np.float32) * target[None, None, :], axis=2)
        positive = cosine >= minimum_cosine
        negative = cosine <= -minimum_cosine
        return negative if int((negative & valid).sum()) > int((positive & valid).sum()) else positive

    @staticmethod
    def _refine_floor_mask(seed: np.ndarray, residual: np.ndarray, normals: np.ndarray | None, residual_threshold: float) -> np.ndarray:
        seed_bool = seed > 0
        refined = seed_bool & np.isfinite(residual) & (residual <= residual_threshold)
        if normals is not None and normals.shape[:2] == seed.shape[:2]:
            refined &= V22SceneAnalyzer._normal_consistency_mask(normals, refined)
        result = refined.astype(np.uint8) * 255
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        result = cv2.bitwise_and(result, seed.astype(np.uint8))
        if (result > 0).any():
            result = V22SceneAnalyzer._largest_component(result)
        return result

    def analyze(self, image_path, progress_cb=None):
        scene = super().analyze(image_path, progress_cb=progress_cb)
        metric_depth = getattr(self.depth, "last_metric_depth", None)
        if metric_depth is None:
            raise RuntimeError("V2.2 geometry requires a metric-depth provider.")

        points = getattr(self.depth, "last_points", None)
        intrinsics = getattr(self.depth, "last_intrinsics", None)
        normals = getattr(self.depth, "last_normals", None)
        metric = self.metric_floor.estimate(scene.floor_mask, metric_depth, points=points, intrinsics=intrinsics)

        if points is not None:
            xyz = points.squeeze(0) if points.ndim == 4 else points
        else:
            xyz = self.metric_floor._points_from_depth(metric_depth, self.metric_floor._focal(intrinsics))

        residual = np.abs(np.tensordot(xyz, metric.normal, axes=([2], [0])) + metric.equation[3])
        floor_residual = residual[scene.floor_mask > 0]
        if floor_residual.size:
            p90 = float(np.percentile(floor_residual, 90))
            p97 = float(np.percentile(floor_residual, 97))
            threshold = min(max(p90 * 1.75, 0.012), max(p97 * 1.25, 0.025), 0.10)
        else:
            threshold = 0.05

        original_area = int((scene.floor_mask > 0).sum())
        refined = self._refine_floor_mask(scene.floor_mask, residual, normals, threshold)
        refined_area = int((refined > 0).sum())

        # Guardrail only: preserve Heavy-AI semantic segmentation if geometric
        # refinement collapses the floor to an unusably small region.
        if original_area >= 500 and refined_area < max(250, int(original_area * 0.12)):
            logger.warning("[V2.2] Metric refinement rejected: area collapsed %d -> %d pixels.", original_area, refined_area)
            refined = scene.floor_mask.copy()
            refined_area = original_area

        polygon = self.polygon_engine.extract(refined)
        homography = self.homography_engine.compute(polygon)
        scene.floor_mask = refined
        scene.floor_polygon = polygon
        scene.homography = homography.matrix
        scene.floor_plane.normal = metric.normal
        scene.floor_plane.distance = float(metric.equation[3])
        scene.camera_pose.pitch = metric.pitch
        scene.camera_pose.roll = metric.roll
        scene.metadata["v22_geometry"] = {
            "depth_provider": self.depth.name,
            "segmenter": self.segmenter.name,
            "metric_plane_residual_p95_m": metric.residual_p95,
            "metric_plane_inlier_ratio": metric.inlier_ratio,
            "metric_plane_threshold_m": threshold,
            "metric_points": points is not None,
            "surface_normals": normals is not None,
            "semantic_area_px": original_area,
            "refined_area_px": refined_area,
            "refined_area_ratio": (refined_area / original_area) if original_area else 0.0,
            "bottom_edge_forced": False,
        }
        return scene
