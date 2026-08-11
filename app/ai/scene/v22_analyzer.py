"""V2.2 scene analyzer: semantic segmentation + metric 3D geometry."""

from __future__ import annotations

import cv2
import numpy as np

from app.ai.geometry.metric_floor import MetricFloorEstimator
from app.ai.scene.analyzer import SceneAnalyzer


class V22SceneAnalyzer(SceneAnalyzer):
    """Refines the existing scene pipeline using metric depth/3D geometry."""

    def __init__(self, detector, segmenter, depth) -> None:
        super().__init__(detector=detector, segmenter=segmenter, depth=depth)
        self.metric_floor = MetricFloorEstimator()

    def analyze(self, image_path, progress_cb=None):
        scene = super().analyze(image_path, progress_cb=progress_cb)
        metric_depth = getattr(self.depth, "last_metric_depth", None)
        if metric_depth is None:
            raise RuntimeError("V2.2 geometry requires a metric-depth provider.")

        points = getattr(self.depth, "last_points", None)
        intrinsics = getattr(self.depth, "last_intrinsics", None)
        normals = getattr(self.depth, "last_normals", None)
        metric = self.metric_floor.estimate(
            scene.floor_mask,
            metric_depth,
            points=points,
            intrinsics=intrinsics,
        )

        h, w = scene.floor_mask.shape
        if points is not None:
            xyz = points.squeeze(0) if points.ndim == 4 else points
        else:
            xyz = self.metric_floor._points_from_depth(metric_depth, self.metric_floor._focal(intrinsics))
        residual = np.abs(np.tensordot(xyz, metric.normal, axes=([2], [0])) + metric.equation[3])
        floor_residual = residual[scene.floor_mask > 0]
        threshold = max(float(np.percentile(floor_residual, 92)) * 1.8, 0.015) if floor_residual.size else 0.05
        refined = ((scene.floor_mask > 0) & (residual <= threshold)).astype(np.uint8) * 255

        if normals is not None and normals.shape[:2] == (h, w):
            valid_normals = normals[scene.floor_mask > 0]
            target = np.median(valid_normals, axis=0)
            target /= max(np.linalg.norm(target), 1e-6)
            cosine = np.sum(normals * target[None, None, :], axis=2)
            refined = ((refined > 0) & (cosine >= 0.72)).astype(np.uint8) * 255

        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        refined[h - max(8, h // 35) :, :] = 255
        refined = self._largest_component(refined)

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
        }
        return scene
