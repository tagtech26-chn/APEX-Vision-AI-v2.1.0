"""V2.2 scene analyzer: semantic segmentation + metric 3D geometry."""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from app.ai.geometry.metric_floor import MetricFloorEstimator
from app.ai.scene.analyzer import SceneAnalyzer

logger = logging.getLogger("apex.ai")


class V22SceneAnalyzer(SceneAnalyzer):
    """Refine the existing scene with metric geometry and optional Gemini spatial guidance.

    Gemini is used only as a high-level spatial advisor. Deterministic OpenCV
    geometry remains authoritative, and every Gemini floor quad is validated
    against the local semantic floor mask before it can affect rendering.
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

    @staticmethod
    def _apply_gemini_floor_quad(scene, image_path, protected_mask):
        """Use Gemini's spatial floor quad only after deterministic validation."""
        enabled = os.getenv("APEX_V22_GEOMETRY_ADVISOR", "off").strip().lower()
        if enabled not in {"gemini", "google", "gemini_er"}:
            scene.metadata["spatial_advisor"] = {"provider": "disabled", "accepted": False}
            return scene

        try:
            from app.ai.geometry.gemini_spatial import GeminiSpatialAdvisor

            advisor = GeminiSpatialAdvisor()
            if not advisor.available:
                scene.metadata["spatial_advisor"] = {
                    "provider": advisor.name,
                    "accepted": False,
                    "reason": "GEMINI_API_KEY or google-genai is not configured",
                }
                return scene

            result = advisor.analyze(image_path)
            h, w = scene.height, scene.width
            quad = np.asarray(result["floor_quad"], dtype=np.float32)
            quad[:, 0] *= float(w - 1)
            quad[:, 1] *= float(h - 1)

            quad_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(quad_mask, np.round(quad).astype(np.int32), 255)
            semantic = scene.floor_mask > 0
            proposed = quad_mask > 0
            intersection = int((semantic & proposed).sum())
            semantic_area = int(semantic.sum())
            quad_area = int(proposed.sum())
            coverage = intersection / max(quad_area, 1)
            semantic_capture = intersection / max(semantic_area, 1)

            # Gemini may see the whole floor but must still agree with local
            # segmentation. These thresholds prevent hallucinated walls/ceiling.
            accepted = (
                coverage >= 0.55
                and semantic_capture >= 0.15
                and quad_area >= max(1000, int(h * w * 0.04))
            )
            if not accepted:
                scene.metadata["spatial_advisor"] = {
                    "provider": advisor.name,
                    "accepted": False,
                    "confidence": result["confidence"],
                    "semantic_overlap": coverage,
                    "semantic_capture": semantic_capture,
                    "reason": "quad rejected by local semantic-mask guardrails",
                }
                logger.warning(
                    "[V2.2] Gemini floor quad rejected: overlap=%.3f capture=%.3f confidence=%.3f",
                    coverage,
                    semantic_capture,
                    result["confidence"],
                )
                return scene

            fused = quad_mask
            if protected_mask is not None and (protected_mask > 0).any():
                fused = cv2.subtract(fused, protected_mask.astype(np.uint8))
            fused = V22SceneAnalyzer._largest_component(fused)
            polygon = V22SceneAnalyzer._quad_to_polygon(quad)
            homography = scene.homography
            try:
                homography_result = scene_analyzer_homography(scene, polygon)
                homography = homography_result.matrix
            except Exception as exc:
                logger.warning("[V2.2] Gemini homography recompute failed; keeping local homography: %s", exc)

            scene.floor_mask = fused
            scene.floor_polygon = polygon
            scene.homography = homography
            scene.metadata["spatial_advisor"] = {
                "provider": advisor.name,
                "model": advisor.model,
                "accepted": True,
                "confidence": result["confidence"],
                "semantic_overlap": coverage,
                "semantic_capture": semantic_capture,
                "notes": result.get("notes", ""),
            }
            logger.info(
                "[V2.2] Gemini spatial floor quad accepted confidence=%.3f overlap=%.3f capture=%.3f",
                result["confidence"],
                coverage,
                semantic_capture,
            )
        except Exception as exc:
            logger.warning("[V2.2] Gemini spatial advisor unavailable; keeping local geometry: %s", exc)
            scene.metadata["spatial_advisor"] = {
                "provider": "gemini_robotics_er_1.6",
                "accepted": False,
                "reason": str(exc),
            }
        return scene

    @staticmethod
    def _quad_to_polygon(quad: np.ndarray) -> np.ndarray:
        """Return a stable float32 quadrilateral for the existing polygon engine."""
        from app.ai.geometry.polygon import PolygonEngine

        ordered = PolygonEngine.order_points(quad.astype(np.float32))
        return ordered.astype(np.float32)

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

        # Run the remote spatial advisor last so it can improve the floor quad
        # without becoming a dependency of the local CPU validation path.
        return self._apply_gemini_floor_quad(scene, image_path, scene.protected_object_mask)


def scene_analyzer_homography(scene, polygon):
    """Compute homography using the same engine already owned by SceneAnalyzer."""
    from app.ai.geometry.homography import HomographyEngine

    return HomographyEngine().compute(polygon)
