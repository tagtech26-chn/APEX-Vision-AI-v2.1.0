"""Scene analysis orchestration and provider factory."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from app.ai.config import heavy_models_available, resolve_provider
from app.ai.depth.base import DepthEstimator
from app.ai.detection.base import Detection, ObjectDetector
from app.ai.geometry.homography import HomographyEngine
from app.ai.geometry.plane import PlaneEstimator, PlaneResult
from app.ai.geometry.polygon import PolygonEngine
from app.ai.scene.result import SceneResult
from app.ai.segmentation.base import Segmenter

logger = logging.getLogger("apex.ai")
_ANALYZER_LOCK = threading.Lock()


class SceneAnalyzer:
    """Runs detection -> segmentation -> depth -> geometry to build a SceneResult."""

    def __init__(self, detector: ObjectDetector, segmenter: Segmenter, depth: DepthEstimator) -> None:
        self.detector = detector
        self.segmenter = segmenter
        self.depth = depth
        self.polygon_engine = PolygonEngine()
        self.homography_engine = HomographyEngine()
        self.plane_estimator = PlaneEstimator()

    @property
    def providers(self) -> dict[str, str]:
        return {"detector": self.detector.name, "segmenter": self.segmenter.name, "depth": self.depth.name}

    @staticmethod
    def _downscale(image: np.ndarray, max_dim: int | None = None) -> np.ndarray:
        from app.core.config import settings
        limit = max_dim or settings.render_max_dim
        height, width = image.shape[:2]
        largest = max(height, width)
        if largest <= limit:
            return image
        scale = limit / largest
        return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    def _carve_obstructions(self, image: np.ndarray, floor_mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]], np.ndarray]:
        """Subtract foreground objects while preserving their exact occlusion mask."""
        empty = np.zeros_like(floor_mask)
        if not hasattr(self.segmenter, "segment_many"):
            return floor_mask, [], empty
        try:
            detections = self.detector.detect(image, "sofa. couch. armchair. chair. table. rug. plant. lamp. cabinet.")
        except Exception as exc:
            logger.warning("Obstruction detection failed (%s); skipping carve.", exc)
            return floor_mask, [], empty
        boxes = [d.box for d in detections]
        if not boxes:
            return floor_mask, [], empty
        try:
            masks = self.segmenter.segment_many(image, boxes=boxes)
        except Exception as exc:
            logger.warning("Obstruction segmentation failed (%s); skipping carve.", exc)
            return floor_mask, [], empty

        obstruction = np.zeros_like(floor_mask)
        table_boxes_mask = np.zeros_like(floor_mask)
        table_box_list: list[tuple[int, int, int, int]] = []
        floor_area = int((floor_mask > 0).sum())
        for detection, mask in zip(detections, masks):
            label = detection.label.lower()
            if "table" in label:
                x0, y0, x1, y1 = detection.box
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(floor_mask.shape[1], x1), min(floor_mask.shape[0], y1)
                if x1 > x0 and y1 > y0:
                    table_boxes_mask[y0:y1, x0:x1] = 255
                    table_box_list.append((x0, y0, x1, y1))
            if "rug" in label and floor_area > 0:
                rug_area = int((mask > 0).sum())
                if rug_area >= 0.5 * floor_area:
                    logger.info("Skipping floor-scale rug (%.0f%% of floor).", 100.0 * rug_area / floor_area)
                    continue
            obstruction = np.maximum(obstruction, mask)

        obstruction = cv2.bitwise_or(obstruction, table_boxes_mask)
        protected_mask = cv2.bitwise_and(obstruction, floor_mask)
        carve_mask = cv2.erode(obstruction, np.ones((31, 31), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(obstruction)
        for idx in range(1, count):
            if stats[idx, cv2.CC_STAT_AREA] < 400 or stats[idx, cv2.CC_STAT_HEIGHT] < 31:
                carve_mask[labels == idx] = 255
        carve_mask = cv2.bitwise_or(carve_mask, table_boxes_mask)
        carve_mask = cv2.bitwise_and(carve_mask, floor_mask)
        cleaned = cv2.subtract(floor_mask, carve_mask)
        return self._largest_component(cleaned), table_box_list, protected_mask

    @staticmethod
    def _reclaim_under_tables(floor_mask: np.ndarray, depth: np.ndarray, plane: PlaneResult, table_boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
        if not table_boxes or depth is None:
            return floor_mask
        n0, n1, n2, d = plane.equation
        ys, xs = np.mgrid[0 : floor_mask.shape[0], 0 : floor_mask.shape[1]]
        resid = np.abs(n0 * xs + n1 * ys + n2 * depth.astype(np.float32) + d)
        floor_resid = resid[floor_mask > 0]
        if floor_resid.size < 100:
            return floor_mask
        tolerance = float(np.percentile(floor_resid, 95))
        seeds = cv2.dilate(floor_mask, np.ones((3, 3), np.uint8)) > 0
        result = floor_mask.copy()
        for x0, y0, x1, y1 in table_boxes:
            candidates = np.zeros_like(floor_mask, dtype=bool)
            candidates[y0:y1, x0:x1] = (resid[y0:y1, x0:x1] <= tolerance) & (floor_mask[y0:y1, x0:x1] == 0)
            if not candidates.any():
                continue
            count, labels, _, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), connectivity=8)
            for idx in range(1, count):
                if (seeds & (labels == idx)).any():
                    result[labels == idx] = 255
        return result

    @staticmethod
    def _reclaim_object_floor_contacts(floor_mask: np.ndarray, depth: np.ndarray, plane: PlaneResult, protected_mask: np.ndarray) -> np.ndarray:
        """Recover visible floor lost when SAM object masks touch furniture bases."""
        if depth is None or protected_mask is None or not (protected_mask > 0).any():
            return floor_mask
        n0, n1, n2, d = plane.equation
        ys, xs = np.mgrid[0 : floor_mask.shape[0], 0 : floor_mask.shape[1]]
        resid = np.abs(n0 * xs + n1 * ys + n2 * depth.astype(np.float32) + d)
        existing = resid[floor_mask > 0]
        if existing.size < 500:
            return floor_mask
        tolerance = float(np.percentile(existing, 97))
        near_object = cv2.dilate(protected_mask, np.ones((15, 15), np.uint8)) > 0
        near_floor = cv2.dilate(floor_mask, np.ones((9, 9), np.uint8)) > 0
        candidate = (near_object & near_floor & (floor_mask == 0) & (resid <= tolerance)).astype(np.uint8)
        if not candidate.any():
            return floor_mask
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        result = floor_mask.copy()
        floor_seed = cv2.dilate(floor_mask, np.ones((5, 5), np.uint8)) > 0
        accepted = 0
        for idx in range(1, count):
            component = labels == idx
            if stats[idx, cv2.CC_STAT_AREA] < 6:
                continue
            if np.any(floor_seed & component):
                result[component] = 255
                accepted += int(stats[idx, cv2.CC_STAT_AREA])
        if accepted:
            logger.info("[AI] Reclaimed %d floor pixels at object/floor contact edges.", accepted)
        return result

    @staticmethod
    def _fill_floor_notches(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Conservatively smooth small segmentation boundary notches."""
        binary = mask > 0
        if not binary.any():
            return mask
        h, w = binary.shape
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.bitwise_and(smoothed, mask | cv2.dilate(mask, np.ones((5, 5), np.uint8)))
        smoothed[max(8, h - max(8, h // 30)) :] = 255
        return smoothed.astype(np.uint8)

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if count <= 1:
            return mask
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        result = np.zeros(mask.shape, dtype=np.uint8)
        result[labels == largest] = 255
        return result

    def analyze(self, image_path: str | Path, progress_cb=None) -> SceneResult:
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        image = self._downscale(image)
        logger.info("Analysis image: %s (%dx%d)", image_path.name, image.shape[1], image.shape[0])

        def report(fraction: float, message: str) -> None:
            if progress_cb is not None:
                progress_cb(fraction, message)

        report(0.05, "Loading AI models...")
        height, width = image.shape[:2]
        scene = SceneResult()
        scene.image = image
        scene.width = width
        scene.height = height
        scene.metadata["providers"] = self.providers

        report(0.15, "Detecting floor...")
        detection = self.detector.detect_floor(image)
        logger.info("Floor detected with %s: %s", self.detector.name, detection)

        report(0.35, "Segmenting floor...")
        floor_mask = self.segmenter.segment(image, box=detection.box)
        if hasattr(self.segmenter, "segment_many"):
            floor_mask = self._fill_floor_notches(floor_mask, image)
        floor_mask, table_boxes, protected_mask = self._carve_obstructions(image, floor_mask)
        scene.floor_mask = floor_mask
        scene.protected_object_mask = protected_mask
        scene.metadata["occlusion"] = {"provider": self.detector.name, "protected_pixels": int((protected_mask > 0).sum()), "enabled": bool((protected_mask > 0).any())}

        report(0.55, "Building depth map...")
        depth = self.depth.predict(image)
        scene.depth_map = depth

        report(0.7, "Extracting floor geometry...")
        plane = self.plane_estimator.estimate(floor_mask, depth)
        scene.floor_plane.normal = plane.normal
        scene.floor_plane.distance = float(plane.equation[3])
        scene.camera_pose.pitch = plane.pitch
        scene.camera_pose.roll = plane.roll

        if table_boxes:
            floor_mask = self._reclaim_under_tables(floor_mask, depth, plane, table_boxes)
        floor_mask = self._reclaim_object_floor_contacts(floor_mask, depth, plane, protected_mask)
        scene.floor_mask = floor_mask

        report(0.82, "Finalising floor geometry...")
        polygon = self.polygon_engine.extract(floor_mask)
        scene.floor_polygon = polygon
        homography = self.homography_engine.compute(polygon)
        scene.homography = homography.matrix
        scene.metadata["floor_geometry"] = {"polygon": polygon.astype(float).tolist(), "homography_condition": float(np.linalg.cond(homography.matrix))}
        report(0.9, "Scene ready")
        return scene


def build_scene_analyzer(provider: str | None = None) -> SceneAnalyzer:
    """Build a SceneAnalyzer using the requested provider stack."""
    provider = resolve_provider(provider)
    if provider == "auto":
        available, missing = heavy_models_available()
        if available:
            provider = "heavy"
            logger.info("Auto mode: heavy AI stack selected.")
        else:
            provider = "light"
            logger.info("Auto mode: heavy AI stack unavailable (%s); using heuristics.", ", ".join(missing))
    if provider == "v22":
        return _build_v22()
    if provider == "heavy":
        return _build_heavy()
    if provider == "light":
        return _build_light()
    raise RuntimeError(f"Unhandled provider: {provider}")


def _build_heavy() -> SceneAnalyzer:
    from app.ai.depth.depth_anything import DepthAnythingProvider
    from app.ai.detection.grounding_dino import GroundingDINOProvider
    from app.ai.segmentation.sam2 import SAM2Provider
    return SceneAnalyzer(detector=GroundingDINOProvider(), segmenter=SAM2Provider(), depth=DepthAnythingProvider())


def _build_v22() -> SceneAnalyzer:
    """Build the isolated v2.2 geometry stack; never falls back silently."""
    from app.ai.depth.metric3d import Metric3DProvider
    from app.ai.depth.unidepth import UniDepthV2Provider
    from app.ai.detection.grounding_dino import GroundingDINOProvider
    from app.ai.scene.v22_analyzer import V22SceneAnalyzer
    from app.ai.segmentation.sam3 import SAM3Provider

    depth_name = os.getenv("APEX_V22_DEPTH", "metric3d").strip().lower()
    device = os.getenv("APEX_V22_DEVICE", "auto").strip().lower()
    resolved_device = None if device == "auto" else device
    if depth_name == "metric3d":
        depth = Metric3DProvider(device=resolved_device)
    elif depth_name == "unidepth":
        depth = UniDepthV2Provider(device=resolved_device)
    else:
        raise RuntimeError("APEX_V22_DEPTH must be 'metric3d' or 'unidepth'.")

    segmenter = SAM3Provider(prompt="floor", device=resolved_device)
    return V22SceneAnalyzer(detector=GroundingDINOProvider(), segmenter=segmenter, depth=depth)


def _build_light() -> SceneAnalyzer:
    from app.ai.depth.heuristic import HeuristicDepth
    from app.ai.detection.heuristic import HeuristicDetector
    from app.ai.segmentation.heuristic import HeuristicSegmenter
    return SceneAnalyzer(detector=HeuristicDetector(), segmenter=HeuristicSegmenter(), depth=HeuristicDepth())
