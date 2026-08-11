"""Lightweight heuristic object detector (no external models)."""

from __future__ import annotations

import cv2
import numpy as np

from app.ai.detection.base import Detection, ObjectDetector
from app.ai.segmentation.heuristic import estimate_floor_mask


_FURNITURE_PROMPT = (
    "sofa. couch. armchair. chair. table. rug. plant. lamp. "
    "cabinet. bed. furniture."
)


class HeuristicDetector(ObjectDetector):
    """Detect floor and conservative foreground obstructions without ML models."""

    name = "heuristic"

    def detect(self, image: np.ndarray, prompt: str) -> list[Detection]:
        if prompt == "floor":
            mask = estimate_floor_mask(image)
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                return []

            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            score = float(mask[y1:y2, x1:x2].mean() / 255.0)
            return [Detection(label="floor", score=score, box=(x1, y1, x2, y2))]

        if prompt.strip().lower() == _FURNITURE_PROMPT:
            return self._detect_obstructions(image)

        return []

    @staticmethod
    def _geometric_floor_top(image: np.ndarray) -> int:
        """Estimate the wall/floor transition without depending on floor pixels."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = cv2.GaussianBlur(gray, (0, 0), max(1.0, w / 160.0))
        x0, x1 = int(w * 0.15), int(w * 0.85)
        y0, y1 = int(h * 0.42), int(h * 0.78)
        if y1 <= y0 + 4:
            return int(h * 0.62)
        profile = np.mean(np.abs(np.diff(gray[y0:y1, x0:x1], axis=0)), axis=1)
        profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 9), 0).ravel()
        peak = int(np.argmax(profile)) + y0
        return int(np.clip(peak + max(4, int(h * 0.012)), h * 0.50, h * 0.76))

    @classmethod
    def _detect_obstructions(cls, image: np.ndarray) -> list[Detection]:
        """Find sizeable foreground regions in the geometric floor envelope.

        The floor segmenter intentionally removes high-contrast furniture from
        the floor mask. Object detection therefore uses the image itself and
        only uses the floor mask/geometric transition as a spatial prior.
        """
        h, w = image.shape[:2]
        if h < 20 or w < 20:
            return []

        floor = estimate_floor_mask(image)
        floor_bool = floor > 0
        mask_rows = np.where(floor_bool.any(axis=1))[0]
        mask_top = int(mask_rows.min()) if len(mask_rows) else h

        # Do not rely exclusively on the segmented floor's first occupied row:
        # a large obstruction can cause the bottom-connected floor component to
        # start too low. The geometric transition remains available even when
        # the floor mask has already carved the obstruction away.
        geometric_top = cls._geometric_floor_top(image)
        floor_top = min(mask_top, geometric_top)
        if floor_top >= h - 4:
            return []

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Bottom-centre pixels are the cleanest floor reference in the fixture
        # and in typical room scenes. Exclude the outer edges to avoid walls.
        seed_y0 = max(int(h * 0.86), floor_top + int(h * 0.20))
        seed_y0 = min(seed_y0, h - 2)
        seed_x0, seed_x1 = int(w * 0.25), int(w * 0.75)
        seed = lab[seed_y0:, seed_x0:seed_x1].reshape(-1, 3)
        if seed.size == 0:
            return []

        median = np.median(seed, axis=0)
        mad = np.median(np.abs(seed - median), axis=0)
        scale = float(max(4.0, 1.4826 * np.mean(mad)))
        colour_distance = np.linalg.norm(lab - median, axis=2)

        # A uniform sofa/cabinet is intentionally detectable by colour alone;
        # texture is only an additional cue for patterned objects.
        colour_threshold = max(18.0, 3.0 * scale)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean = cv2.boxFilter(gray, -1, (15, 15))
        mean2 = cv2.boxFilter(gray * gray, -1, (15, 15))
        local_std = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))
        texture_threshold = max(22.0, float(np.percentile(local_std[seed_y0:, :], 90)) * 1.6)

        candidate = np.zeros((h, w), dtype=np.uint8)
        candidate[floor_top:, :] = (
            (colour_distance[floor_top:, :] > colour_threshold)
            | (local_std[floor_top:, :] > texture_threshold)
        ).astype(np.uint8) * 255

        # Suppress very narrow edge fragments before connected-component
        # extraction, while retaining large solid furniture regions.
        candidate = cv2.morphologyEx(
            candidate, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2
        )
        candidate = cv2.morphologyEx(
            candidate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1
        )

        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        min_area = max(400, int(h * w * 0.002))
        max_area = int(h * w * 0.35)
        detections: list[Detection] = []

        for idx in range(1, count):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            bw = int(stats[idx, cv2.CC_STAT_WIDTH))
            bh = int(stats[idx, cv2.CC_STAT_HEIGHT))
            if area < min_area or area > max_area:
                continue
            if bw < max(20, int(w * 0.04)) or bh < max(20, int(h * 0.04)):
                continue

            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            fill = area / float(max(1, bw * bh))
            score = float(np.clip(0.55 + 0.45 * fill, 0.55, 0.99))
            detections.append(Detection(label="furniture", score=score, box=(x, y, x2, y2)))

        detections.sort(key=lambda item: item.score, reverse=True)
        return detections[:8]
