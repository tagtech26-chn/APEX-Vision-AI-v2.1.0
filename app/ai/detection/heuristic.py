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
    def _detect_obstructions(image: np.ndarray) -> list[Detection]:
        """Find sizeable foreground regions in the geometric floor envelope.

        Do not restrict the candidate to ``estimate_floor_mask`` itself.  The
        floor segmenter intentionally carves out high-contrast furniture, so
        doing so here would remove the very pixels we need to detect.  Instead,
        use the top of the estimated floor as a geometric envelope and compare
        the image against a clean bottom-floor colour reference.
        """
        floor = estimate_floor_mask(image)
        floor_bool = floor > 0
        if not floor_bool.any():
            return []

        h, w = image.shape[:2]
        floor_rows = np.where(floor_bool.any(axis=1))[0]
        floor_top = int(floor_rows.min())
        if floor_top >= h - 4:
            return []

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # The bottom-centre region is deliberately used as the floor reference:
        # it is least likely to contain walls, windows or furniture.
        y0 = max(floor_top, int(h * 0.88))
        x0, x1 = int(w * 0.25), int(w * 0.75)
        seed = lab[y0:, x0:x1].reshape(-1, 3)
        if seed.size == 0:
            return []

        median = np.median(seed, axis=0)
        mad = np.median(np.abs(seed - median), axis=0)
        scale = float(max(4.0, 1.4826 * np.mean(mad)))
        colour_distance = np.linalg.norm(lab - median, axis=2)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean = cv2.boxFilter(gray, -1, (15, 15))
        mean2 = cv2.boxFilter(gray * gray, -1, (15, 15))
        local_std = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))

        # Keep the adaptive threshold conservative.  A high percentile derived
        # from the already-carved floor mask misses uniform sofas/cabinets.
        colour_threshold = max(18.0, 3.0 * scale)
        texture_threshold = max(22.0, float(np.percentile(local_std[y0:, :], 90)) * 1.6)

        candidate = np.zeros((h, w), dtype=np.uint8)
        candidate[floor_top:, :] = (
            (colour_distance[floor_top:, :] > colour_threshold)
            | (local_std[floor_top:, :] > texture_threshold)
        ).astype(np.uint8) * 255

        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_CLOSE,
            np.ones((17, 17), np.uint8),
            iterations=2,
        )
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_OPEN,
            np.ones((7, 7), np.uint8),
            iterations=1,
        )

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate, connectivity=8
        )
        min_area = max(400, int(h * w * 0.002))
        max_area = int(h * w * 0.35)
        detections: list[Detection] = []

        for idx in range(1, count):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            bw = int(stats[idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
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
            detections.append(
                Detection(label="furniture", score=score, box=(x, y, x2, y2))
            )

        detections.sort(key=lambda item: item.score, reverse=True)
        return detections[:8]
