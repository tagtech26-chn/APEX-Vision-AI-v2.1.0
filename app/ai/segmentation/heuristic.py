"""Lightweight heuristic floor/object segmentation (no external models)."""

from __future__ import annotations

import cv2
import numpy as np

from app.ai.segmentation.base import SamplerSegmenterMixin, Segmenter


def estimate_floor_mask(image: np.ndarray) -> np.ndarray:
    """Automatically estimate the visible floor mask.

    Uses adaptive LAB colour distance from a bottom-centre seed band, then
    carves out high-texture regions (rugs, patterned furniture) that the
    colour threshold cannot separate from the floor.
    """
    h, w = image.shape[:2]

    y0 = int(h * 0.94)
    x0, x1 = int(w * 0.20), int(w * 0.80)
    if y0 >= h or x0 >= x1:
        return np.zeros((h, w), dtype=np.uint8)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    seed = lab[y0:, x0:x1]
    seed_median = np.median(seed.reshape(-1, 3), axis=0)

    dist = np.abs(lab - seed_median).sum(axis=2)
    region = dist[y0:, x0:x1]
    threshold = float(np.maximum(region.mean() + 2.5 * region.std(), 12.0))

    mask = (dist < threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = _largest_component(mask)

    mask = _carve_high_texture(mask, image)

    fine = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, fine, iterations=1)

    return mask.astype(np.uint8)


def _carve_high_texture(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Remove rugs/furniture inside the floor that are textured differently."""
    if (mask > 0).sum() == 0:
        return mask

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.boxFilter(gray * gray, -1, (15, 15))
    mean = cv2.boxFilter(gray, -1, (15, 15))
    local_std = np.sqrt(np.maximum(gray2 - mean * mean, 0.0))

    floor_median = float(np.median(local_std[mask > 0]))
    carve_threshold = max(14.0, floor_median * 5.0)
    interior = cv2.erode(mask, np.ones((15, 15), np.uint8))

    carved = mask.copy()
    carved[(local_std > carve_threshold) & (interior > 0)] = 0
    return _largest_component(carved)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return mask

    largest = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]
    for i in range(2, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest = i
            largest_area = area

    result = np.zeros(mask.shape, dtype=np.uint8)
    result[labels == largest] = 255
    return result


def estimate_obstruction_mask(
    image: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    """Segment a detected foreground box without a learned model.

    The mask starts from colour/texture separation inside the detector box and
    is then dilated slightly so antialiased edges cannot leak tile pixels.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    result = np.zeros((h, w), dtype=np.uint8)
    if x2 <= x1 or y2 <= y1:
        return result

    crop = image[y1:y2, x1:x2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Border pixels are more likely to be floor/background. Use their robust
    # statistics as the local background reference.
    bh, bw = crop.shape[:2]
    border = np.concatenate(
        [
            lab[: max(2, bh // 10), :, :].reshape(-1, 3),
            lab[-max(2, bh // 10) :, :, :].reshape(-1, 3),
            lab[:, : max(2, bw // 10), :].reshape(-1, 3),
            lab[:, -max(2, bw // 10) :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    reference = np.median(border, axis=0)
    colour_distance = np.linalg.norm(lab - reference, axis=2)

    mean = cv2.boxFilter(gray, -1, (11, 11))
    mean2 = cv2.boxFilter(gray * gray, -1, (11, 11))
    local_std = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))

    colour_threshold = max(12.0, float(np.percentile(colour_distance, 65)))
    texture_threshold = max(10.0, float(np.percentile(local_std, 72)))
    candidate = (colour_distance > colour_threshold) | (
        local_std > texture_threshold
    )

    candidate = (candidate.astype(np.uint8) * 255)
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2
    )
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1
    )

    # Keep the dominant connected foreground region, but fall back to the box
    # when the image is nearly uniform and there is no reliable separation.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = 1 + int(np.argmax(areas))
        if int(areas.max()) >= max(100, int(crop.shape[0] * crop.shape[1] * 0.04)):
            candidate = np.where(labels == idx, 255, 0).astype(np.uint8)
        else:
            candidate = np.full(candidate.shape, 255, dtype=np.uint8)
    else:
        candidate = np.full(candidate.shape, 255, dtype=np.uint8)

    candidate = cv2.dilate(candidate, np.ones((5, 5), np.uint8), iterations=1)
    result[y1:y2, x1:x2] = candidate
    return result


class HeuristicSegmenter(SamplerSegmenterMixin, Segmenter):
    """Produces floor and conservative foreground masks via image statistics."""

    name = "heuristic"

    def segment(
        self,
        image: np.ndarray,
        box: tuple[int, int, int, int] | None = None,
        points: np.ndarray | None = None,
    ) -> np.ndarray:
        return estimate_floor_mask(image)

    def segment_many(
        self,
        image: np.ndarray,
        boxes: list[tuple[int, int, int, int]] | None = None,
        points: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        return [
            estimate_obstruction_mask(image, box)
            for box in (boxes or [])
        ]
