"""Lightweight heuristic floor/object segmentation (no external models)."""

from __future__ import annotations

import cv2
import numpy as np

from app.ai.segmentation.base import SamplerSegmenterMixin, Segmenter


def _estimate_floor_boundary(image: np.ndarray) -> int:
    """Estimate the wall/floor transition using a smoothed horizontal edge profile."""
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


def estimate_floor_mask(image: np.ndarray) -> np.ndarray:
    """Estimate only the visible floor, with an explicit geometric prior."""
    h, w = image.shape[:2]
    if h < 20 or w < 20:
        return np.zeros((h, w), dtype=np.uint8)
    boundary = _estimate_floor_boundary(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    y0 = max(boundary, int(h * 0.50))
    x0, x1 = int(w * 0.20), int(w * 0.80)
    seed = lab[int(h * 0.90) :, x0:x1]
    if seed.size == 0:
        return np.zeros((h, w), dtype=np.uint8)
    seed_median = np.median(seed.reshape(-1, 3), axis=0)
    dist = np.linalg.norm(lab - seed_median, axis=2)
    seed_dist = dist[int(h * 0.90) :, x0:x1]
    threshold = max(10.0, float(np.percentile(seed_dist, 90) * 2.0))
    candidate = (dist <= threshold).astype(np.uint8) * 255
    candidate[:y0, :] = 0

    # Retain only regions connected to the bottom edge. This prevents walls,
    # windows and disconnected furniture patches from becoming floor islands.
    bottom = np.zeros_like(candidate)
    bottom[h - 2 : h, :] = candidate[h - 2 : h, :]
    reachable = cv2.dilate(bottom, np.ones((9, 9), np.uint8), iterations=1)
    for _ in range(20):
        expanded = cv2.dilate(reachable, np.ones((7, 7), np.uint8), iterations=1)
        expanded[candidate == 0] = 0
        if np.array_equal(expanded, reachable):
            break
        reachable = expanded
    mask = reachable
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask[:y0, :] = 0
    mask = _carve_high_texture(mask, image)
    mask[:y0, :] = 0
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


def estimate_obstruction_mask(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Segment a detected foreground box without a learned model."""
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
    bh, bw = crop.shape[:2]
    border = np.concatenate([
        lab[: max(2, bh // 10)].reshape(-1, 3),
        lab[-max(2, bh // 10) :].reshape(-1, 3),
        lab[:, : max(2, bw // 10)].reshape(-1, 3),
        lab[:, -max(2, bw // 10) :].reshape(-1, 3),
    ], axis=0)
    reference = np.median(border, axis=0)
    colour_distance = np.linalg.norm(lab - reference, axis=2)
    mean = cv2.boxFilter(gray, -1, (11, 11))
    mean2 = cv2.boxFilter(gray * gray, -1, (11, 11))
    local_std = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))
    colour_threshold = max(12.0, float(np.percentile(colour_distance, 65)))
    texture_threshold = max(10.0, float(np.percentile(local_std, 72)))
    candidate = ((colour_distance > colour_threshold) | (local_std > texture_threshold)).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
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

    def segment(self, image: np.ndarray, box: tuple[int, int, int, int] | None = None, points: np.ndarray | None = None) -> np.ndarray:
        return estimate_floor_mask(image)

    def segment_many(self, image: np.ndarray, boxes: list[tuple[int, int, int, int]] | None = None, points: np.ndarray | None = None) -> list[np.ndarray]:
        return [estimate_obstruction_mask(image, box) for box in (boxes or [])]
