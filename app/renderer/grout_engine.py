"""Grout painting engine."""

from __future__ import annotations

import cv2
import numpy as np


class GroutEngine:
    """Paint a physically scaled grout joint around each projected tile."""

    @staticmethod
    def _thickness_pixels(tile_pixels: int, tile_size_mm: int, grout_width_mm: int) -> int:
        """Convert the requested physical grout width to projected pixels.

        The previous implementation treated millimetres as literal pixels,
        which made a 2 mm joint appear roughly 4x too wide on a 600 mm tile.
        Keeping the conversion tied to the actual projected tile size makes
        grout visually consistent across 300/600/1200 mm products.
        """
        tile_px = max(1, int(tile_pixels))
        tile_mm = max(1, int(tile_size_mm))
        grout_mm = max(0, int(grout_width_mm))
        if grout_mm == 0:
            return 0
        # A one-pixel minimum keeps small joints visible after rasterisation;
        # the physical ratio remains the controlling value.
        return max(1, int(round(tile_px * grout_mm / tile_mm)))

    def apply(
        self,
        tile: np.ndarray,
        grout_width_mm: int = 2,
        grout_color=(220, 220, 220),
        tile_size_mm: int = 600,
    ) -> np.ndarray:
        result = tile.copy()
        h, w = result.shape[:2]
        thickness = min(
            max(0, self._thickness_pixels(min(h, w), tile_size_mm, grout_width_mm)),
            max(0, min(h, w) // 4),
        )
        if thickness <= 0:
            return result

        color = tuple(int(c) for c in grout_color)
        # Draw half-open borders so adjacent tiles form one continuous joint
        # rather than accumulating an extra pixel at every seam.
        cv2.rectangle(result, (0, 0), (w - 1, thickness - 1), color, -1)
        cv2.rectangle(result, (0, h - thickness), (w - 1, h - 1), color, -1)
        cv2.rectangle(result, (0, 0), (thickness - 1, h - 1), color, -1)
        cv2.rectangle(result, (w - thickness, 0), (w - 1, h - 1), color, -1)
        return result
