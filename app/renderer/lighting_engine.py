"""Lighting engine: extract and apply room illumination."""

from __future__ import annotations

import cv2
import numpy as np


class LightingEngine:
    """Preserve low-frequency floor illumination without leaking object shading."""

    def extract(
        self,
        room: np.ndarray,
        floor_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build a smooth illumination field from the visible floor only.

        Blurring the entire room lets bright windows and dark furniture bleed
        into the floor illumination map.  That makes projected tiles acquire
        false shadows and uneven exposure.  Non-floor pixels are replaced with
        the floor median before the low-frequency blur so the field remains a
        property of the floor plane.
        """
        lab = cv2.cvtColor(room, cv2.COLOR_BGR2LAB).astype(np.float32)
        lightness = lab[:, :, 0]

        if floor_mask is None or not np.any(floor_mask > 0):
            source = lightness
        else:
            mask = floor_mask > 0
            reference = float(np.median(lightness[mask]))
            source = np.full_like(lightness, reference, dtype=np.float32)
            source[mask] = lightness[mask]

        sigma = max(room.shape[:2]) / 45.0
        return cv2.GaussianBlur(source, (0, 0), sigma)

    def apply(
        self,
        projection: np.ndarray,
        lighting: np.ndarray,
        floor_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if floor_mask is not None and np.any(floor_mask > 0):
            ref = lighting[floor_mask > 0]
        else:
            ref = lighting

        mean = float(np.median(ref)) + 1e-6
        factor = lighting / mean
        # Keep the original material readable.  Lighting is a gentle shading
        # term, not a second colour/contrast transform.
        factor = np.clip(factor, 0.86, 1.16)

        result = projection.astype(np.float32)
        result *= factor[..., None]
        return np.clip(result, 0, 255).astype(np.uint8)
