"""Tile pattern canvas generators."""

from __future__ import annotations

import cv2
import numpy as np


class TilePatterns:
    """Build tile patterns in the same coordinate plane as the homography."""

    def __init__(self, default_canvas: int = 2048) -> None:
        self.default_canvas = int(max(512, default_canvas))

    def create(self, tile: np.ndarray, pattern: str) -> np.ndarray:
        pattern = (pattern or "Straight").strip().lower()

        if pattern in {"brick", "brick pattern"}:
            return self.create_brick_canvas(tile)
        if pattern in {"herringbone", "fishbone"}:
            return self.create_herringbone_canvas(tile)
        if pattern in {"chevron"}:
            return self.create_chevron_canvas(tile)
        return self.create_straight_canvas(tile)

    def _canvas_size(self, tile: np.ndarray) -> int:
        tile_h, tile_w = tile.shape[:2]
        # The V2.2 homography maps the floor to 2048x2048. Do not generate a
        # larger unrelated texture plane because inverse warping would change
        # scale and origin while adding CPU/memory cost.
        size = max(self.default_canvas, max(tile_h, tile_w) * 12)
        return int(min(size, self.default_canvas))

    def create_straight_canvas(self, tile: np.ndarray) -> np.ndarray:
        tile_h, tile_w = tile.shape[:2]
        canvas_size = self._canvas_size(tile)
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

        for y in range(0, canvas_size, tile_h):
            for x in range(0, canvas_size, tile_w):
                y2 = min(y + tile_h, canvas_size)
                x2 = min(x + tile_w, canvas_size)
                canvas[y:y2, x:x2] = tile[: y2 - y, : x2 - x]
        return canvas

    def create_brick_canvas(self, tile: np.ndarray) -> np.ndarray:
        tile_h, tile_w = tile.shape[:2]
        canvas_size = self._canvas_size(tile)
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

        row = 0
        for y in range(0, canvas_size, tile_h):
            offset = tile_w // 2 if row % 2 else 0
            for x in range(-offset, canvas_size, tile_w):
                x1 = max(x, 0)
                y1 = max(y, 0)
                x2 = min(x + tile_w, canvas_size)
                y2 = min(y + tile_h, canvas_size)
                src_x = max(0, -x)
                canvas[y1:y2, x1:x2] = tile[: y2 - y1, src_x : src_x + (x2 - x1)]
            row += 1
        return canvas

    @staticmethod
    def _plank(tile: np.ndarray, ratio: float = 2.0) -> np.ndarray:
        """Return a rectangular plank derived from the tile texture."""
        h, w = tile.shape[:2]
        length = int(round(w * ratio))
        return cv2.resize(tile, (length, h), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _rotated(img: np.ndarray, angle: float) -> np.ndarray:
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        nw = int(h * sin + w * cos)
        nh = int(h * cos + w * sin)
        matrix[0, 2] += nw / 2 - w / 2
        matrix[1, 2] += nh / 2 - h / 2
        return cv2.warpAffine(
            img,
            matrix,
            (nw, nh),
            flags=cv2.INTER_CUBIC,
            borderValue=(0, 0, 0),
        )

    @staticmethod
    def _rotated_with_mask(img: np.ndarray, angle: float):
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        nw = int(h * sin + w * cos)
        nh = int(h * cos + w * sin)
        matrix[0, 2] += nw / 2 - w / 2
        matrix[1, 2] += nh / 2 - h / 2
        rotated = cv2.warpAffine(
            img,
            matrix,
            (nw, nh),
            flags=cv2.INTER_CUBIC,
            borderValue=(0, 0, 0),
        )
        ones = np.full((h, w), 255, dtype=np.uint8)
        mask = cv2.warpAffine(
            ones,
            matrix,
            (nw, nh),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        return rotated, mask

    @staticmethod
    def _paste(
        canvas: np.ndarray,
        img: np.ndarray,
        cx: int,
        cy: int,
        mask: np.ndarray | None = None,
    ) -> None:
        h, w = img.shape[:2]
        x1 = cx - w // 2
        y1 = cy - h // 2
        x2 = x1 + w
        y2 = y1 + h
        canvas_h, canvas_w = canvas.shape[:2]

        dx1, dy1 = max(-x1, 0), max(-y1, 0)
        dx2, dy2 = max(x2 - canvas_w, 0), max(y2 - canvas_h, 0)
        if dx1 >= w or dy1 >= h or dx2 >= w or dy2 >= h:
            return

        sx1, sy1 = dx1, dy1
        sx2, sy2 = w - dx2, h - dy2
        cx1, cy1 = max(x1, 0), max(y1, 0)
        cx2, cy2 = min(x2, canvas_w), min(y2, canvas_h)
        if cx1 >= cx2 or cy1 >= cy2 or sx1 >= sx2 or sy1 >= sy2:
            return

        roi = canvas[cy1:cy2, cx1:cx2]
        src = img[sy1:sy2, sx1:sx2]
        if mask is None:
            roi[...] = src
        else:
            cv2.copyTo(src, mask[sy1:sy2, sx1:sx2], roi)

    def _zigzag_canvas(self, tile: np.ndarray, checker: bool) -> np.ndarray:
        """Draw planks at +/-45 degrees without black rotation corners."""
        plank = self._plank(tile, ratio=2.0)
        rotated_neg, mask_neg = self._rotated_with_mask(plank, -45)
        rotated_pos, mask_pos = self._rotated_with_mask(plank, 45)

        cell = max(rotated_neg.shape[0], rotated_neg.shape[1])
        canvas_size = self._canvas_size(tile)
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

        # Overlap the rotated bounding boxes so their transparent corners do
        # not become uncovered black regions after perspective projection.
        step = max(1, cell // 3)
        cols = int(np.ceil(canvas_size / step)) + 2
        rows = int(np.ceil(canvas_size / step)) + 2

        for r in range(rows):
            for c in range(cols):
                use_neg = (r + c) % 2 == 0 if checker else (r % 2 == 0)
                img = rotated_neg if use_neg else rotated_pos
                mask = mask_neg if use_neg else mask_pos

                cx = c * step
                cy = r * step
                if not checker:
                    cy += step // 2 if c % 2 else 0
                else:
                    cx += step // 2 if r % 2 else 0

                self._paste(canvas, img, cx, cy, mask)

        return canvas

    def create_herringbone_canvas(self, tile: np.ndarray) -> np.ndarray:
        return self._zigzag_canvas(tile, checker=True)

    def create_chevron_canvas(self, tile: np.ndarray) -> np.ndarray:
        return self._zigzag_canvas(tile, checker=False)
