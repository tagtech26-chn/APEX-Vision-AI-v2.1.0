"""Tile projection onto the room floor plane."""

from __future__ import annotations

import cv2
import numpy as np

from app.renderer.grout_engine import GroutEngine
from app.renderer.mask_feather import MaskFeather
from app.renderer.occlusion import OcclusionMask
from app.renderer.patterns import TilePatterns
from app.renderer.projection_scale import evaluate_projection_scale


class TileProjector:
    """Wrap a physically scaled tile pattern into the room perspective.

    The scene homography maps the detected floor quad to a 2048x2048 proxy
    plane. The texture canvas therefore uses that same coordinate space;
    using a larger unrelated canvas silently changes apparent tile scale.
    """

    PLANE_SPAN_PIXELS = 2048
    REFERENCE_FLOOR_MM = 7800

    def __init__(self, debug: bool = False) -> None:
        self.patterns = TilePatterns()
        self.grout = GroutEngine()
        self.feather = MaskFeather()
        self.debug = debug
        self.last_scale_diagnostics: dict[str, float | int] | None = None
        self.last_occlusion_diagnostics: dict[str, float | int | bool] | None = None
        self.last_grout_mask: np.ndarray | None = None

    @staticmethod
    def _to_square(tile: np.ndarray) -> np.ndarray:
        """Center-crop a texture swatch to a square without stretching it."""
        h, w = tile.shape[:2]
        if h == w:
            return tile
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        return tile[y0 : y0 + side, x0 : x0 + side]

    @staticmethod
    def _preserve_texture_detail(tile: np.ndarray) -> np.ndarray:
        """Recover fine ceramic/stone detail lost in catalogue thumbnails."""
        if min(tile.shape[:2]) < 32:
            return tile
        blur = cv2.GaussianBlur(tile, (0, 0), 1.0)
        return cv2.addWeighted(tile, 1.12, blur, -0.12, 0)

    def _tile_pixels(self, tile_size_mm: int) -> int:
        """Return tile size in the same 2048px coordinate system as homography."""
        size = max(100, int(tile_size_mm))
        tiles_across = int(round(self.REFERENCE_FLOOR_MM / size))
        tiles_across = max(3, min(tiles_across, 40))
        return max(48, self.PLANE_SPAN_PIXELS // tiles_across)

    @staticmethod
    def _plane_shift(homography: np.ndarray, floor_mask: np.ndarray, canvas_size: int) -> tuple[float, float]:
        """Shift the pattern to the centre of the projected floor bounds."""
        rows, cols = np.where(floor_mask > 0)
        if rows.size == 0:
            return 0.0, 0.0
        stride = max(1, int(np.sqrt(rows.size / 10000)))
        rows = rows[::stride]
        cols = cols[::stride]
        points = np.stack([cols, rows, np.ones_like(cols, dtype=np.float64)], axis=-1)
        plane = points @ homography.T
        denominator = plane[:, 2]
        valid = np.abs(denominator) > 1e-10
        if not np.any(valid):
            return 0.0, 0.0
        px = plane[valid, 0] / denominator[valid]
        py = plane[valid, 1] / denominator[valid]
        cx = 0.5 * (float(px.min()) + float(px.max()))
        cy = 0.5 * (float(py.min()) + float(py.max()))
        sx = canvas_size / 2.0 - cx
        sy = canvas_size / 2.0 - cy
        span_x = float(px.max()) - float(px.min())
        span_y = float(py.max()) - float(py.min())
        if span_x > canvas_size or span_y > canvas_size:
            return 0.0, 0.0
        return sx, sy

    @staticmethod
    def _grout_tile_mask(tile_pixels: int, tile_size_mm: int, grout_width_mm: int) -> np.ndarray:
        """Create a binary grout mask aligned with physical tile dimensions."""
        mask = np.zeros((tile_pixels, tile_pixels, 3), dtype=np.uint8)
        thickness = GroutEngine._thickness_pixels(tile_pixels, tile_size_mm, grout_width_mm)
        if thickness <= 0:
            return mask
        thickness = min(thickness, max(1, tile_pixels // 8))
        cv2.rectangle(mask, (0, 0), (tile_pixels - 1, thickness - 1), (255, 255, 255), -1)
        cv2.rectangle(mask, (0, tile_pixels - thickness), (tile_pixels - 1, tile_pixels - 1), (255, 255, 255), -1)
        cv2.rectangle(mask, (0, 0), (thickness - 1, tile_pixels - 1), (255, 255, 255), -1)
        cv2.rectangle(mask, (tile_pixels - thickness, 0), (tile_pixels - 1, tile_pixels - 1), (255, 255, 255), -1)
        return mask

    def project(self, tile_image: np.ndarray, homography: np.ndarray, output_size: tuple[int, int], tile_size_mm: int = 600, pattern: str = "Straight", floor_mask: np.ndarray | None = None, grout_width: int = 2, grout_color=(220, 220, 220)) -> np.ndarray:
        if homography is None:
            raise RuntimeError("Homography missing.")
        width, height = output_size
        tile_pixels = self._tile_pixels(tile_size_mm)
        self.last_scale_diagnostics = evaluate_projection_scale(tile_size_mm=tile_size_mm, tile_pixels=tile_pixels, plane_span_pixels=self.PLANE_SPAN_PIXELS, reference_floor_mm=self.REFERENCE_FLOOR_MM).as_dict()

        tile = self._to_square(tile_image)
        tile = self._preserve_texture_detail(tile)
        tile = cv2.resize(tile, (tile_pixels, tile_pixels), interpolation=cv2.INTER_LANCZOS4)
        tile = self.grout.apply(tile=tile, grout_width_mm=grout_width, grout_color=grout_color, tile_size_mm=tile_size_mm)
        canvas = self.patterns.create(tile, pattern)
        grout_tile = self._grout_tile_mask(tile_pixels, tile_size_mm, grout_width)
        grout_canvas = self.patterns.create(grout_tile, pattern)

        try:
            transform = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            transform = homography

        if floor_mask is not None:
            sx, sy = self._plane_shift(homography, floor_mask, canvas.shape[0])
            if sx or sy:
                shift = np.array([[1, 0, -sx], [0, 1, -sy], [0, 0, 1]], dtype=np.float64)
                transform = transform @ shift

        warped = cv2.warpPerspective(canvas, transform, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        warped_grout = cv2.warpPerspective(grout_canvas, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        self.last_grout_mask = (warped_grout[:, :, 0] > 80).astype(np.uint8)
        if self.debug:
            cv2.imwrite("output/debug_tile.png", tile)
            cv2.imwrite("output/debug_canvas.png", canvas)
            cv2.imwrite("output/debug_grout_mask.png", self.last_grout_mask * 255)
            cv2.imwrite("output/debug_projection.png", warped)
        return warped

    def blend(self, room: np.ndarray, projection: np.ndarray, floor_mask: np.ndarray, alpha: float = 0.92, occlusion_mask: np.ndarray | None = None) -> np.ndarray:
        """Blend the projection over the floor while preserving foreground objects."""
        mask = self.feather.feather(floor_mask, radius=17)
        self.last_occlusion_diagnostics = OcclusionMask.leakage_diagnostics(mask, occlusion_mask)
        mask = OcclusionMask.apply(mask, occlusion_mask)[..., None]
        room = room.astype(np.float32)
        projection = projection.astype(np.float32)
        projection = projection * alpha + room * (1.0 - alpha)
        result = room * (1.0 - mask) + projection * mask
        result = np.clip(result, 0, 255)
        if self.debug:
            cv2.imwrite("output/debug_final.png", result.astype(np.uint8))
        return result.astype(np.uint8)
