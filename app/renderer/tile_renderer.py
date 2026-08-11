"""Main tile renderer orchestrator."""

from __future__ import annotations

import cv2
import numpy as np

from app.ai.scene.result import SceneResult
from app.renderer.color_matcher import ColorMatcher
from app.renderer.grout_engine import GroutEngine
from app.renderer.lighting_engine import LightingEngine
from app.renderer.material_engine import MaterialEngine
from app.renderer.tile_projector import TileProjector


class TileRenderer:
    """Composite a material into a scene using a surface-specific profile."""

    def __init__(self, debug: bool = False) -> None:
        self.projector = TileProjector(debug=debug)
        self.grout = GroutEngine()
        self.matcher = ColorMatcher()
        self.lighting = LightingEngine()
        self.material = MaterialEngine()

    def render(
        self,
        scene: SceneResult,
        tile: np.ndarray,
        alpha: float = 0.92,
        grout_width: int = 2,
        grout_color=(220, 220, 220),
        tile_size_mm: int = 600,
        pattern: str = "Straight",
        material_profile: str = "generic",
        material_intelligence: dict[str, object] | None = None,
        smart_removal: bool = True,
        furniture_shadow: bool = True,
        enhance_lighting: bool = True,
        visualization_mode: str = "Realistic",
    ) -> np.ndarray:
        if scene.image is None:
            raise RuntimeError("Scene image missing.")
        if scene.floor_mask is None:
            raise RuntimeError("Floor mask missing.")
        if scene.homography is None:
            raise RuntimeError("Homography missing.")

        render_mask = self._floor_render_mask(scene)
        protected = scene.protected_object_mask if smart_removal else None

        projection = self.projector.project(
            tile_image=tile,
            homography=scene.homography,
            output_size=scene.size,
            tile_size_mm=tile_size_mm,
            pattern=pattern,
            floor_mask=render_mask,
            grout_width=grout_width,
            grout_color=grout_color,
        )
        if self.projector.last_scale_diagnostics is not None:
            scene.metadata["projection_scale"] = dict(self.projector.last_scale_diagnostics)

        projection = self.material.enhance(
            projection,
            profile=material_profile,
            finish=str((material_intelligence or {}).get("finish", "satin")),
            texture_scale_factor=float((material_intelligence or {}).get("texture_scale_factor", 1.0)),
        )

        if enhance_lighting and visualization_mode == "Realistic":
            lighting = self.lighting.extract(scene.image, render_mask)
            projection = self.lighting.apply(projection, lighting, render_mask)

        projection = self.matcher.match(
            room=scene.image,
            projection=projection,
            floor_mask=render_mask,
        )

        # Keep grout joints visually legible after material/lighting/color
        # processing. The seam mask is generated from the same physical tile
        # geometry used for projection, so it remains aligned in perspective.
        grout_mask = self.projector.last_grout_mask
        if grout_mask is not None and grout_width > 0:
            gm = np.clip(grout_mask.astype(np.float32), 0.0, 1.0)[..., None]
            grout_bgr = np.asarray(tuple(int(c) for c in grout_color), dtype=np.float32)
            projection = projection.astype(np.float32) * (1.0 - gm * 0.42) + grout_bgr * (gm * 0.42)
            projection = np.clip(projection, 0, 255).astype(np.uint8)

        if protected is not None:
            scene.metadata.setdefault("occlusion", {})["applied"] = True
            scene.metadata["occlusion"]["protected_pixels"] = int((protected > 0).sum())
        else:
            scene.metadata.setdefault("occlusion", {})["applied"] = False

        result = self.projector.blend(
            room=scene.image,
            projection=projection,
            floor_mask=render_mask,
            alpha=alpha if visualization_mode == "Realistic" else min(alpha, 0.98),
            occlusion_mask=protected,
        )
        if self.projector.last_occlusion_diagnostics is not None:
            scene.metadata.setdefault("occlusion", {}).update(self.projector.last_occlusion_diagnostics)

        if visualization_mode == "Realistic":
            original = scene.image.astype(np.float32)
            rendered = result.astype(np.float32)
            gray = cv2.cvtColor(scene.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            local = gray - cv2.GaussianBlur(gray, (0, 0), 9)
            rendered += local[..., None] * 0.035
            result = np.clip(rendered, 0, 255).astype(np.uint8)
            _ = original

        return result

    @staticmethod
    def _floor_render_mask(scene: SceneResult) -> np.ndarray:
        """Recover small AI-carving gaps at object/floor contact edges."""
        if scene.floor_mask is None:
            return np.zeros(scene.size[::-1], dtype=np.uint8)

        mask = scene.floor_mask.copy()
        protected = scene.protected_object_mask
        depth = scene.depth_map
        if protected is None or depth is None or not (protected > 0).any():
            return mask

        try:
            normal = np.asarray(scene.floor_plane.normal, dtype=np.float32).reshape(-1)
            distance = float(scene.floor_plane.distance)
            if normal.size < 3 or abs(float(normal[2])) < 1e-6:
                return mask

            h, w = mask.shape
            ys, xs = np.mgrid[0:h, 0:w]
            residual = np.abs(
                float(normal[0]) * xs.astype(np.float32)
                + float(normal[1]) * ys.astype(np.float32)
                + float(normal[2]) * depth.astype(np.float32)
                + distance
            )

            existing = residual[mask > 0]
            if existing.size < 500:
                return mask

            tolerance = float(np.percentile(existing, 97))
            object_band = cv2.dilate(protected, np.ones((17, 17), np.uint8))
            candidate = (object_band > 0) & (mask == 0) & (residual <= tolerance)

            recovered = cv2.morphologyEx(
                candidate.astype(np.uint8) * 255,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            )
            mask = cv2.bitwise_or(mask, recovered)
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            )
            return mask
        except (TypeError, ValueError, AttributeError, IndexError):
            return mask
