"""SAM2 segmentation provider (heavy provider)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.ai.segmentation.base import Segmenter
from app.core.config import settings


class SAM2Provider(Segmenter):
    """Segment Anything 2 box-prompted segmenter."""

    name = "sam2"

    def __init__(
        self,
        config_dir: str | None = None,
        config_file: str | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        self.config_dir = config_dir or settings.sam2_config_dir
        self.config_file = config_file or settings.sam2_config_file
        self.checkpoint_path = checkpoint_path or settings.sam2_ckpt

        config_path = Path(self.config_dir) / self.config_file
        if not config_path.exists():
            raise FileNotFoundError(f"SAM2 config not found: {config_path}")
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.checkpoint_path}")

        from hydra.core.global_hydra import GlobalHydra
        from hydra import initialize_config_dir

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        initialize_config_dir(version_base=None, config_dir=self.config_dir)

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.model = build_sam2(
            config_file=self.config_file,
            ckpt_path=self.checkpoint_path,
            device="cpu",
            mode="eval",
        )
        self.predictor = SAM2ImagePredictor(self.model)

    @staticmethod
    def _floor_candidate_score(mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
        """Score a SAM2 floor candidate using perspective-room priors."""
        binary = mask > 0
        h, w = binary.shape
        area = float(binary.mean())
        if area <= 0.005:
            return -1e9

        x1, y1, x2, y2 = box
        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))

        bottom_band = binary[max(0, h - max(8, h // 20) ) :]
        bottom_touch = float(bottom_band.mean())
        lower_support = float(binary[h // 2 :].mean())
        box_mask = np.zeros_like(binary)
        box_mask[y1:y2, x1:x2] = True
        inside_ratio = float((binary & box_mask).sum()) / max(float(binary.sum()), 1.0)

        ys, _ = np.where(binary)
        centroid_y = float(ys.mean()) / max(h - 1, 1)
        edge = max(2, w // 80)
        side_touch = float(binary[:, :edge].mean() + binary[:, -edge:].mean())

        return (
            4.0 * bottom_touch
            + 2.0 * lower_support
            + 1.5 * inside_ratio
            + centroid_y
            - 1.5 * max(0.0, area - 0.80)
            - 0.25 * side_touch
        )

    def _segment_floor(
        self,
        image: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Use SAM2's multi-mask output and choose the floor-like hypothesis."""
        masks, _, _ = self.predictor.predict(
            box=np.asarray(box, dtype=np.float32)[None, :],
            multimask_output=True,
        )
        candidates = [(m * 255).astype(np.uint8) for m in masks]
        return max(candidates, key=lambda m: self._floor_candidate_score(m, box))

    def segment(
        self,
        image: np.ndarray,
        box: tuple[int, int, int, int] | None = None,
        points: np.ndarray | None = None,
    ) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        if box is not None:
            return self._segment_floor(image, box)

        kwargs: dict = {"multimask_output": False}
        if points is not None:
            kwargs["point_coords"] = points.astype(np.float32)
            kwargs["point_labels"] = np.ones(len(points), dtype=np.int32)
        masks, _, _ = self.predictor.predict(**kwargs)
        return (masks[0] * 255).astype(np.uint8)

    def segment_many(
        self,
        image: np.ndarray,
        boxes: list[tuple[int, int, int, int]] | None = None,
        points: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Segment several prompts after a single image embedding."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)

        if boxes is None:
            kwargs: dict = {"multimask_output": False}
            if points is not None:
                kwargs["point_coords"] = points.astype(np.float32)
                kwargs["point_labels"] = np.ones(len(points), dtype=np.int32)
            masks, _, _ = self.predictor.predict(**kwargs)
            return [(masks[0] * 255).astype(np.uint8)]

        results: list[np.ndarray] = []
        for box in boxes:
            masks, _, _ = self.predictor.predict(
                box=np.asarray(box, dtype=np.float32)[None, :],
                multimask_output=False,
            )
            results.append((masks[0] * 255).astype(np.uint8))
        return results
