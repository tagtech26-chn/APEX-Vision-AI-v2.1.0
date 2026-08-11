"""SAM3 concept segmentation for the isolated v2.2 geometry lab."""

from __future__ import annotations

import numpy as np

from app.ai.segmentation.base import Segmenter


class SAM3Provider(Segmenter):
    """SAM3 image concept segmentation for floor and object occlusion."""

    name = "sam3_concept"

    def __init__(self, prompt: str = "floor", device: str | None = None) -> None:
        import torch
        from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
        from sam3.model.sam3_image_processor import Sam3Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt = prompt
        checkpoint = download_ckpt_from_hf(version="sam3")
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint,
            load_from_HF=False,
            device=self.device,
        )
        self.processor = Sam3Processor(self.model)

    @staticmethod
    def _pick_mask(masks) -> np.ndarray:
        array = masks.detach().float().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
        if array.ndim == 4:
            array = array[:, 0]
        if array.ndim == 2:
            array = array[None]
        if array.shape[0] == 0:
            return np.zeros(array.shape[-2:], dtype=np.uint8)
        # SAM3 may return several concepts/instances. Preserve their union instead
        # of selecting the largest object, which is important for furniture occlusion.
        return (np.any(array > 0.5, axis=0)).astype(np.uint8) * 255

    def _set_image(self, image: np.ndarray):
        from PIL import Image
        rgb = image[:, :, ::-1]
        return self.processor.set_image(Image.fromarray(rgb))

    def segment_text(self, image: np.ndarray, prompt: str) -> np.ndarray:
        state = self._set_image(image)
        output = self.processor.set_text_prompt(state=state, prompt=prompt)
        return self._pick_mask(output.get("masks", []))

    def segment(self, image: np.ndarray, box=None, points=None) -> np.ndarray:
        mask = self.segment_text(image, self.prompt)
        if box is None:
            return mask
        x0, y0, x1, y1 = map(int, box)
        constrained = np.zeros_like(mask)
        x0, x1 = max(0, x0), min(mask.shape[1], x1)
        y0, y1 = max(0, y0), min(mask.shape[0], y1)
        if x1 > x0 and y1 > y0:
            constrained[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        return constrained

    def segment_many(self, image: np.ndarray, boxes=None):
        if not boxes:
            return [self.segment_text(image, "sofa, couch, chair, table, cabinet, rug, plant, lamp, furniture")]
        # A single concept pass is much cheaper than one SAM3 inference per object.
        # Constrain the semantic union to each GroundingDINO box so an object's mask
        # cannot occlude unrelated floor regions.
        union = self.segment_text(image, "sofa, couch, chair, table, cabinet, rug, plant, lamp, furniture")
        masks = []
        for box in boxes:
            x0, y0, x1, y1 = map(int, box)
            mask = np.zeros_like(union)
            x0, x1 = max(0, x0), min(union.shape[1], x1)
            y0, y1 = max(0, y0), min(union.shape[0], y1)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = union[y0:y1, x0:x1]
            masks.append(mask)
        return masks
