"""Optional SAM3.1 text-prompt segmenter for the v2.2 geometry lab."""

from __future__ import annotations

import numpy as np

from app.ai.segmentation.base import Segmenter


class SAM3Provider(Segmenter):
    """SAM3.1 text-prompt segmentation with lazy checkpoint loading."""

    name = "sam3.1"

    def __init__(self, prompt: str = "floor", device: str | None = None) -> None:
        import torch
        from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
        from sam3.model.sam3_image_processor import Sam3Processor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt = prompt
        checkpoint = download_ckpt_from_hf(version="sam3.1")
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
            raise ValueError("SAM3 returned no masks.")
        areas = array.reshape(array.shape[0], -1).sum(axis=1)
        return (array[int(np.argmax(areas))] > 0.5).astype(np.uint8) * 255

    def segment_text(self, image: np.ndarray, prompt: str) -> np.ndarray:
        from PIL import Image
        rgb = image[:, :, ::-1]
        state = self.processor.set_image(Image.fromarray(rgb))
        output = self.processor.set_text_prompt(state=state, prompt=prompt)
        return self._pick_mask(output["masks"])

    def segment(self, image: np.ndarray, box=None, points=None) -> np.ndarray:
        return self.segment_text(image, self.prompt)

    def segment_many(self, image: np.ndarray, boxes=None):
        mask = self.segment_text(image, "sofa, couch, chair, table, cabinet, rug, plant, lamp, furniture")
        if boxes is None:
            return [mask]
        return [mask.copy() for _ in boxes]
