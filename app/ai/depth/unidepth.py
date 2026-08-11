"""UniDepthV2 provider for the v2.2 geometry benchmark."""

from __future__ import annotations

import cv2
import numpy as np

from app.ai.depth.base import DepthEstimator


class UniDepthV2Provider(DepthEstimator):
    """Lazy UniDepthV2 provider exposing metric depth, points and intrinsics."""

    name = "unidepth_v2"

    def __init__(self, model_name: str = "lpiccinelli/UniDepth-v2-vitl14", device: str | None = None) -> None:
        import torch
        from unidepth.models import UniDepthV2

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = UniDepthV2.from_pretrained(model_name).to(self.device).eval()
        self.model.resolution_level = 8
        self.last_metric_depth: np.ndarray | None = None
        self.last_points: np.ndarray | None = None
        self.last_intrinsics: np.ndarray | None = None
        self.last_confidence: np.ndarray | None = None

    def predict(self, image: np.ndarray) -> np.ndarray:
        torch = self.torch
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(self.device)
        with torch.no_grad():
            predictions = self.model.infer(tensor)

        depth = predictions["depth"].squeeze().detach().float().cpu().numpy()
        points = predictions.get("points")
        intrinsics = predictions.get("intrinsics")
        confidence = predictions.get("confidence")

        self.last_metric_depth = depth
        self.last_points = points.squeeze().detach().float().cpu().numpy() if points is not None else None
        self.last_intrinsics = intrinsics.squeeze().detach().float().cpu().numpy() if intrinsics is not None else None
        self.last_confidence = confidence.squeeze().detach().float().cpu().numpy() if confidence is not None else None
        return self.normalise(depth)
