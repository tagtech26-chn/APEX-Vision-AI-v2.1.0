"""Metric3Dv2 provider for the v2.2 AI geometry lab."""

from __future__ import annotations

import cv2
import numpy as np

from app.ai.depth.base import DepthEstimator


class Metric3DProvider(DepthEstimator):
    """Lazy Metric3Dv2 provider with metric depth and surface-normal output."""

    name = "metric3d_v2"

    def __init__(self, model_name: str = "metric3d_vit_small", device: str | None = None) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = model_name
        self.model = torch.hub.load("yvanyin/metric3d", model_name, pretrain=True)
        self.model.to(self.device).eval()
        self.last_metric_depth: np.ndarray | None = None
        self.last_normals: np.ndarray | None = None
        self.last_confidence: np.ndarray | None = None

    def predict(self, image: np.ndarray) -> np.ndarray:
        torch = self.torch
        rgb_origin = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = rgb_origin.shape[:2]
        input_size = (616, 1064)
        scale = min(input_size[0] / original_h, input_size[1] / original_w)
        resized_w = max(1, int(original_w * scale))
        resized_h = max(1, int(original_h * scale))
        rgb = cv2.resize(rgb_origin, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.copyMakeBorder(
            rgb,
            (input_size[0] - resized_h) // 2,
            input_size[0] - resized_h - (input_size[0] - resized_h) // 2,
            (input_size[1] - resized_w) // 2,
            input_size[1] - resized_w - (input_size[1] - resized_w) // 2,
            cv2.BORDER_CONSTANT,
            value=[123.675, 116.28, 103.53],
        )
        pad_h = input_size[0] - resized_h
        pad_w = input_size[1] - resized_w
        pad_info = [pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2]
        mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)[:, None, None]
        tensor = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        tensor = ((tensor - mean) / std)[None].to(self.device)

        with torch.no_grad():
            pred_depth, confidence, output = self.model.inference({"input": tensor})

        depth = pred_depth.squeeze()
        depth = depth[pad_info[0] : depth.shape[0] - pad_info[1], pad_info[2] : depth.shape[1] - pad_info[3]]
        depth = torch.nn.functional.interpolate(
            depth[None, None], (original_h, original_w), mode="bilinear", align_corners=False
        ).squeeze()
        depth = depth * (1000.0 * scale / 1000.0)
        depth = torch.clamp(depth, 0, 300)
        metric_depth = depth.detach().float().cpu().numpy()

        normals = None
        if "prediction_normal" in output:
            normal = output["prediction_normal"][:, :3].squeeze()
            normal = normal[:, pad_info[0] : normal.shape[1] - pad_info[1], pad_info[2] : normal.shape[2] - pad_info[3]]
            normal = torch.nn.functional.interpolate(
                normal[None], (original_h, original_w), mode="bilinear", align_corners=False
            ).squeeze(0)
            normals = normal.detach().float().cpu().numpy().transpose(1, 2, 0)
            norm = np.linalg.norm(normals, axis=2, keepdims=True)
            normals = normals / np.maximum(norm, 1e-6)

        self.last_metric_depth = metric_depth
        self.last_normals = normals
        self.last_confidence = confidence.squeeze().detach().float().cpu().numpy()
        return self.normalise(metric_depth)
