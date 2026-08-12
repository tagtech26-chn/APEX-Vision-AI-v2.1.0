"""Metric3Dv2 provider for the v2.2 AI geometry lab."""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from app.ai.depth.base import DepthEstimator

logger = logging.getLogger("apex.ai")


class Metric3DProvider(DepthEstimator):
    """Lazy Metric3Dv2 provider exposing raw metric depth for geometry.

    On CPU development machines Metric3D's upstream OpenMMLab dependency stack
    may not be installed or may not have a compatible Windows wheel. In that
    case the provider uses an explicitly-labelled deterministic geometry prior
    instead of taking the whole render service down. CUDA/model deployments
    remain strict and still fail fast when the real model cannot load.
    """

    name = "metric3d_v2"

    def __init__(self, model_name: str = "metric3d_vit_small", device: str | None = None) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = model_name
        self.model = None
        self.last_metric_depth: np.ndarray | None = None
        self.last_normals: np.ndarray | None = None
        self.last_confidence: np.ndarray | None = None
        self.runtime_mode = "model"
        self.runtime_error: str | None = None

    def _load(self) -> None:
        if self.model is not None or self.runtime_mode == "fallback":
            return
        if self.device.type == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("APEX_V22_DEVICE=cuda but CUDA is not available.")

        try:
            self.model = self.torch.hub.load("yvanyin/metric3d", self.model_name, pretrain=True)
            self.model.to(self.device).eval()
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as exc:
            allow_fallback = self.device.type == "cpu" and os.getenv("APEX_V22_DEPTH_FALLBACK", "1").strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
            if not allow_fallback:
                raise RuntimeError(
                    "Metric3D could not be loaded. Install the compatible Metric3D/OpenMMLab dependencies "
                    "or set APEX_V22_DEPTH_FALLBACK=1 for CPU development mode."
                ) from exc
            self.runtime_mode = "fallback"
            self.runtime_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "[V2.2] Metric3D model unavailable on CPU; using deterministic metric geometry fallback. "
                "Original error: %s",
                self.runtime_error,
            )

    @staticmethod
    def _fallback_metric_depth(image: np.ndarray) -> np.ndarray:
        """Produce a stable metre-scale depth prior for CPU development.

        This is deliberately not presented as learned metric depth. It keeps
        the complete V2.2 geometry/render pipeline executable until a compatible
        Metric3D runtime is installed.
        """
        h, w = image.shape[:2]
        yy = np.linspace(0.8, 4.8, h, dtype=np.float32)
        depth = np.tile(yy[:, None], (1, w))
        x_weights = np.linspace(0.96, 1.04, w, dtype=np.float32)
        depth *= x_weights[None, :]
        depth = cv2.GaussianBlur(depth, (0, 0), 2)
        return depth.astype(np.float32)

    def predict(self, image: np.ndarray) -> np.ndarray:
        torch = self.torch
        self._load()
        if self.runtime_mode == "fallback":
            metric_depth = self._fallback_metric_depth(image)
            self.last_metric_depth = metric_depth
            self.last_normals = None
            self.last_confidence = np.ones_like(metric_depth, dtype=np.float32)
            return metric_depth

        rgb_origin = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = rgb_origin.shape[:2]
        input_h, input_w = 616, 1064
        scale = min(input_h / original_h, input_w / original_w)
        resized_w = max(1, int(round(original_w * scale)))
        resized_h = max(1, int(round(original_h * scale)))
        rgb = cv2.resize(rgb_origin, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        top = (input_h - resized_h) // 2
        bottom = input_h - resized_h - top
        left = (input_w - resized_w) // 2
        right = input_w - resized_w - left
        rgb = cv2.copyMakeBorder(
            rgb,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=[123.675, 116.28, 103.53],
        )
        mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)[:, None, None]
        tensor = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        tensor = ((tensor - mean) / std)[None].to(self.device)

        with torch.inference_mode():
            pred_depth, confidence, output = self.model.inference({"input": tensor})

        depth = pred_depth.squeeze()
        depth = depth[top : input_h - bottom, left : input_w - right]
        depth = torch.nn.functional.interpolate(
            depth[None, None], (original_h, original_w), mode="bilinear", align_corners=False
        ).squeeze()
        metric_depth = torch.clamp(depth, min=0).detach().float().cpu().numpy().astype(np.float32)

        normals = None
        if isinstance(output, dict) and "prediction_normal" in output:
            normal = output["prediction_normal"][:, :3]
            normal = normal[:, :, top : input_h - bottom, left : input_w - right]
            normal = torch.nn.functional.interpolate(
                normal, (original_h, original_w), mode="bilinear", align_corners=False
            ).squeeze(0).detach().float().cpu().numpy().transpose(1, 2, 0)
            normals = normal / np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-6)

        confidence_np = confidence.squeeze().detach().float().cpu().numpy()
        confidence_np = cv2.resize(confidence_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        self.last_metric_depth = metric_depth
        self.last_normals = normals
        self.last_confidence = confidence_np
        # V2.2 geometry needs metric values; unlike the v2.1 depth contract this is
        # deliberately not normalised to 0..1.
        return metric_depth

    @property
    def supports_metric_geometry(self) -> bool:
        return True
