"""Optional Gemini Robotics spatial advisor for room geometry.

This module is deliberately advisory: it never replaces deterministic OpenCV
rendering. It asks a vision-language model for a normalized floor quadrilateral
and validates the result before the v2.2 scene analyzer uses it.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger("apex.ai")


class GeminiSpatialAdvisor:
    """Ask Gemini Robotics-ER for a floor quad and scene-space hints."""

    name = "gemini_robotics_er_1.6"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv(
            "APEX_V22_GEMINI_MODEL", "gemini-robotics-er-1.6-preview"
        )
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()

    @property
    def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import google.genai  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def _schema() -> dict:
        return {
            "type": "object",
            "properties": {
                "floor_visible": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "floor_quad": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
                "notes": {"type": "string"},
            },
            "required": ["floor_visible", "confidence", "floor_quad", "notes"],
        }

    def analyze(self, image_path: str | Path) -> dict:
        if not self.available:
            raise RuntimeError("Gemini spatial advisor is not configured.")

        from google import genai
        from google.genai import types

        image_path = Path(image_path)
        image_bytes = image_path.read_bytes()
        suffix = image_path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"

        prompt = """
You are the spatial geometry module for a tile visualization system.
Analyze this room photograph and identify ONLY the visible walkable floor plane.
Do not treat rugs, tables, sofas, cabinets, walls, doors, or ceiling as floor.
Return the four corners of the visible floor region as a convex quadrilateral.
Coordinates MUST be normalized to 0..1 where x=0 is the left edge and y=0 is the top.
Use the outer visible floor boundary, including the lower image boundary when the
floor continues out of frame. The four points should follow image order around the
quadrilateral. Do not invent floor behind furniture. If the floor boundary is
uncertain, lower confidence rather than guessing.
"""

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_json_schema=self._schema(),
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini spatial advisor returned no text.")
        result = json.loads(response.text)
        return self._validate(result)

    @staticmethod
    def _validate(result: dict) -> dict:
        if not isinstance(result, dict):
            raise ValueError("Gemini geometry result is not an object.")
        confidence = float(result.get("confidence", 0.0))
        quad = np.asarray(result.get("floor_quad", []), dtype=np.float32)
        if not bool(result.get("floor_visible", False)):
            raise ValueError("Gemini did not confirm a visible floor.")
        if quad.shape != (4, 2) or not np.isfinite(quad).all():
            raise ValueError("Gemini floor_quad must contain four finite points.")
        if confidence < 0.55:
            raise ValueError(f"Gemini floor confidence too low: {confidence:.3f}")
        if np.any(quad < 0) or np.any(quad > 1):
            raise ValueError("Gemini floor_quad contains coordinates outside 0..1.")

        # Convexity and non-degeneracy checks in normalized coordinates.
        cross = []
        for i in range(4):
            a = quad[(i + 1) % 4] - quad[i]
            b = quad[(i + 2) % 4] - quad[(i + 1) % 4]
            cross.append(float(np.cross(a, b)))
        if not all(v > 1e-5 for v in cross) and not all(v < -1e-5 for v in cross):
            raise ValueError("Gemini floor_quad is not a convex quadrilateral.")

        return {
            "floor_visible": True,
            "confidence": confidence,
            "floor_quad": quad.tolist(),
            "notes": str(result.get("notes", "")),
        }
