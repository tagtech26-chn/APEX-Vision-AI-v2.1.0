"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

VALID_PATTERNS = {"Straight", "Brick", "Herringbone", "Chevron"}
VALID_MATERIAL_PROFILES = {"generic", "ceramic", "stone", "wood", "vinyl", "carpet"}


class RenderRequest(BaseModel):
    room: int = Field(ge=1, description="Room id (1-based).")
    tile: int = Field(ge=1, description="Tile id from the catalog.")
    tile_size: int = Field(default=600, ge=100, le=3000)
    grout_width: int = Field(default=2, ge=0, le=20)
    grout_color: list[int] = Field(default=[220, 220, 220], max_length=3)
    pattern: str = Field(default="Straight")
    material_profile: Literal["auto", "generic", "ceramic", "stone", "wood", "vinyl", "carpet"] = Field(
        default="generic",
        description="Surface-specific rendering profile, or auto for deterministic baseline classification.",
    )
    smart_removal: bool = Field(default=True, description="Preserve detected foreground objects above the floor material.")
    furniture_shadow: bool = Field(default=True, description="Retain furniture/object shading during floor compositing.")
    enhance_lighting: bool = Field(default=True, description="Apply low-frequency room illumination to the projected material.")
    surface: Literal["Floor", "Wall", "Ceiling"] = Field(default="Floor")
    environment: Literal["Interior", "Exterior"] = Field(default="Interior")
    visualization_mode: Literal["Realistic", "Material Only"] = Field(default="Realistic")

    @field_validator("grout_color")
    @classmethod
    def validate_color(cls, value: list[int]) -> list[int]:
        if len(value) != 3:
            raise ValueError("grout_color must contain exactly 3 values (B, G, R).")
        if any(not 0 <= c <= 255 for c in value):
            raise ValueError("grout_color values must be in range 0..255.")
        return value

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, value: str) -> str:
        if value not in VALID_PATTERNS:
            raise ValueError(
                f"Unsupported pattern {value!r}. Expected one of {sorted(VALID_PATTERNS)}."
            )
        return value


class RenderResponse(BaseModel):
    success: bool = True
    image: str
    filename: str
