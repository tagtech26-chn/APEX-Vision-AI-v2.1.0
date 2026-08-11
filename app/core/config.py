"""Application settings, driven by environment variables with sane defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(slots=True)
class Settings:
    """Runtime configuration for the APEX Vision AI application."""

    host: str = field(default_factory=lambda: _env("APEX_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("APEX_PORT", 8000))
    debug: bool = field(default_factory=lambda: _env_bool("APEX_DEBUG", False))
    write_debug_images: bool = field(default_factory=lambda: _env_bool("APEX_WRITE_DEBUG", False))
    app_version: str = field(default_factory=lambda: _env("APEX_VERSION", "2.1.0"))

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    assets_dir: Path = field(default_factory=Path)
    output_dir: Path = field(default_factory=Path)
    uploads_dir: Path = field(default_factory=Path)
    catalog_dir: Path = field(default_factory=Path)
    scenes_dir: Path = field(default_factory=Path)

    ai_provider: str = field(default_factory=lambda: _env("APEX_AI_PROVIDER", "auto").lower())

    allowed_hosts: tuple[str, ...] = field(default_factory=lambda: _env_list("APEX_ALLOWED_HOSTS", ("localhost", "127.0.0.1")))
    hsts_enabled: bool = field(default_factory=lambda: _env_bool("APEX_HSTS_ENABLED", False))

    grounding_dino_config: str = field(default_factory=lambda: _env("GROUNDING_DINO_CONFIG"))
    grounding_dino_ckpt: str = field(default_factory=lambda: _env("GROUNDING_DINO_CKPT"))
    sam2_config_dir: str = field(default_factory=lambda: _env("SAM2_CONFIG_DIR"))
    sam2_config_file: str = field(default_factory=lambda: _env("SAM2_CONFIG_FILE", "sam2.1/sam2.1_hiera_l.yaml"))
    sam2_ckpt: str = field(default_factory=lambda: _env("SAM2_CKPT"))
    depth_anything_root: str = field(default_factory=lambda: _env("DEPTH_ANYTHING_ROOT"))
    depth_anything_ckpt: str = field(default_factory=lambda: _env("DEPTH_ANYTHING_CKPT"))

    render_tile_size_mm: int = field(default_factory=lambda: _env_int("APEX_TILE_MM", 600))
    render_grout_width: int = field(default_factory=lambda: _env_int("APEX_GROUT", 2))
    render_grout_color: tuple[int, int, int] = field(default_factory=lambda: (220, 220, 220))
    render_alpha: float = field(default_factory=lambda: _env_float("APEX_ALPHA", 0.92))
    render_pattern: str = field(default_factory=lambda: _env("APEX_PATTERN", "Straight"))
    render_max_dim: int = field(default_factory=lambda: _env_int("APEX_RENDER_MAX_DIM", 2048))
    tile_texture_max_dim: int = field(default_factory=lambda: _env_int("APEX_TILE_TEXTURE_MAX_DIM", 1024))
    tile_cache_max_items: int = field(default_factory=lambda: _env_int("APEX_TILE_CACHE_MAX_ITEMS", 32))

    def __post_init__(self) -> None:
        root = self.project_root
        self.assets_dir = Path(_env("APEX_ASSETS", str(root / "assets")))
        self.output_dir = Path(_env("APEX_OUTPUT", str(root / "output")))
        self.uploads_dir = Path(_env("APEX_UPLOADS", str(root / "uploads")))
        self.catalog_dir = Path(_env("APEX_CATALOG", str(root / "catalog")))
        self.scenes_dir = Path(_env("APEX_SCENES", str(self.assets_dir / "scenes")))
        for folder in (self.assets_dir, self.output_dir, self.uploads_dir, self.catalog_dir, self.scenes_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self.validate()

    def validate(self) -> None:
        """Fail fast on invalid production configuration."""
        if self.port < 1 or self.port > 65535:
            raise ValueError("APEX_PORT must be between 1 and 65535")
        if not self.app_version.strip():
            raise ValueError("APEX_VERSION must not be empty")
        if self.ai_provider not in {"auto", "heavy", "light", "v22"}:
            raise ValueError("APEX_AI_PROVIDER must be one of: auto, heavy, light, v22")
        if not self.allowed_hosts:
            raise ValueError("APEX_ALLOWED_HOSTS must contain at least one host")
        if not 0.0 <= self.render_alpha <= 1.0:
            raise ValueError("APEX_ALPHA must be between 0 and 1")
        if self.render_max_dim < 256:
            raise ValueError("APEX_RENDER_MAX_DIM must be at least 256")
        if self.tile_texture_max_dim < 64:
            raise ValueError("APEX_TILE_TEXTURE_MAX_DIM must be at least 64")
        if self.tile_cache_max_items < 0:
            raise ValueError("APEX_TILE_CACHE_MAX_ITEMS must not be negative")
        if self.render_tile_size_mm <= 0:
            raise ValueError("APEX_TILE_MM must be greater than 0")
        if self.render_grout_width < 0:
            raise ValueError("APEX_GROUT must not be negative")

    @property
    def rooms_dir(self) -> Path:
        path = self.assets_dir / "rooms"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def tile_images_dir(self) -> Path:
        path = self.assets_dir / "tiles" / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def tile_thumbs_dir(self) -> Path:
        path = self.assets_dir / "tiles" / "thumbnails"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def catalog_file(self) -> Path:
        return self.catalog_dir / "tiles.json"


settings = Settings()
