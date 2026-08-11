"""AI pipeline configuration and provider selection."""

from __future__ import annotations

from app.core.config import settings

VALID_PROVIDERS = {"auto", "heavy", "light", "v22"}

DETECTION_PROMPTS = {
    "floor": "floor, ground, tiles, marble floor, ceramic floor",
    "wall": "wall",
    "ceiling": "ceiling",
    "furniture": "table, chair, sofa, bed, cabinet, door, window",
}


def resolve_provider(requested: str | None = None) -> str:
    """Normalise the requested provider and validate it."""
    provider = (requested or settings.ai_provider).strip().lower()
    if provider not in VALID_PROVIDERS:
        raise ValueError(
            f"Unknown AI provider {provider!r}. "
            f"Expected one of: {', '.join(sorted(VALID_PROVIDERS))}."
        )
    return provider


def heavy_models_available() -> tuple[bool, list[str]]:
    """Return whether the original heavy model stack is reachable plus missing parts."""
    missing: list[str] = []
    try:
        import groundingdino  # noqa: F401
    except Exception:
        missing.append("groundingdino")
    try:
        import sam2  # noqa: F401
    except Exception:
        missing.append("sam2")
    return (not missing), missing


def v22_models_available() -> tuple[bool, list[str]]:
    """Check only import-level availability; checkpoints remain lazy-loaded."""
    missing: list[str] = []
    try:
        import sam3  # noqa: F401
    except Exception:
        missing.append("sam3")
    depth = getattr(settings, "v22_depth_provider", "metric3d")
    if depth == "metric3d":
        try:
            import torch  # noqa: F401
        except Exception:
            missing.append("torch")
    elif depth == "unidepth":
        try:
            import unidepth  # noqa: F401
        except Exception:
            missing.append("unidepth")
    else:
        missing.append(f"unknown depth provider: {depth}")
    return (not missing), missing
