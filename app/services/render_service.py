"""Render service: orchestrates scene analysis + tile rendering."""

from __future__ import annotations

import logging
import os
import pickle
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2

from app.ai.material.classifier import classify_surface
from app.ai.quality import SceneQualityEvaluator
from app.ai.scene.geometry_quality import evaluate_floor_geometry
from app.ai.scene.result import SceneResult
from app.cache.scene_cache import SceneCache
from app.core.config import settings
from app.core.metrics import render_metrics
from app.renderer.tile_renderer import TileRenderer

logger = logging.getLogger("apex.render")
_ANALYZER_LOCK = threading.Lock()
_TILE_CACHE_LOCK = threading.Lock()
_TILE_CACHE: OrderedDict[str, object] = OrderedDict()
_QUALITY_EVALUATOR = SceneQualityEvaluator()
_SCENE_PIPELINE_VERSION = "v22-geometry-safe-5"


class RenderService:
    """Loads/builds a SceneResult, then renders a material with bounded reuse."""

    def __init__(self, analyzer=None, cache: SceneCache | None = None, renderer: TileRenderer | None = None) -> None:
        self.cache = cache or SceneCache()
        self.renderer = renderer or TileRenderer(debug=settings.write_debug_images)
        self.analyzer = analyzer

    def get_analyzer(self):
        if self.analyzer is None:
            with _ANALYZER_LOCK:
                if self.analyzer is None:
                    from app.ai.scene.analyzer import build_scene_analyzer
                    self.analyzer = build_scene_analyzer(settings.ai_provider)
        return self.analyzer

    @staticmethod
    def _tile_fingerprint(tile_path: Path) -> str:
        stat = tile_path.stat()
        return f"{tile_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"

    @staticmethod
    def _load_tile(tile_path: Path):
        key = RenderService._tile_fingerprint(tile_path)
        with _TILE_CACHE_LOCK:
            cached = _TILE_CACHE.get(key)
            if cached is not None:
                _TILE_CACHE.move_to_end(key)
                return cached
        tile = cv2.imread(str(tile_path))
        if tile is None:
            raise ValueError(f"Unable to load tile: {tile_path}")
        max_dim = settings.tile_texture_max_dim
        height, width = tile.shape[:2]
        if max(height, width) > max_dim:
            scale = max_dim / float(max(height, width))
            tile = cv2.resize(tile, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        with _TILE_CACHE_LOCK:
            if settings.tile_cache_max_items > 0:
                _TILE_CACHE[key] = tile
                _TILE_CACHE.move_to_end(key)
                while len(_TILE_CACHE) > settings.tile_cache_max_items:
                    _TILE_CACHE.popitem(last=False)
        return tile

    def render(self, room_path: str | Path, tile_path: str | Path, tile_size_mm: int = 600, grout_width: int = 2,
               grout_color=(220, 220, 220), pattern: str = "Straight", material_profile: str = "auto",
               alpha: float = 0.92, smart_removal: bool = True, furniture_shadow: bool = True,
               enhance_lighting: bool = True, surface: str = "Floor", environment: str = "Interior",
               visualization_mode: str = "Realistic", progress_cb=None) -> str:
        started = time.perf_counter()
        room_path = Path(room_path)
        tile_path = Path(tile_path)
        if not room_path.exists():
            raise FileNotFoundError(f"Room image not found: {room_path}")
        if not tile_path.exists():
            raise FileNotFoundError(f"Tile image not found: {tile_path}")

        def report(fraction: float, message: str) -> None:
            if progress_cb is not None:
                progress_cb(fraction, message)

        room_key = room_path.stem
        scene_started = time.perf_counter()
        scene = self._load_or_build_scene(room_path, progress_cb=progress_cb)
        render_metrics.stage("scene", time.perf_counter() - scene_started)

        quality_started = time.perf_counter()
        quality = _QUALITY_EVALUATOR.evaluate(scene)
        scene.metadata["quality"] = quality
        geometry_quality = evaluate_floor_geometry(scene.floor_mask, scene.floor_polygon, scene.homography)
        scene.metadata["geometry_quality"] = geometry_quality.as_dict()
        render_metrics.stage("quality", time.perf_counter() - quality_started)
        logger.info("[AI] Scene quality score=%.2f grade=%s floor_coverage=%.4f depth_valid=%.4f geometry_score=%.4f perspective=%.4f",
                    quality["score"], quality["grade"], quality["floor_coverage"], quality["depth_valid_ratio"],
                    geometry_quality.score, geometry_quality.perspective_score)

        tile_started = time.perf_counter()
        tile = self._load_tile(tile_path)
        render_metrics.stage("tile_load", time.perf_counter() - tile_started)

        resolved_profile = material_profile
        if material_profile == "auto":
            classify_started = time.perf_counter()
            material_intelligence = classify_surface(tile)
            resolved_profile = str(material_intelligence["material"])
            scene.metadata["material_classification"] = material_intelligence
            render_metrics.stage("material_classification", time.perf_counter() - classify_started)
            logger.info("[AI] Material baseline profile=%s finish=%s confidence=%.4f scale_factor=%.4f",
                        resolved_profile, material_intelligence["finish"], material_intelligence["confidence"],
                        material_intelligence["texture_scale_factor"])
        else:
            material_intelligence = {"material": resolved_profile, "finish": "satin", "confidence": 1.0,
                                     "texture_scale_factor": 1.0, "method": "explicit-profile"}
            scene.metadata["material_classification"] = material_intelligence

        scene.metadata["visualizer_options"] = {"smart_removal": smart_removal, "furniture_shadow": furniture_shadow,
                                                  "enhance_lighting": enhance_lighting, "surface": surface,
                                                  "environment": environment, "visualization_mode": visualization_mode}
        report(0.92, "Rendering tiles...")
        logger.info("Rendering room=%s tile=%s size=%smm grout=%s pattern=%s material=%s finish=%s lighting=%s shadow=%s removal=%s",
                    room_key, tile_path.name, tile_size_mm, grout_width, pattern, resolved_profile,
                    material_intelligence["finish"], enhance_lighting, furniture_shadow, smart_removal)
        render_started = time.perf_counter()
        result = self.renderer.render(scene=scene, tile=tile, tile_size_mm=tile_size_mm, grout_width=grout_width,
                                      grout_color=grout_color, pattern=pattern, alpha=alpha,
                                      material_profile=resolved_profile, material_intelligence=material_intelligence,
                                      smart_removal=smart_removal, furniture_shadow=furniture_shadow,
                                      enhance_lighting=enhance_lighting, visualization_mode=visualization_mode)
        render_metrics.stage("render", time.perf_counter() - render_started)

        report(0.97, "Writing image...")
        output_dir = settings.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{room_key}_render.png"
        write_started = time.perf_counter()
        if not cv2.imwrite(str(output_file), result):
            raise IOError(f"Unable to write render output: {output_file}")
        render_metrics.stage("write", time.perf_counter() - write_started)
        render_metrics.stage("total", time.perf_counter() - started)
        report(1.0, "Done")
        return str(output_file)

    def _cache_key(self, room_path: Path) -> str:
        analyzer = self.get_analyzer()
        providers = analyzer.providers
        pipeline = "v22" if analyzer.__class__.__name__ == "V22SceneAnalyzer" else "v6"
        geometry_advisor = os.getenv("APEX_V22_GEOMETRY_ADVISOR", "off").strip().lower() if pipeline == "v22" else "off"
        return (f"{room_path.stem}__{pipeline}__{_SCENE_PIPELINE_VERSION}__"
                f"{providers['detector']}__{providers['segmenter']}__{providers['depth']}__advisor-{geometry_advisor}")

    @staticmethod
    def _source_fingerprint(room_path: Path) -> str:
        stat = room_path.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"

    def _load_or_build_scene(self, room_path: Path, progress_cb=None) -> SceneResult:
        room_key = self._cache_key(room_path)
        fingerprint = self._source_fingerprint(room_path)
        if not self.cache.enabled:
            render_metrics.cache_miss()
            logger.warning("[CACHE] Disabled: APEX_CACHE_SIGNING_KEY is not configured")
        elif self.cache.exists(room_key):
            try:
                scene = self.cache.load(room_key)
                cached_fp = scene.metadata.get("source_fingerprint")
                cached_pipeline = scene.metadata.get("scene_pipeline_version")
                cached_geometry_version = scene.metadata.get("v22_geometry", {}).get("version")
                compatible = (cached_pipeline == _SCENE_PIPELINE_VERSION and
                              (cached_geometry_version is None or cached_geometry_version == "2.2-geometry-safe"))
                if compatible and cached_fp == fingerprint:
                    render_metrics.cache_hit()
                    logger.info("[CACHE] Hit: %s", room_key)
                    if progress_cb is not None:
                        progress_cb(0.1, "Loading cached scene...")
                    return scene
                render_metrics.cache_miss()
                logger.info("[CACHE] Scene pipeline/source changed for %s, re-analysing", room_path.name)
            except (OSError, TypeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
                render_metrics.cache_error()
                render_metrics.cache_miss()
                logger.warning("[CACHE] Invalid cache for %s: %s", room_path.name, exc)
        else:
            render_metrics.cache_miss()

        logger.info("[AI] Analysing room: %s", room_path.name)
        analyzer = self.get_analyzer()
        scene = analyzer.analyze(room_path, progress_cb=progress_cb)
        scene.metadata["source_fingerprint"] = fingerprint
        scene.metadata["scene_pipeline_version"] = _SCENE_PIPELINE_VERSION
        self.cache.save(room_key, scene)
        return scene
