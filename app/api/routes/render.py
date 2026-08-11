"""Render endpoint with bounded job state and runtime instrumentation."""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.api.deps import services
from app.api.schemas import RenderRequest
from app.core.metrics import render_metrics

logger = logging.getLogger("apex.api.render")
router = APIRouter(prefix="/api/render", tags=["Render"])
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, dict] = {}
_room_current: dict[str, str] = {}
_JOB_TTL_SECONDS = 30 * 60
_TERMINAL_STATUSES = {"done", "error", "superseded"}


def _prune_old_jobs() -> None:
    cutoff = time.time() - _JOB_TTL_SECONDS
    expired = [job_id for job_id, job in _jobs.items() if job.get("status") in _TERMINAL_STATUSES and job.get("created_at", 0) < cutoff]
    for job_id in expired:
        _jobs.pop(job_id, None)
    for room_key, job_id in list(_room_current.items()):
        if job_id not in _jobs:
            _room_current.pop(room_key, None)


def _job(job_id: str, **fields) -> dict:
    return _jobs.setdefault(job_id, {"job_id": job_id, "created_at": time.time(), **fields})


def _is_current(job_id: str, room_key: str) -> bool:
    return _room_current.get(room_key) == job_id


def _run_render_job(job_id: str, room_path: str, room_key: str, tile_path: str, **params) -> None:
    started_at = time.perf_counter()
    render_metrics.started()

    def report(fraction: float, message: str) -> None:
        job = _jobs.get(job_id)
        if job is not None:
            job["progress"] = max(0.0, min(1.0, fraction))
            job["message"] = message

    def finish(status: str, message: str) -> None:
        final = _jobs.setdefault(job_id, {})
        final.update(status=status, message=message)

    job = _jobs.get(job_id)
    if job is not None:
        job["status"] = "processing"

    if not _is_current(job_id, room_key):
        render_metrics.superseded()
        finish("superseded", "Superseded by a newer request")
        return

    try:
        output = services.render.render(room_path=room_path, tile_path=tile_path, progress_cb=report, **params)
        duration = time.perf_counter() - started_at
        if not _is_current(job_id, room_key):
            render_metrics.superseded()
            finish("superseded", "Superseded by a newer request")
            return
        filename = Path(output).name
        final = _jobs.setdefault(job_id, {})
        final.update(status="done", progress=1.0, message="Done", image=f"/output/{filename}", filename=filename, duration_seconds=round(duration, 4))
        render_metrics.finished(duration)
        logger.info("Render job completed job_id=%s duration_seconds=%.3f", job_id, duration)
    except Exception:
        duration = time.perf_counter() - started_at
        render_metrics.failed(duration)
        logger.exception("Render job failed job_id=%s duration_seconds=%.3f", job_id, duration)
        job = _jobs.setdefault(job_id, {})
        job.update(status="error", message="Render failed", duration_seconds=round(duration, 4))


@router.post("")
async def render(request: RenderRequest):
    _prune_old_jobs()
    try:
        room = services.rooms.get_room(request.room)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        tile = services.tiles.get_tile(request.tile)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    room_key = Path(room).stem
    previous = _room_current.get(room_key)
    if previous:
        old = _jobs.get(previous)
        if old is not None and old.get("status") in ("queued", "processing"):
            old["status"] = "superseded"
            old["message"] = "Superseded by a newer request"
            render_metrics.superseded()

    job_id = uuid.uuid4().hex[:10]
    _room_current[room_key] = job_id
    _job(
        job_id,
        status="queued",
        progress=0.0,
        message="Queued",
        material_profile=request.material_profile,
        tile_size=request.tile_size,
        grout_width=request.grout_width,
        pattern=request.pattern,
    )
    _executor.submit(
        _run_render_job,
        job_id,
        str(room),
        room_key,
        tile["image_path"],
        tile_size_mm=request.tile_size,
        grout_width=request.grout_width,
        grout_color=tuple(request.grout_color),
        pattern=request.pattern,
        material_profile=request.material_profile,
        smart_removal=request.smart_removal,
        furniture_shadow=request.furniture_shadow,
        enhance_lighting=request.enhance_lighting,
        surface=request.surface,
        environment=request.environment,
        visualization_mode=request.visualization_mode,
    )
    return {"job_id": job_id, "status": "queued", "progress": 0.0, "message": "Queued", "material_profile": request.material_profile}


@router.get("/metrics")
def metrics():
    return {"success": True, "metrics": render_metrics.snapshot()}


@router.get("/{job_id}")
def render_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return job
