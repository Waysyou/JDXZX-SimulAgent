from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import app as pipeline
from src.runtime_support import (
    SimulAgentError,
    checks_ok,
    configure_logging,
    remove_runtime_path,
    run_startup_checks,
)


PROJECT_ROOT = Path(__file__).resolve().parent
UI_ROOT = PROJECT_ROOT / "ui_connected_v4"
SETTINGS = pipeline.SETTINGS
UPLOAD_ROOT = SETTINGS.runtime_root / "v4_uploads"
logger = configure_logging(SETTINGS)


@dataclass
class SessionRecord:
    state: dict[str, Any]
    mode: str
    lock: threading.RLock = field(default_factory=threading.RLock)
    status: str = "等待音频输入"
    pipeline_status: str = "模型已连接"
    file_status: str = "IDLE"
    last_audio_path: str | None = None
    audio_events: deque[dict[str, Any]] = field(default_factory=deque)
    emitted_audio_ids: set[int] = field(default_factory=set)
    temp_paths: set[Path] = field(default_factory=set)
    worker: threading.Thread | None = None
    closed: bool = False
    remove_cache_on_close: bool = False
    resources_released: bool = False
    created_at: float = field(default_factory=time.time)


application = FastAPI(title="进点双子星 V4 Local API")
_sessions: dict[str, SessionRecord] = {}
_sessions_lock = threading.RLock()
_model_lock = threading.Lock()
_model_state: dict[str, Any] = {
    "status": "idle",
    "message": "等待启动模型",
    "elapsed": 0.0,
}


def _get_session(session_id: str) -> SessionRecord:
    with _sessions_lock:
        record = _sessions.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="会话不存在或已结束")
    return record


def _register_session(record: SessionRecord) -> str:
    session_id = str(record.state["session_id"])
    evicted: SessionRecord | None = None
    with _sessions_lock:
        if len(_sessions) >= SETTINGS.max_sessions:
            oldest = min(_sessions, key=lambda key: _sessions[key].created_at)
            evicted = _sessions.pop(oldest)
        _sessions[session_id] = record
    if evicted is not None:
        _close_record(evicted, remove_cache=True)
    return session_id


def _close_record(record: SessionRecord, *, remove_cache: bool) -> None:
    """Stop one session and release native/thread/filesystem resources once."""
    with record.lock:
        if record.closed:
            return
        record.closed = True
        record.remove_cache_on_close = remove_cache
        if record.mode == "file":
            pipeline.stop_uploaded_stream(record.state)
        worker = record.worker
    if (
        worker is not None
        and worker.is_alive()
        and worker is not threading.current_thread()
    ):
        worker.join(timeout=SETTINGS.worker_join_seconds)
    if worker is not None and worker.is_alive():
        logger.warning(
            "Session worker still stopping; deferred cleanup: %s",
            record.state.get("session_id"),
        )
        return
    _release_record_resources(record, remove_cache=remove_cache)


def _release_record_resources(record: SessionRecord, *, remove_cache: bool) -> None:
    with record.lock:
        if record.resources_released:
            return
        record.resources_released = True
        temp_paths = list(record.temp_paths)
        record.temp_paths.clear()
    pipeline.release_live_session_resources(
        record.state,
        remove_cache=remove_cache,
    )
    if remove_cache:
        for path in temp_paths:
            existed = path.exists()
            if existed and not remove_runtime_path(SETTINGS, path):
                logger.warning("Unable to remove session temp path: %s", path)


def _remove_upload_temp_paths(record: SessionRecord) -> None:
    with record.lock:
        temp_paths = list(record.temp_paths)
        record.temp_paths.clear()
    for path in temp_paths:
        existed = path.exists()
        if existed and not remove_runtime_path(SETTINGS, path):
            logger.warning("Unable to remove upload temp path: %s", path)


def _extract_audio_path(value: object) -> str | None:
    if isinstance(value, dict):
        candidate = value.get("value")
        return str(candidate) if candidate else None
    return None


def _record_audio_event(record: SessionRecord, path_value: str | None) -> None:
    if not path_value:
        return
    resolved = str(Path(path_value).resolve())
    segment_id: int | None = None
    for key, metadata in record.state.get("tts_paths", {}).items():
        candidate = metadata.get("path")
        if candidate and str(Path(candidate).resolve()) == resolved:
            segment_id = int(key)
            break
    if segment_id is None or segment_id in record.emitted_audio_ids:
        return
    record.emitted_audio_ids.add(segment_id)
    record.last_audio_path = resolved
    record.audio_events.append(
        {
            "segment_id": segment_id,
            "url": f"/api/sessions/{record.state['session_id']}/audio/{segment_id}",
        }
    )


def _apply_live_outputs(record: SessionRecord, outputs: tuple[object, ...]) -> None:
    if not outputs:
        return
    record.state = outputs[0]  # type: ignore[assignment]
    if len(outputs) > 4:
        _record_audio_event(record, _extract_audio_path(outputs[4]))
    if len(outputs) > 6 and isinstance(outputs[6], str):
        record.status = outputs[6]
    if len(outputs) > 7 and isinstance(outputs[7], str):
        record.pipeline_status = outputs[7]


def _snapshot(record: SessionRecord, *, drain_audio: bool = True) -> dict[str, Any]:
    state = record.state
    asr_state = state.get("asr_state", {})
    rows = []
    for time_label, chinese, english, segment_id in pipeline._live_translation_rows(state):
        segment_key = str(int(segment_id))
        rows.append(
            {
                "time": str(time_label),
                "chinese": str(chinese),
                "english": str(english),
                "segment_id": int(segment_id),
                "has_audio": segment_key in state.get("tts_paths", {}),
            }
        )
    window_records = asr_state.get("window_records", [])
    latency = float(window_records[-1].get("window_latency", 0.0)) if window_records else 0.0
    stable_count = sum(
        1 for item in asr_state.get("segments", []) if item.get("status") == "FINAL"
    )
    total_seconds = float(asr_state.get("total_samples", 0)) / SETTINGS.sample_rate
    vad_state = str(asr_state.get("vad_state", "IDLE"))
    input_state = {
        "IDLE": "等待讲话",
        "SPEECH": "讲话中",
        "SILENCE_PENDING": "检测停顿",
        "ONLINE_SPEECH": "流式识别",
    }.get(vad_state, vad_state)
    events = list(record.audio_events)
    if drain_audio:
        record.audio_events.clear()
    return {
        "session_id": str(state["session_id"]),
        "mode": record.mode,
        "status": record.status,
        "pipeline": record.pipeline_status,
        "model_status": dict(_model_state),
        "input_seconds": round(total_seconds, 2),
        "input_state": input_state,
        "latency": round(latency, 3),
        "stable_count": stable_count,
        "queue_size": len(state.get("audio_queue", [])),
        "queue_max": int(state.get("max_audio_queue", SETTINGS.max_audio_queue)),
        "current_segment_id": state.get("current_segment_id"),
        "played_segment_ids": list(state.get("played_segment_ids", [])),
        "rows": rows,
        "audio_events": events,
        "last_error": str(state.get("last_error", "")),
        "finalization_status": str(state.get("finalization_status", "IDLE")),
        "finalization_message": str(state.get("finalization_message", "")),
        "file_status": record.file_status,
    }


def _warm_models_sync() -> dict[str, Any]:
    with _model_lock:
        if _model_state["status"] == "ready":
            return dict(_model_state)
        started = time.perf_counter()
        checks = run_startup_checks(SETTINGS, ui_root=UI_ROOT)
        _model_state["checks"] = checks
        if not checks_ok(checks):
            missing = ", ".join(
                item["name"] for item in checks if item["status"] == "error"
            )
            _model_state.update(
                status="error",
                message=f"启动自检失败：{missing}",
                elapsed=round(time.perf_counter() - started, 2),
            )
            logger.error("Model warmup blocked by startup checks: %s", missing)
            return dict(_model_state)
        _model_state.update(status="loading", message="正在启动中文流式识别")
        try:
            pipeline.asr.load_live()
            _model_state["message"] = "正在启动中英翻译"
            pipeline.translator.load()
            _model_state["message"] = "正在启动英文语音合成"
            pipeline.tts.load()
            elapsed = time.perf_counter() - started
            _model_state.update(
                status="ready",
                message="全部模型已就绪",
                elapsed=round(elapsed, 2),
                asr=pipeline.asr.live_backend,
                translator=pipeline.translator.__class__.__name__,
                tts="PiperTTS",
            )
        except Exception as exc:
            _model_state.update(
                status="error",
                message=f"模型启动失败：{exc}",
                elapsed=round(time.perf_counter() - started, 2),
            )
            logger.exception("Model warmup failed")
        return dict(_model_state)


def _process_chunk_sync(record: SessionRecord, samples: np.ndarray) -> dict[str, Any]:
    with record.lock:
        outputs = pipeline.process_live_chunk((16000, samples), record.state)
        _apply_live_outputs(record, outputs)
        return _snapshot(record)


def _start_finalization_sync(record: SessionRecord) -> dict[str, Any]:
    with record.lock:
        outputs = pipeline.start_microphone_finalization(record.state)
        _apply_live_outputs(record, outputs)
        return _snapshot(record)


def _poll_finalization_sync(record: SessionRecord) -> dict[str, Any]:
    with record.lock:
        # Final ASR/translation can finish before the last TTS items have been
        # handed to the browser.  Keep draining that queue after DONE so the
        # final sentence is not stranded without an autoplay event.
        if (
            record.state.get("finalization_status") == "RUNNING"
            or record.state.get("audio_queue")
        ):
            outputs = pipeline.poll_microphone_finalization(record.state)
            _apply_live_outputs(record, outputs)
        return _snapshot(record)


def _apply_file_yield(record: SessionRecord, values: tuple[object, ...]) -> None:
    if len(values) < 10:
        return
    record.state = values[1]  # type: ignore[assignment]
    _record_audio_event(record, _extract_audio_path(values[5]))
    if isinstance(values[7], str):
        record.status = values[7]
    if isinstance(values[8], str):
        record.pipeline_status = values[8]


def _file_worker(record: SessionRecord, generator: Any) -> None:
    record.file_status = "RUNNING"
    try:
        for values in generator:
            with record.lock:
                _apply_file_yield(record, values)
        record.file_status = "DONE"
    except Exception as exc:
        with record.lock:
            record.file_status = "ERROR"
            record.state["last_error"] = f"文件流式处理失败：{exc}"
            record.status = "文件流式处理已停止"
        logger.exception("File session worker failed: %s", record.state["session_id"])
    finally:
        _remove_upload_temp_paths(record)
        if record.closed:
            _release_record_resources(
                record,
                remove_cache=record.remove_cache_on_close,
            )


@application.get("/api/health")
async def health() -> dict[str, Any]:
    checks = run_startup_checks(SETTINGS, ui_root=UI_ROOT)
    with _sessions_lock:
        active_sessions = len(_sessions)
    return {
        "status": "ok" if checks_ok(checks) else "degraded",
        "ui": "v4-connected",
        "models": dict(_model_state),
        "active_sessions": active_sessions,
        "checks": checks,
    }


@application.exception_handler(SimulAgentError)
async def handle_pipeline_error(_request: Request, exc: SimulAgentError) -> JSONResponse:
    logger.error("Pipeline error [%s/%s]: %s", exc.stage, exc.code, exc)
    return JSONResponse(status_code=500, content={"detail": exc.as_dict()})


@application.post("/api/models/warmup")
async def warmup_models() -> dict[str, Any]:
    result = await asyncio.to_thread(_warm_models_sync)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result


@application.post("/api/sessions")
async def create_microphone_session() -> dict[str, Any]:
    if _model_state["status"] != "ready":
        raise HTTPException(status_code=409, detail="请先启动模型")
    state = pipeline._new_live_state("microphone")
    record = SessionRecord(
        state=state,
        mode="microphone",
        status="麦克风会话已建立，等待音频",
        pipeline_status="Sherpa Zipformer → LocalAgreement → Qwen 翻译 → Piper TTS",
    )
    _register_session(record)
    return _snapshot(record)


@application.post("/api/sessions/{session_id}/chunks")
async def ingest_microphone_chunk(session_id: str, request: Request) -> dict[str, Any]:
    record = _get_session(session_id)
    if record.mode != "microphone":
        raise HTTPException(status_code=409, detail="当前会话不是麦克风模式")
    body = await request.body()
    if not body:
        return _snapshot(record)
    if len(body) > SETTINGS.max_chunk_bytes:
        raise HTTPException(status_code=413, detail="音频块过大")
    samples_int16 = np.frombuffer(body, dtype="<i2")
    samples = samples_int16.astype(np.float32) / 32768.0
    return await asyncio.to_thread(_process_chunk_sync, record, samples)


@application.post("/api/sessions/{session_id}/finalize")
async def finalize_microphone_session(session_id: str) -> dict[str, Any]:
    record = _get_session(session_id)
    if record.mode != "microphone":
        raise HTTPException(status_code=409, detail="当前会话不是麦克风模式")
    return await asyncio.to_thread(_start_finalization_sync, record)


@application.get("/api/sessions/{session_id}")
async def session_status(session_id: str) -> dict[str, Any]:
    record = _get_session(session_id)
    if record.mode == "microphone":
        return await asyncio.to_thread(_poll_finalization_sync, record)
    with record.lock:
        return _snapshot(record)


@application.post("/api/sessions/{session_id}/stop")
async def stop_session(session_id: str) -> dict[str, Any]:
    record = _get_session(session_id)
    if record.mode == "file":
        pipeline.stop_uploaded_stream(record.state)
        record.status = "已请求停止文件流式处理"
        return _snapshot(record)
    return await asyncio.to_thread(_start_finalization_sync, record)


@application.post("/api/file-sessions")
async def create_file_session(file: UploadFile = File(...)) -> dict[str, Any]:
    if _model_state["status"] != "ready":
        raise HTTPException(status_code=409, detail="请先启动模型")
    filename = Path(file.filename or "speech.wav").name
    if Path(filename).suffix.lower() != ".wav":
        raise HTTPException(status_code=415, detail="目前只支持 WAV 音频")
    upload_dir = UPLOAD_ROOT / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / filename
    try:
        written = 0
        with destination.open("wb") as output:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > SETTINGS.max_file_upload_bytes:
                    raise HTTPException(status_code=413, detail="上传音频过大")
                output.write(chunk)
    except Exception:
        remove_runtime_path(SETTINGS, upload_dir)
        raise
    finally:
        await file.close()

    generator = pipeline.stream_uploaded_audio(str(destination))
    try:
        first_values = next(generator)
    except StopIteration as exc:
        remove_runtime_path(SETTINGS, upload_dir)
        raise HTTPException(status_code=500, detail="文件流式处理未能启动") from exc
    state = first_values[1]
    record = SessionRecord(
        state=state,
        mode="file",
        status="正在准备文件流式处理",
        pipeline_status="源音频同步播放 → 0.4 秒音频块 → 本地同传链路",
        temp_paths={upload_dir},
    )
    with record.lock:
        _apply_file_yield(record, first_values)
    _register_session(record)
    worker = threading.Thread(
        target=_file_worker,
        args=(record, generator),
        name=f"v4-file-{state['session_id']}",
        daemon=True,
    )
    record.worker = worker
    worker.start()
    return _snapshot(record)


@application.get("/api/sessions/{session_id}/audio/{segment_id}")
async def segment_audio(session_id: str, segment_id: int) -> FileResponse:
    record = _get_session(session_id)
    with record.lock:
        metadata = record.state.get("tts_paths", {}).get(str(segment_id), {})
        candidate = Path(str(metadata.get("path", "")))
    if not candidate.exists() or candidate.suffix.lower() != ".wav":
        raise HTTPException(status_code=404, detail="该片段的英文语音尚未生成")
    return FileResponse(candidate, media_type="audio/wav", filename=candidate.name)


@application.get("/api/results")
async def list_results() -> dict[str, Any]:
    output_dir = pipeline.LIVE_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        (path for path in output_dir.iterdir() if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return {
        "files": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "modified": path.stat().st_mtime,
            }
            for path in files[: SETTINGS.result_list_limit]
        ]
    }


@application.get("/api/system")
async def system_status() -> dict[str, Any]:
    rows = await asyncio.to_thread(pipeline._system_status_rows)
    return {"models": dict(_model_state), "rows": rows}


@application.on_event("shutdown")
def shutdown_sessions() -> None:
    with _sessions_lock:
        records = list(_sessions.values())
        _sessions.clear()
    for record in records:
        _close_record(record, remove_cache=True)


application.mount("/", StaticFiles(directory=UI_ROOT, html=True), name="v4-ui")


if __name__ == "__main__":
    port = int(os.environ.get("V4_WEB_PORT", "7861"))
    if os.environ.get("INBROWSER", "1") == "1":
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(application, host="127.0.0.1", port=port, log_level="info")
