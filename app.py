from __future__ import annotations

import html
import json
import os
import queue
import re
import statistics
import threading
import time
import traceback
import uuid
import wave
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from src.app_config import RuntimeSettings
from src.contracts import audio_queue_item, translation_record, tts_record
from src.file_pipeline import _combine_wavs
from src.realtime_input import iter_audio_blocks, load_audio_16k
from src.runtime_support import (
    SimulAgentError,
    configure_logging,
    remove_runtime_path,
    session_directory,
)
from src.stream_asr import StreamingASR
from src.text_stabilizer import OpenCC, TextStabilizer
from src.translation import build_translator, validate_translation
from src.tts import PiperTTS
from src.ui_v3 import APP_CSS as V3_APP_CSS, APP_JS as V3_APP_JS


PROJECT_ROOT = Path(__file__).resolve().parent
SETTINGS = RuntimeSettings.from_env(PROJECT_ROOT)
OUTPUT_ROOT = SETTINGS.output_root
LIVE_OUTPUT_DIR = SETTINGS.live_output_dir
# Native streaming Zipformer is shared by microphone input and uploaded WAV
# playback so both demo paths exercise the same incremental state machine.
os.environ.setdefault("ASR_LIVE_BACKEND", "sherpa")
os.environ.setdefault(
    "ASR_LIVE_MODEL_DIR",
    str(SETTINGS.cache_root / "models" / "funasr-paraformer-zh-streaming"),
)
os.environ.setdefault(
    "SHERPA_MODEL_DIR",
    str(
        SETTINGS.cache_root
        / "models"
        / "sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30"
    ),
)
logger = configure_logging(SETTINGS)
asr = StreamingASR()
translator = build_translator()
tts = PiperTTS()
_TRANSPORT_LOCK = threading.RLock()
_FILE_TRANSPORTS: dict[str, dict[str, threading.Event]] = {}
_MODEL_PIPELINE_LOCK = threading.RLock()
_FINALIZATION_LOCK = threading.RLock()
_FINALIZATION_THREADS: dict[str, threading.Thread] = {}
# Application-level translation attempts.  QwenTranslator may perform its own
# constrained retry internally; this outer limit prevents a failed segment
# from blocking the stream indefinitely while still leaving one later pass
# available for transient failures.
MAX_TRANSLATION_ATTEMPTS = 3


def _degraded_translation_text(source: str) -> str:
    """Return a deterministic English-only last-resort translation.

    This path is deliberately conservative and is used only after the local
    translator has exhausted its bounded retries.  It keeps the output stream
    renderable and speakable without ever sending Chinese text to the English
    TTS voice.  Arabic figures are retained when the recognizer produced them.
    """
    numbers = re.findall(r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)*(?:%)?", str(source))
    unique_numbers: list[str] = []
    for value in numbers:
        if value not in unique_numbers:
            unique_numbers.append(value)
    if unique_numbers:
        return f"The speaker mentions {', '.join(unique_numbers)}."
    return "The speaker continues with this point."


def _register_file_transport(session_id: str) -> dict[str, threading.Event]:
    control = {
        "paused": threading.Event(),
        "stopped": threading.Event(),
    }
    with _TRANSPORT_LOCK:
        _FILE_TRANSPORTS[session_id] = control
    return control


def _file_transport(session_id: str) -> dict[str, threading.Event] | None:
    with _TRANSPORT_LOCK:
        return _FILE_TRANSPORTS.get(session_id)


def _release_file_transport(session_id: str) -> None:
    with _TRANSPORT_LOCK:
        _FILE_TRANSPORTS.pop(session_id, None)


def pause_uploaded_stream(state: dict | None) -> None:
    if not state:
        return
    control = _file_transport(str(state.get("session_id", "")))
    if control:
        control["paused"].set()


def resume_uploaded_stream(state: dict | None) -> None:
    if not state:
        return
    control = _file_transport(str(state.get("session_id", "")))
    if control:
        control["paused"].clear()


def stop_uploaded_stream(state: dict | None) -> None:
    if not state:
        return
    control = _file_transport(str(state.get("session_id", "")))
    if control:
        control["stopped"].set()
        control["paused"].clear()


def release_live_session_resources(
    state: dict | None,
    *,
    remove_cache: bool = False,
) -> None:
    """Release native recognizer and transport objects for an abandoned session."""
    if not state:
        return
    session_id = str(state.get("session_id", ""))
    stop_uploaded_stream(state)
    _release_file_transport(session_id)
    asr.release_incremental_state(state.get("asr_state"))
    if remove_cache and session_id:
        for category in ("live_sources", "live_tts"):
            try:
                remove_runtime_path(
                    SETTINGS,
                    session_directory(SETTINGS, category, session_id),
                )
            except ValueError:
                logger.warning("Refused invalid session cache path: %s", session_id)


def _format_segments(items: list[dict], text_key: str = "text") -> str:
    return "\n".join(
        (
            f"[{item['start']:>7.2f}s - {item['end']:>7.2f}s] {item[text_key]}"
            + (
                " 〔可疑候选，仅展示且未翻译〕"
                if item.get("status") == "REJECTED"
                else ""
            )
        )
        for item in items
    )


def _audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio_file:
        return audio_file.getnframes() / audio_file.getframerate()


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * 0.95) - 1)]


def _resolve_result_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path if path.exists() else None


def _save_live_results(
    source_audio: str,
    stable_segments: list[dict],
    translated_by_id: dict[int, tuple[str, str]],
    audio_by_id: dict[int, str],
    window_records: list[dict],
    pipeline_diagnostics: dict | None = None,
) -> dict:
    """Save one complete text/WAV/metrics result set for the status page."""
    if not stable_segments:
        raise ValueError("没有可保存的稳定片段")

    LIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(source_audio).stem
    translation_path = LIVE_OUTPUT_DIR / f"{stem}_translation.txt"
    tts_path = LIVE_OUTPUT_DIR / f"{stem}_translation_tts.wav"
    metrics_path = LIVE_OUTPUT_DIR / f"{stem}_pipeline_metrics.json"

    translations = [
        translated_by_id[item["segment_id"]][1]
        for item in stable_segments
        if item["segment_id"] in translated_by_id
    ]
    if len(translations) != len(stable_segments):
        raise SimulAgentError(
            "稳定片段与英文翻译数量不一致",
            code="EXPORT_TRANSLATION_ALIGNMENT",
            stage="export",
        )
    translation_path.write_text(
        "\n".join(translations) + "\n",
        encoding="utf-8",
    )

    segment_wavs = [
        Path(audio_by_id[item["segment_id"]])
        for item in stable_segments
        if item["segment_id"] in audio_by_id
    ]
    if len(segment_wavs) != len(stable_segments):
        raise SimulAgentError(
            "稳定片段与英文语音数量不一致",
            code="EXPORT_AUDIO_ALIGNMENT",
            stage="export",
        )
    _combine_wavs(segment_wavs, tts_path)

    latencies = [float(item["window_latency"]) for item in window_records]
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_audio": str(Path(source_audio).name),
        "translation_file": str(translation_path.relative_to(PROJECT_ROOT)),
        "tts_file": str(tts_path.relative_to(PROJECT_ROOT)),
        "stable_segments": len(stable_segments),
        "windows": window_records,
        "average_window_latency": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "p95_window_latency": round(_p95(latencies), 3),
        "translation_backend": translator.__class__.__name__,
        "translation_device": translator.device,
        "translation_load_seconds": translator.load_seconds,
        "tts_length_scale": tts.length_scale,
        "tts_duration_seconds": round(_audio_duration(tts_path), 2),
        "asr_backend": asr.live_backend,
        "degraded_translation_ids": list(
            (pipeline_diagnostics or {}).get("degraded_translation_ids", [])
        ),
        "audio_quality": (pipeline_diagnostics or {}).get("audio_quality", {}),
        "final_commit_policy": (pipeline_diagnostics or {}).get(
            "commit_policy", {}
        ),
    }
    metrics_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata["metrics_file"] = str(metrics_path.relative_to(PROJECT_ROOT))
    return metadata


def _system_status_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            rows.append(
                [
                    "CUDA / GPU",
                    "正常",
                    (
                        f"{torch.cuda.get_device_name(0)} | "
                        f"显存 {used / 2**30:.1f}/{total / 2**30:.1f} GB"
                    ),
                ]
            )
        else:
            rows.append(["CUDA / GPU", "不可用", "将无法运行默认 GPU 推理"])
    except Exception as exc:
        rows.append(["CUDA / GPU", "检查失败", str(exc)])

    model_checks = [
        (
            "实时中文 ASR",
            asr.sherpa_model_dir / "encoder.int8.onnx",
            (
                "Sherpa-ONNX Streaming Zipformer + LocalAgreement"
                f" | {'已加载' if asr.live_backend == 'sherpa' and asr._live_model is not None else '懒加载'}"
            ),
        ),
        (
            "上传音频流式 ASR",
            asr.sherpa_model_dir / "encoder.int8.onnx",
            (
                "与麦克风共用 Sherpa-ONNX Streaming Zipformer"
                f" | {'已加载' if asr.live_backend == 'sherpa' and asr._live_model is not None else '懒加载'}"
            ),
        ),
        (
            "中译英",
            translator.model_dir,
            f"{translator.__class__.__name__} | {'已加载' if translator._model is not None else '懒加载'}",
        ),
        (
            "英文 TTS",
            tts.voice_model,
            f"Piper en_US-lessac-medium | {'已加载' if tts._voice is not None else '懒加载'}",
        ),
    ]
    for name, path, description in model_checks:
        rows.append(
            [
                name,
                "正常" if path.exists() else "缺失",
                f"{description} | {path}",
            ]
        )
    rows.append(
        [
            "文本稳定器",
            "正常",
            f"OpenCC {'已安装' if OpenCC is not None else '未安装'} | 纠错词表与重叠去重",
        ]
    )
    rows.append(
        [
            "结果目录",
            "正常" if OUTPUT_ROOT.exists() else "尚未生成",
            str(OUTPUT_ROOT),
        ]
    )
    return rows


def refresh_results_page():
    """Return live system health and the newest generated result artifacts."""
    system_rows = _system_status_rows()
    candidates = sorted(
        OUTPUT_ROOT.rglob("*_pipeline_metrics.json")
        if OUTPUT_ROOT.exists()
        else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return (
            system_rows,
            "尚无完整运行结果。请先在“实时工作台”运行一条音频。",
            [],
            None,
            None,
            None,
            None,
        )

    metrics_path = candidates[0]
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    translation_path = _resolve_result_path(metadata.get("translation_file"))
    tts_path = _resolve_result_path(metadata.get("tts_file"))
    windows = metadata.get("windows", [])
    latencies = [
        float(item["window_latency"])
        for item in windows
        if "window_latency" in item
    ]
    text_lines = 0
    if translation_path:
        text_lines = sum(
            bool(line.strip())
            for line in translation_path.read_text(encoding="utf-8").splitlines()
        )
    tts_duration = _audio_duration(tts_path) if tts_path else 0.0
    result_rows = [
        ["生成时间", str(metadata.get("created_at", datetime.fromtimestamp(metrics_path.stat().st_mtime).isoformat(timespec="seconds")))],
        ["翻译后端", str(metadata.get("translation_backend", translator.__class__.__name__))],
        ["稳定片段", str(metadata.get("stable_segments", text_lines))],
        ["音频窗口", str(len(windows))],
        ["平均窗口耗时", f"{statistics.mean(latencies):.3f} 秒" if latencies else "无数据"],
        ["P95窗口耗时", f"{_p95(latencies):.3f} 秒" if latencies else "无数据"],
        ["英文文本行数", str(text_lines)],
        ["合成语音时长", f"{tts_duration:.1f} 秒" if tts_path else "文件缺失"],
    ]
    summary = (
        f"最新结果：{translation_path.name if translation_path else '文本缺失'} | "
        f"{tts_path.name if tts_path else '音频缺失'}"
    )
    return (
        system_rows,
        summary,
        result_rows,
        str(translation_path) if translation_path else None,
        str(tts_path) if tts_path else None,
        str(metrics_path),
        str(tts_path) if tts_path else None,
    )


def run_streaming_asr(audio_path: str | None):
    """Real local ASR path. The model is loaded only after the user clicks."""
    if not audio_path:
        yield "请先上传 WAV 音频文件。", "", "", None, "等待音频输入", [], ""
        return

    stabilizer = TextStabilizer()
    if hasattr(translator, "reset_context"):
        translator.reset_context()
    raw_segments: list[dict] = []
    stable_segments: list[dict] = []
    translated_by_id: dict[int, tuple[str, str]] = {}
    audio_by_id: dict[int, str] = {}
    metrics: list[list[str]] = []
    window_records: list[dict] = []
    tts_dir = PROJECT_ROOT / ".cache" / "runtime" / "tts"

    try:
        yield "正在加载本地 ASR 模型……", "", "", None, "初始化中", [], ""
        for result in asr.transcribe_stream(audio_path):
            raw_segments.extend(result["items"])
            for item in result["items"]:
                stabilizer.feed(item["text"], item["start"], item["end"])
            stable_segments = stabilizer.flush()
            changed = [
                item
                for item in stable_segments
                if item["segment_id"] not in translated_by_id
                or translated_by_id[item["segment_id"]][0] != item["text"]
            ]
            translation_error = ""
            tts_error = ""
            if changed:
                try:
                    translations = translator.translate_many([item["text"] for item in changed])
                    for item, translated in zip(changed, translations):
                        translated_by_id[item["segment_id"]] = (item["text"], translated)
                except Exception as exc:  # keep ASR visible if MT is unavailable
                    translation_error = str(exc)
                    for item in changed:
                        translated_by_id[item["segment_id"]] = (
                            item["text"],
                            f"[翻译暂不可用] {item['text']}",
                        )
            for item in changed:
                translated = translated_by_id[item["segment_id"]][1]
                if translated.startswith("[翻译暂不可用]"):
                    continue
                try:
                    tts_dir.mkdir(parents=True, exist_ok=True)
                    segment_audio_path = tts_dir / f"segment_{item['segment_id']:04d}.wav"
                    tts.synthesize(translated, segment_audio_path)
                    audio_by_id[item["segment_id"]] = str(segment_audio_path)
                except Exception as exc:  # keep text outputs if TTS is unavailable
                    tts_error = str(exc)
            translated_segments = [
                {
                    **item,
                    "translation": translated_by_id[item["segment_id"]][1],
                }
                for item in stable_segments
                if item["segment_id"] in translated_by_id
            ]
            metrics.append(
                [
                    f"{result['window_start']:.1f}-{result['window_end']:.1f}s",
                    f"{result['window_latency']:.2f}s",
                    str(len(result["items"])),
                    str(len(stable_segments)),
                ]
            )
            window_records.append(
                {
                    "window_start": result["window_start"],
                    "window_end": result["window_end"],
                    "window_latency": result["window_latency"],
                    "segment_count": len(result["items"]),
                    "stable_count": len(stable_segments),
                }
            )
            status = (
                f"已处理 {result['window_end']:.1f}/{result['total_seconds']:.1f} 秒 | "
                f"本窗口耗时 {result['window_latency']:.2f} 秒"
            )
            if translation_error:
                pipeline = (
                    f"ASR 已完成，{translator.__class__.__name__} 异常："
                    f"{translation_error}"
                )
            elif tts_error:
                pipeline = (
                    f"ASR + 稳定器 + {translator.__class__.__name__} 已运行"
                    f"（{translator.device}）；"
                    f"Piper TTS 异常：{tts_error}"
                )
            else:
                pipeline = (
                    f"ASR + 稳定器 + {translator.__class__.__name__} + Piper TTS 已运行"
                    f"（MT {translator.device}）"
                )
            latest_audio = audio_by_id.get(stable_segments[-1]["segment_id"]) if stable_segments else None
            yield (
                _format_segments(raw_segments),
                _format_segments(stable_segments),
                _format_segments(translated_segments, "translation"),
                latest_audio,
                status,
                metrics,
                pipeline,
            )

        translated_segments = [
            {
                **item,
                "translation": translated_by_id[item["segment_id"]][1],
            }
            for item in stable_segments
            if item["segment_id"] in translated_by_id
        ]
        latest_audio = audio_by_id.get(stable_segments[-1]["segment_id"]) if stable_segments else None
        try:
            saved = _save_live_results(
                audio_path,
                stable_segments,
                translated_by_id,
                audio_by_id,
                window_records,
            )
            final_pipeline = (
                f"真实翻译和英文语音完成，共 {len(translated_segments)} 个片段；"
                f"完整结果已保存到 {LIVE_OUTPUT_DIR}"
            )
        except Exception as exc:
            saved = None
            final_pipeline = (
                f"推理完成，但完整结果保存失败：{exc}。"
                "片段文本和最新语音仍可查看。"
            )
        yield (
            _format_segments(raw_segments),
            _format_segments(stable_segments),
            _format_segments(translated_segments, "translation"),
            latest_audio,
            f"识别完成，共 {len(stable_segments)} 个稳定片段",
            metrics,
            final_pipeline,
        )
    except Exception as exc:  # pragma: no cover - displayed in the UI
        yield f"识别失败：{exc}", "", "", None, "错误", metrics, "流水线已安全停止"


def _safe_source_stem(source_name: str | None, fallback: str) -> str:
    stem = Path(source_name or "").stem.strip()
    safe = "".join(
        character
        for character in stem
        if character.isalnum() or character in {"-", "_"}
    )
    return safe or fallback


def _new_live_state(
    input_mode: str = "microphone",
    source_name: str | None = None,
) -> dict:
    """Create one browser-session state shared by microphone and file input."""
    mode_prefix = "file" if input_mode == "file" else "mic"
    session_id = (
        f"{mode_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    source_stem = _safe_source_stem(source_name, session_id)
    source_path = session_directory(
        SETTINGS,
        "live_sources",
        session_id,
    ) / f"{source_stem}.wav"
    return {
        "session_id": session_id,
        "source_audio": str(source_path),
        "source_name": source_name or f"{session_id}.wav",
        "input_mode": input_mode,
        "input_label": "上传音频" if input_mode == "file" else "麦克风",
        "asr_state": asr.new_incremental_state(),
        "translations": {},
        "translation_sources": {},
        "translation_attempts": {},
        "translation_retry_segments": {},
        "tts_paths": {},
        "audio_queue": [],
        "audio_cursor": 0.0,
        "current_segment_id": None,
        "played_segment_ids": [],
        "dropped_audio_ids": [],
        "audio_queue_high_watermark": 0,
        "failed_translation_ids": [],
        "degraded_translation_ids": [],
        "translation_degradation_reasons": {},
        "translation_jobs_pending": 0,
        "max_audio_queue": SETTINGS.max_audio_queue,
        "last_error": "",
        "finalization_status": "IDLE",
        "finalization_message": "",
        "input_closed": False,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }


def _append_live_source(state: dict, audio: tuple[int, np.ndarray] | None) -> None:
    if audio is None:
        return
    _, data = audio
    if data.size == 0:
        return
    path = Path(state["source_audio"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        sf.write(str(path), data, SETTINGS.sample_rate, subtype="PCM_16")
        return
    with sf.SoundFile(str(path), mode="r+") as source_file:
        source_file.seek(0, 2)
        source_file.write(data)


def _update_asr_backpressure(state: dict) -> None:
    """Expose TTS queue pressure to the adaptive ASR commit policy."""
    maximum = max(1, int(state.get("max_audio_queue", 4)))
    state["asr_state"]["tts_queue_pressure"] = min(
        1.0,
        len(state.get("audio_queue", [])) / maximum,
    )


def _render_live_subtitles(state: dict) -> str:
    segments = [
        item
        for item in state.get("asr_state", {}).get("segments", [])
        if item.get("status") != "REJECTED"
    ]
    translations = state.get("translations", {})
    if not segments:
        return '<div class="live-subtitles-empty">等待语音输入……</div>'
    queue_ids = {
        int(item["segment_id"])
        for item in state.get("audio_queue", [])
    }
    played_ids = {int(item) for item in state.get("played_segment_ids", [])}
    current_id = state.get("current_segment_id")
    parts = ['<div class="live-subtitle-list">']
    for item in segments:
        segment_id = int(item["segment_id"])
        translation = translations.get(str(segment_id), {}).get("text", "翻译中……")
        audio_meta = state.get("tts_paths", {}).get(str(segment_id), {})
        audio_start = audio_meta.get("audio_start", 0.0)
        audio_end = audio_meta.get("audio_end", 0.0)
        if segment_id == current_id:
            status = "playing"
        elif segment_id in queue_ids:
            status = "queued"
        elif segment_id in played_ids:
            status = "played"
        else:
            status = "pending"
        if item.get("status") == "PARTIAL":
            status += " partial"
        parts.append(
            (
                f'<div class="live-subtitle-row {status}" '
                f'data-segment-id="{segment_id}" '
                f'data-audio-start="{audio_start:.3f}" '
                f'data-audio-end="{audio_end:.3f}">'
                f'<span class="live-subtitle-time">'
                f'{float(item["start"]):.2f}s</span>'
                f'<span class="live-subtitle-en">{html.escape(translation)}</span>'
                f'</div>'
            )
        )
    parts.append("</div>")
    return "".join(parts)


def _live_translation_rows(state: dict) -> list[list[object]]:
    """Return paired Chinese/English units as selectable playback rows."""
    stable_segments = [
        item
        for item in state.get("asr_state", {}).get("segments", [])
        if item.get("status") == "FINAL"
    ]
    translations = state.get("translations", {})
    return [
        [
            f"{float(item['start']):.2f}s",
            item["text"],
            translations.get(str(item["segment_id"]), {}).get("text", "翻译中……"),
            int(item["segment_id"]),
        ]
        for item in stable_segments[-24:]
    ]


def _render_live_translation(state: dict) -> str:
    """Render the visible bilingual units without relying on Gradio's grid DOM."""
    rows = _live_translation_rows(state)
    if not rows:
        return (
            '<div class="v3-unit-empty">'
            '<strong>等待开始同声传译</strong>'
            '<span>稳定的中文片段和对应英文会按时间依次出现在这里。</span>'
            '</div>'
        )

    current_id = state.get("current_segment_id")
    played_ids = {int(item) for item in state.get("played_segment_ids", [])}
    tts_paths = state.get("tts_paths", {})
    parts = ['<div class="v3-unit-list" role="table" aria-label="实时中英文语义对照">']
    parts.append(
        '<div class="v3-unit-head" role="row">'
        '<span>时间</span><span>中文 · 已确认</span>'
        '<span>English · 点击播放</span><span>片段</span></div>'
    )
    for row in rows:
        time_label, chinese, english, segment_id_value = row
        segment_id = int(segment_id_value)
        state_class = ""
        if segment_id == current_id:
            state_class = " is-playing"
        elif segment_id in played_ids:
            state_class = " is-played"
        has_audio = str(segment_id) in tts_paths
        audio_class = " has-audio" if has_audio else " is-pending"
        button_label = "播放此英文片段" if has_audio else "英文语音生成中"
        parts.append(
            f'<div class="v3-unit-row{state_class}{audio_class}" role="row" '
            f'data-segment-id="{segment_id}">'
            f'<time role="cell">{html.escape(str(time_label))}</time>'
            f'<p role="cell" lang="zh-CN">{html.escape(str(chinese))}</p>'
            f'<button type="button" class="v3-unit-play" role="cell" '
            f'data-segment-id="{segment_id}" aria-label="{button_label}">'
            f'<span lang="en">{html.escape(str(english))}</span><i>▶</i></button>'
            f'<span class="v3-unit-id" role="cell">U{segment_id:02d}</span>'
            '</div>'
        )
    parts.append('</div>')
    return "".join(parts)


def play_translation_segment(state: dict | None, segment_value: str | int | float | None):
    """Load one TTS WAV selected from the custom bilingual unit list."""
    if not state:
        return gr.skip()
    try:
        segment_id = str(int(float(str(segment_value).strip())))
    except (TypeError, ValueError):
        return gr.skip()
    audio_meta = state.get("tts_paths", {}).get(segment_id, {})
    path = Path(audio_meta.get("path", ""))
    return gr.update(value=str(path), visible=True) if path.exists() else gr.skip()


def _live_metrics(state: dict) -> list[list[str]]:
    rows = []
    for item in state.get("asr_state", {}).get("window_records", [])[-12:]:
        window_label = f"{item['window_start']:.2f}-{item['window_end']:.2f}s"
        if item.get("skipped_as_silence"):
            window_label += "（静音跳过）"
        elif item.get("endpoint_reason"):
            window_label += f"（端点：{item['endpoint_reason']}）"
        else:
            window_label += f"（{item.get('vad_state', 'IDLE')}）"
        rows.append(
            [
                window_label,
                f"{item['window_latency']:.2f}s",
                str(item["segment_count"]),
                str(item["stable_count"]),
            ]
        )
    return rows


def _queue_live_tts(state: dict, stable_segments: list[dict]) -> None:
    with _MODEL_PIPELINE_LOCK:
        _queue_live_tts_unlocked(state, stable_segments)


def _enqueue_translated_tts_unlocked(
    state: dict,
    item: dict,
    translated_text: str,
) -> None:
    """Create one ordered TTS item after a translation was accepted."""
    segment_id = str(item["segment_id"])
    tts_dir = session_directory(SETTINGS, "live_tts", str(state["session_id"]))
    tts_dir.mkdir(parents=True, exist_ok=True)
    path = tts_dir / f"segment_{int(item['segment_id']):04d}.wav"
    try:
        tts.synthesize(translated_text, path)
        duration = _audio_duration(path)
        state["tts_paths"][segment_id] = tts_record(
            str(path),
            state["audio_cursor"],
            state["audio_cursor"] + duration,
        )
        state["audio_cursor"] += duration
        state["audio_queue"].append(
            audio_queue_item(item["segment_id"], str(path))
        )
        state["audio_queue_high_watermark"] = max(
            int(state.get("audio_queue_high_watermark", 0)),
            len(state["audio_queue"]),
        )
    except Exception as exc:
        state["last_error"] = f"TTS 异常：{exc}"
        logger.exception("TTS generation failed for segment %s", segment_id)


def _apply_degraded_translation_unlocked(
    state: dict,
    item: dict,
    reason: str,
) -> None:
    """Finalize one exhausted segment with a safe English-only fallback."""
    segment_id = str(item["segment_id"])
    fallback_text = _degraded_translation_text(item.get("text", ""))
    state["translation_sources"][segment_id] = item["text"]
    state.setdefault("translation_retry_segments", {}).pop(segment_id, None)
    state["translations"][segment_id] = translation_record(
        item["text"],
        fallback_text,
        "DEGRADED",
    )
    if int(item["segment_id"]) in state.get("failed_translation_ids", []):
        state["failed_translation_ids"].remove(int(item["segment_id"]))
    degraded_ids = state.setdefault("degraded_translation_ids", [])
    if int(item["segment_id"]) not in degraded_ids:
        degraded_ids.append(int(item["segment_id"]))
    state.setdefault("translation_degradation_reasons", {})[segment_id] = reason
    _enqueue_translated_tts_unlocked(state, item, fallback_text)
    state["last_error"] = (
        f"片段 {segment_id} 已使用英文降级文本（翻译重试上限：{reason}）"
    )


def _queue_live_tts_unlocked(state: dict, stable_segments: list[dict]) -> None:
    """Translate stable segments and enqueue each TTS file exactly once.

    Rejected translations remain as retry candidates, but each segment has a
    bounded application-level attempt count.  This keeps a transient MT/model
    failure from permanently skipping a confirmed segment or retrying forever.
    """
    attempts = state.setdefault("translation_attempts", {})
    retry_segments = state.setdefault("translation_retry_segments", {})
    candidates: dict[str, dict] = {
        str(segment_id): dict(item)
        for segment_id, item in retry_segments.items()
    }
    for item in stable_segments:
        if item.get("status") == "FINAL":
            candidates[str(item["segment_id"])] = item

    exhausted: list[dict] = []
    changed: list[dict] = []
    for segment_id, item in candidates.items():
        if segment_id in state["tts_paths"]:
            retry_segments.pop(segment_id, None)
            continue
        record = state["translations"].get(segment_id, {})
        if (
            record.get("quality_status") == "ACCEPTED"
            and state["translation_sources"].get(segment_id) == item["text"]
        ):
            retry_segments.pop(segment_id, None)
            continue
        if int(attempts.get(segment_id, 0)) >= MAX_TRANSLATION_ATTEMPTS:
            if record.get("quality_status") == "REJECTED":
                exhausted.append(item)
            continue
        previous = retry_segments.get(segment_id)
        if previous and previous.get("text") != item.get("text"):
            # A revised stable prefix is a new translation input.
            attempts[segment_id] = 0
        changed.append(item)
    for item in exhausted:
        _apply_degraded_translation_unlocked(
            state,
            item,
            f"{MAX_TRANSLATION_ATTEMPTS} 次尝试均未通过质量校验",
        )
    if not changed:
        return

    translated: list[str | None]
    translation_errors: list[str]
    for item in changed:
        segment_id = str(item["segment_id"])
        attempts[segment_id] = int(attempts.get(segment_id, 0)) + 1
    try:
        batch = translator.translate_many([item["text"] for item in changed])
        if len(batch) != len(changed):
            raise RuntimeError("翻译结果数量与稳定片段数量不一致")
        translated = [
            validate_translation(item["text"], result)
            for item, result in zip(changed, batch)
        ]
        translation_errors = [""] * len(changed)
    except Exception as batch_exc:
        # One bad response must not discard the other stable segments. Retry
        # item by item while respecting the application-level attempt limit.
        translated = [None] * len(changed)
        translation_errors = [str(batch_exc)] * len(changed)
        for index, item in enumerate(changed):
            segment_id = str(item["segment_id"])
            if int(attempts.get(segment_id, 0)) >= MAX_TRANSLATION_ATTEMPTS:
                translation_errors[index] = (
                    f"已达到翻译重试上限（{MAX_TRANSLATION_ATTEMPTS}次）"
                )
                continue
            try:
                attempts[segment_id] = int(attempts.get(segment_id, 0)) + 1
                translate_one = getattr(translator, "translate", None)
                if translate_one is not None:
                    result = translate_one(item["text"])
                else:
                    result = translator.translate_many([item["text"]])[0]
                translated[index] = validate_translation(item["text"], result)
                translation_errors[index] = ""
            except Exception as exc:
                translation_errors[index] = str(exc)
        state["last_error"] = f"批量翻译已降级为逐段校验：{batch_exc}"

    for item, translated_text, translation_error in zip(
        changed,
        translated,
        translation_errors,
    ):
        segment_id = str(item["segment_id"])
        if translated_text is None:
            if int(attempts.get(segment_id, 0)) >= MAX_TRANSLATION_ATTEMPTS:
                _apply_degraded_translation_unlocked(
                    state,
                    item,
                    f"{translation_error or '质量校验失败'}",
                )
                continue
            # Do not mark a rejected source as translated: the retry candidate
            # must remain discoverable on the next stable chunk/final flush.
            state["translation_sources"].pop(segment_id, None)
            retry_segments[segment_id] = dict(item)
            state["translations"][segment_id] = translation_record(
                item["text"],
                "⚠ 翻译质量校验失败（可重试），未进入语音队列",
                "REJECTED",
            )
            if int(item["segment_id"]) not in state["failed_translation_ids"]:
                state["failed_translation_ids"].append(int(item["segment_id"]))
            state["last_error"] = (
                f"片段 {segment_id} 翻译被质量门拦截：{translation_error}"
            )
            continue
        state["translation_sources"][segment_id] = item["text"]
        retry_segments.pop(segment_id, None)
        if int(item["segment_id"]) in state["failed_translation_ids"]:
            state["failed_translation_ids"].remove(int(item["segment_id"]))
        if int(item["segment_id"]) in state.get("degraded_translation_ids", []):
            state["degraded_translation_ids"].remove(int(item["segment_id"]))
        state.get("translation_degradation_reasons", {}).pop(segment_id, None)
        state["translations"][segment_id] = translation_record(
            item["text"], translated_text, "ACCEPTED"
        )
        _enqueue_translated_tts_unlocked(state, item, translated_text)


def _file_inference_worker(
    state_template: dict,
    pending: queue.Queue,
    completed: queue.Queue,
) -> None:
    """Translate/TTS file-stream segments without blocking audio ingestion.

    The worker owns a private inference state.  Only serializable deltas cross
    back to the Gradio generator, so a browser-session state never contains a
    Queue, Thread, lock, or another object that Gradio cannot copy safely.
    """
    worker_state = _new_live_state(
        "file",
        str(state_template.get("source_name", "uploaded.wav")),
    )
    worker_state["session_id"] = str(state_template["session_id"])
    worker_state["source_audio"] = str(state_template["source_audio"])
    while True:
        segments = pending.get()
        try:
            if segments is None:
                return
            segment_ids = [str(item["segment_id"]) for item in segments]
            before_failed = set(worker_state["failed_translation_ids"])
            before_degraded = set(worker_state.get("degraded_translation_ids", []))
            _queue_live_tts(worker_state, segments)
            completed.put(
                {
                    "translations": {
                        segment_id: dict(worker_state["translations"][segment_id])
                        for segment_id in segment_ids
                        if segment_id in worker_state["translations"]
                    },
                    "translation_sources": {
                        segment_id: worker_state["translation_sources"][segment_id]
                        for segment_id in segment_ids
                        if segment_id in worker_state["translation_sources"]
                    },
                    "translation_attempts": {
                        segment_id: worker_state["translation_attempts"][segment_id]
                        for segment_id in segment_ids
                        if segment_id in worker_state["translation_attempts"]
                    },
                    "translation_retry_segments": {
                        segment_id: dict(item)
                        for segment_id, item in worker_state[
                            "translation_retry_segments"
                        ].items()
                    },
                    "tts_paths": {
                        segment_id: dict(worker_state["tts_paths"][segment_id])
                        for segment_id in segment_ids
                        if segment_id in worker_state["tts_paths"]
                    },
                    "audio_queue": list(worker_state["audio_queue"]),
                    "failed_translation_ids": [
                        segment_id
                        for segment_id in worker_state["failed_translation_ids"]
                        if segment_id not in before_failed
                    ],
                    "cleared_failed_translation_ids": [
                        segment_id
                        for segment_id in before_failed
                        if segment_id not in worker_state["failed_translation_ids"]
                    ],
                    "degraded_translation_ids": list(
                        worker_state.get("degraded_translation_ids", [])
                    ),
                    "cleared_degraded_translation_ids": [
                        segment_id
                        for segment_id in before_degraded
                        if segment_id not in worker_state.get(
                            "degraded_translation_ids", []
                        )
                    ],
                    "translation_degradation_reasons": dict(
                        worker_state.get("translation_degradation_reasons", {})
                    ),
                    "last_error": worker_state.get("last_error", ""),
                }
            )
            # The main thread now owns these playback entries.  Keep the
            # private queue bounded while preserving the cumulative cursor.
            worker_state["audio_queue"].clear()
        except Exception as exc:  # keep the ingestion loop alive on MT/TTS errors
            completed.put({"last_error": f"后台翻译/语音异常：{exc}"})
        finally:
            pending.task_done()


def _apply_file_inference_results(state: dict, completed: queue.Queue) -> int:
    """Merge all ready worker deltas into the browser-session state."""
    applied = 0
    while True:
        try:
            result = completed.get_nowait()
        except queue.Empty:
            break
        try:
            state["translations"].update(result.get("translations", {}))
            state["translation_sources"].update(
                result.get("translation_sources", {})
            )
            state["translation_attempts"].update(
                result.get("translation_attempts", {})
            )
            if "translation_retry_segments" in result:
                state["translation_retry_segments"] = {
                    str(segment_id): dict(item)
                    for segment_id, item in result[
                        "translation_retry_segments"
                    ].items()
                }
            state["tts_paths"].update(result.get("tts_paths", {}))
            if result.get("tts_paths"):
                state["audio_cursor"] = max(
                    float(state.get("audio_cursor", 0.0)),
                    max(
                        float(item.get("audio_end", 0.0))
                        for item in result["tts_paths"].values()
                    ),
                )
            existing_audio_ids = {
                int(item["segment_id"]) for item in state.get("audio_queue", [])
            }
            existing_audio_ids.update(
                int(item) for item in state.get("played_segment_ids", [])
            )
            for item in result.get("audio_queue", []):
                if int(item["segment_id"]) not in existing_audio_ids:
                    state["audio_queue"].append(item)
                    existing_audio_ids.add(int(item["segment_id"]))
            for segment_id in result.get("failed_translation_ids", []):
                if segment_id not in state["failed_translation_ids"]:
                    state["failed_translation_ids"].append(segment_id)
            for segment_id in result.get("cleared_failed_translation_ids", []):
                if segment_id in state["failed_translation_ids"]:
                    state["failed_translation_ids"].remove(segment_id)
            for segment_id in result.get("degraded_translation_ids", []):
                if segment_id not in state.setdefault("degraded_translation_ids", []):
                    state["degraded_translation_ids"].append(segment_id)
            for segment_id in result.get("cleared_degraded_translation_ids", []):
                if segment_id in state.get("degraded_translation_ids", []):
                    state["degraded_translation_ids"].remove(segment_id)
            if "translation_degradation_reasons" in result:
                state["translation_degradation_reasons"] = dict(
                    result["translation_degradation_reasons"]
                )
            if result.get("last_error"):
                state["last_error"] = result["last_error"]
            state["audio_queue_high_watermark"] = max(
                int(state.get("audio_queue_high_watermark", 0)),
                len(state["audio_queue"]),
            )
            applied += 1
        finally:
            completed.task_done()
    return applied


def _consume_live_tts(state: dict) -> str | None:
    if not state.get("audio_queue"):
        return None
    item = state["audio_queue"].pop(0)
    segment_id = int(item["segment_id"])
    state["current_segment_id"] = segment_id
    if segment_id not in state["played_segment_ids"]:
        state["played_segment_ids"].append(segment_id)
    path = Path(item["path"])
    if path.exists():
        return str(path)
    state["last_error"] = f"读取 TTS 音频失败：文件不存在 {path}"
    return None


def _live_callback_outputs(
    state: dict,
    audio_path: str | None = None,
    *,
    preserve_audio_if_empty: bool = True,
) -> tuple[dict, str, str, object, object, str, str, str, list[list[str]]]:
    asr_state = state["asr_state"]
    all_segments = asr_state.get("segments", [])
    stable_segments = [
        item for item in all_segments if item.get("status") == "FINAL"
    ]
    translated_segments = [
        {
            **item,
            "translation": state["translations"].get(
                str(item["segment_id"]), {}
            ).get("text", "翻译中……"),
        }
        for item in stable_segments
        if str(item["segment_id"]) in state["translations"]
    ]
    total_seconds = asr_state.get("total_samples", 0) / SETTINGS.sample_rate
    queue_size = len(state.get("audio_queue", []))
    vad_label = {
        "IDLE": "等待讲话",
        "SPEECH": "讲话中",
        "SILENCE_PENDING": "检测停顿",
        "ONLINE_SPEECH": "流式识别中",
    }.get(asr_state.get("vad_state", "IDLE"), asr_state.get("vad_state", "IDLE"))
    input_label = state.get("input_label", "麦克风")
    status = (
        f"{input_label}已接收 {total_seconds:.1f}s | "
        f"{vad_label} | 稳定 {len(stable_segments)} 段 | "
        f"端点 {asr_state.get('endpoint_count', 0)} 次 | "
        f"音频队列 {queue_size}/{state['max_audio_queue']}"
    )
    audio_quality = asr_state.get("audio_quality", {})
    if audio_quality.get("sample_count", 0):
        status += f" | {audio_quality.get('label', '音量待检测')}"
    if asr_state.get("rejected_utterances"):
        status += f" | 已抑制可疑候选 {asr_state['rejected_utterances']} 段"
    if state.get("dropped_audio_ids"):
        status += f" | 已降级 {len(state['dropped_audio_ids'])} 段语音"
    if state.get("failed_translation_ids"):
        status += f" | 质量门拦截 {len(state['failed_translation_ids'])} 段"
    if state.get("degraded_translation_ids"):
        status += f" | 英文降级 {len(state['degraded_translation_ids'])} 段"
    if state.get("translation_jobs_pending"):
        status += f" | 翻译任务 {state['translation_jobs_pending']} 个"
    if asr.live_backend == "sherpa":
        pipeline = (
            "Sherpa Zipformer 原生流式 ASR → LocalAgreement 稳定前缀"
            f" → {translator.__class__.__name__} → Piper TTS 有界队列"
        )
    else:
        pipeline = (
            f"20ms VAD 端点控制 → {asr.live_backend.upper()} 临时识别"
            f" → 停顿后稳定提交 → {translator.__class__.__name__} → Piper TTS 有界队列"
        )
    if state.get("last_error"):
        pipeline += f" | {state['last_error']}"
    raw_text = _format_segments(
        [item for item in all_segments if item.get("status") != "REJECTED"]
    )
    stable_text = _format_segments(stable_segments)
    translated_text = _render_live_translation(state)
    audio_output: object
    if audio_path:
        audio_output = gr.update(value=audio_path, visible=True)
    elif preserve_audio_if_empty:
        # Streaming callbacks arrive every 0.4 seconds. Returning None would
        # clear the native Gradio player immediately after a TTS segment was
        # published; gr.skip() keeps the current WAV loaded.
        audio_output = gr.skip()
    else:
        audio_output = gr.update(value=None, visible=False)
    return (
        state,
        raw_text,
        stable_text,
        translated_text,
        audio_output,
        _render_live_subtitles(state),
        status,
        pipeline,
        _live_metrics(state),
    )


def reset_live_session():
    state = _new_live_state()
    return _live_callback_outputs(state, preserve_audio_if_empty=False)


def switch_live_input_mode(mode: str):
    """Show only the controls for the selected real-time input source."""
    use_microphone = mode == "麦克风实时输入"
    help_text = (
        "麦克风模式：模型就绪后点击开始同声传译，系统持续接收音频并边说边译。"
        if use_microphone
        else (
            "上传模式：选择 WAV 后点击开始。源音频会同步播放，系统按真实时间"
            "每 0.4 秒推送一块，不能读取尚未播放的后续内容。"
        )
    )
    return (
        gr.update(visible=use_microphone),
        gr.update(visible=not use_microphone),
        help_text,
    )


def _live_outputs_with_messages(
    state: dict,
    status: str,
    pipeline: str,
    *,
    clear_audio: bool = False,
    audio_path: str | None = None,
):
    outputs = list(
        _live_callback_outputs(
            state,
            audio_path=audio_path,
            preserve_audio_if_empty=not clear_audio,
        )
    )
    outputs[6] = status
    outputs[7] = pipeline
    return tuple(outputs)


def warmup_live_models():
    started = time.perf_counter()
    progress_button = gr.update(
        value="⏳\n正在启动模型",
        interactive=False,
    )
    try:
        yield (
            "正在预热中文 ASR……",
            "模型预热 1/3：本地中文流式 ASR",
            progress_button,
        )
        asr.load_live()
        yield (
            "ASR 已就绪，正在预热翻译模型……",
            f"模型预热 2/3：{translator.__class__.__name__}",
            progress_button,
        )
        translator.load()
        yield (
            "ASR 和翻译已就绪，正在预热英文 TTS……",
            "模型预热 3/3：Piper TTS",
            progress_button,
        )
        tts.load()
        elapsed = time.perf_counter() - started
        yield (
            f"全部模型已就绪，用时 {elapsed:.1f}s；点击开始同声传译",
            f"{asr.live_backend.upper()} + {translator.__class__.__name__} + Piper TTS 已预热",
            gr.update(
                value="▶\n开始同声传译",
                interactive=True,
            ),
        )
    except Exception as exc:
        yield (
            f"模型预热失败：{exc}",
            "请检查终端错误和本地模型目录",
            gr.update(
                value="↻\n重新启动模型",
                interactive=True,
            ),
        )


def process_live_chunk(
    audio_chunk: tuple[int, np.ndarray] | None,
    state: dict | None,
):
    state = state or _new_live_state()
    if state.get("input_closed"):
        return _live_outputs_with_messages(
            state,
            state.get("finalization_message", "录音输入已关闭"),
            "忽略停止录音后迟到的浏览器音频块",
        )
    coerced = StreamingASR._coerce_stream_audio(audio_chunk)
    _append_live_source(state, coerced)
    _update_asr_backpressure(state)
    result = asr.transcribe_incremental(
        coerced,
        state["asr_state"],
    )
    _queue_live_tts(state, result.get("new_stable", []))
    return _live_callback_outputs(state, _consume_live_tts(state))


def stream_uploaded_audio(audio_path: str | None):
    """Push an uploaded recording through the native live pipeline in real time."""
    source_name = Path(audio_path).name if audio_path else None
    state = _new_live_state("file", source_name)
    control = _register_file_transport(state["session_id"])
    if not audio_path:
        yield (
            None,
            *_live_outputs_with_messages(
                state,
                "请先上传 WAV 音频",
                "等待上传源演讲音频",
                clear_audio=True,
            ),
        )
        _release_file_transport(state["session_id"])
        return

    path = Path(audio_path)
    if not path.exists():
        yield (
            None,
            *_live_outputs_with_messages(
                state,
                f"找不到上传文件：{path}",
                "文件流式推送未启动",
                clear_audio=True,
            ),
        )
        _release_file_transport(state["session_id"])
        return

    yield (
        None,
        *_live_outputs_with_messages(
            state,
            "正在准备本地模型和源音频……",
            "准备 Sherpa Zipformer、翻译模型与 Piper TTS",
            clear_audio=True,
        ),
    )
    try:
        asr.load_live()
        translator.load()
        tts.load()
        samples = load_audio_16k(path)
    except Exception as exc:
        state["last_error"] = f"准备失败：{exc}"
        yield (
            None,
            *_live_outputs_with_messages(
                state,
                f"上传音频启动失败：{exc}",
                "请检查模型文件、音频格式与终端错误",
                clear_audio=True,
            ),
        )
        _release_file_transport(state["session_id"])
        return

    if samples.size == 0:
        yield (
            None,
            *_live_outputs_with_messages(
                state,
                "上传音频为空，无法开始",
                "文件流式推送未启动",
                clear_audio=True,
            ),
        )
        _release_file_transport(state["session_id"])
        return

    duration = len(samples) / SETTINGS.sample_rate
    yield (
        str(path),
        *_live_outputs_with_messages(
            state,
            f"上传音频已就绪，共 {duration:.1f}s；开始真实时间流式推送",
            (
                "源音频同步播放 → 0.4s 音频块 → Sherpa Zipformer 原生流式 ASR"
                " → LocalAgreement → 翻译 → Piper TTS"
            ),
            clear_audio=True,
        ),
    )

    stream_started = time.perf_counter()
    streamed_seconds = 0.0
    pending_inference: queue.Queue = queue.Queue()
    completed_inference: queue.Queue = queue.Queue()
    inference_worker = threading.Thread(
        target=_file_inference_worker,
        args=(state, pending_inference, completed_inference),
        name=f"file-inference-{state['session_id']}",
        daemon=True,
    )
    inference_worker.start()
    try:
        for sample_rate, block in iter_audio_blocks(
            samples,
            sample_rate=SETTINGS.sample_rate,
            block_seconds=SETTINGS.stream_block_seconds,
        ):
            if control["stopped"].is_set():
                break
            if control["paused"].is_set():
                pause_started = time.perf_counter()
                while control["paused"].is_set() and not control["stopped"].is_set():
                    time.sleep(0.05)
                stream_started += time.perf_counter() - pause_started
            if control["stopped"].is_set():
                break
            streamed_seconds += len(block) / sample_rate
            remaining = streamed_seconds - (time.perf_counter() - stream_started)
            if remaining > 0:
                time.sleep(remaining)

            coerced = (sample_rate, block)
            _append_live_source(state, coerced)
            _update_asr_backpressure(state)
            result = asr.transcribe_incremental(
                coerced,
                state["asr_state"],
            )
            _apply_file_inference_results(state, completed_inference)
            new_stable = result.get("new_stable", [])
            if new_stable:
                pending_inference.put(list(new_stable))
            state["translation_jobs_pending"] = pending_inference.unfinished_tasks
            yield (
                gr.skip(),
                *_live_callback_outputs(state, _consume_live_tts(state)),
            )

        # Source ingestion is now complete.  Keep publishing worker results so
        # the UI remains responsive while the ordered MT/TTS queue catches up.
        while pending_inference.unfinished_tasks:
            if control["stopped"].is_set() and not state["asr_state"].get("segments"):
                break
            _apply_file_inference_results(state, completed_inference)
            state["translation_jobs_pending"] = pending_inference.unfinished_tasks
            yield (
                gr.skip(),
                *_live_callback_outputs(state, _consume_live_tts(state)),
            )
            time.sleep(0.2)
        _apply_file_inference_results(state, completed_inference)
        state["translation_jobs_pending"] = 0
        pending_inference.put(None)
        inference_worker.join(timeout=SETTINGS.worker_join_seconds)
        for outputs in finalize_live_session(state):
            yield (gr.skip(), *outputs)
    except GeneratorExit:
        raise
    except Exception as exc:
        state["last_error"] = f"文件流式推送异常：{exc}"
        logger.exception("Uploaded file streaming failed: %s", state["session_id"])
        yield (
            gr.skip(),
            *_live_outputs_with_messages(
                state,
                f"文件流式推送失败：{exc}",
                "流水线已安全停止",
            ),
        )
    finally:
        if inference_worker.is_alive():
            pending_inference.put(None)
        _release_file_transport(state["session_id"])


def finalize_live_session(state: dict | None):
    state = state or _new_live_state()
    # A naturally completed uploaded stream has already finalized and removed
    # its Sherpa native stream.  A late stop click must not create a second
    # native stream or run the finalizer twice.
    if state.get("asr_state", {}).get("finalized"):
        yield _live_callback_outputs(state)
        return
    _update_asr_backpressure(state)
    result = asr.transcribe_incremental(None, state["asr_state"], final=True)
    _queue_live_tts(state, result.get("new_stable", []))
    stable_segments = [
        item
        for item in state["asr_state"].get("segments", [])
        if item.get("status") == "FINAL"
    ]
    translated_by_id = {
        int(segment_id): (
            item.get("source", ""),
            item.get("text", ""),
        )
        for segment_id, item in state["translations"].items()
        if item.get("quality_status") in {"ACCEPTED", "DEGRADED"}
    }
    audio_by_id = {
        int(segment_id): item["path"]
        for segment_id, item in state["tts_paths"].items()
    }
    if stable_segments and translated_by_id and audio_by_id:
        try:
            _save_live_results(
                state["source_audio"],
                stable_segments,
                translated_by_id,
                audio_by_id,
                state["asr_state"].get("window_records", []),
                {
                    **state["asr_state"],
                    "degraded_translation_ids": list(
                        state.get("degraded_translation_ids", [])
                    ),
                },
            )
        except Exception as exc:
            state["last_error"] = f"结果保存异常：{exc}"
    if state.get("audio_queue"):
        while state["audio_queue"]:
            yield _live_callback_outputs(state, _consume_live_tts(state))
    else:
        yield _live_callback_outputs(state)


def _write_finalization_log(state: dict, message: str) -> Path:
    log_dir = PROJECT_ROOT / ".cache" / "runtime" / "crash_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{state['session_id']}.log"
    path.write_text(
        (
            f"session_id={state['session_id']}\n"
            f"time={datetime.now().isoformat(timespec='seconds')}\n"
            f"message={message}\n\n"
            f"{traceback.format_exc()}"
        ),
        encoding="utf-8",
    )
    return path


def _finalize_microphone_worker(state: dict) -> None:
    """Flush ASR/translation/TTS off the Gradio stop-recording callback."""
    try:
        state["finalization_message"] = "正在刷新 ASR 尾段……"
        if not state.get("asr_state", {}).get("finalized"):
            _update_asr_backpressure(state)
            result = asr.transcribe_incremental(
                None,
                state["asr_state"],
                final=True,
            )
            state["finalization_message"] = "正在校验翻译并生成尾段语音……"
            _queue_live_tts(state, result.get("new_stable", []))

        stable_segments = [
            item
            for item in state["asr_state"].get("segments", [])
            if item.get("status") == "FINAL"
        ]
        translated_by_id = {
            int(segment_id): (
                item.get("source", ""),
                item.get("text", ""),
            )
            for segment_id, item in state["translations"].items()
            if item.get("quality_status") in {"ACCEPTED", "DEGRADED"}
        }
        audio_by_id = {
            int(segment_id): item["path"]
            for segment_id, item in state["tts_paths"].items()
        }
        if stable_segments:
            state["finalization_message"] = "正在保存完整 TXT、WAV 与指标……"
            _save_live_results(
                state["source_audio"],
                stable_segments,
                translated_by_id,
                audio_by_id,
                state["asr_state"].get("window_records", []),
                {
                    **state["asr_state"],
                    "degraded_translation_ids": list(
                        state.get("degraded_translation_ids", [])
                    ),
                },
            )
        state["finalization_status"] = "DONE"
        state["finalization_message"] = "录音尾段处理完成"
    except Exception as exc:
        state["last_error"] = f"录音尾段处理异常：{exc}"
        state["finalization_status"] = "ERROR"
        logger.exception("Microphone finalization failed: %s", state["session_id"])
        try:
            log_path = _write_finalization_log(state, str(exc))
            state["finalization_message"] = f"尾段处理失败，日志：{log_path}"
        except Exception:
            state["finalization_message"] = "尾段处理失败，且日志写入失败"
    finally:
        with _FINALIZATION_LOCK:
            _FINALIZATION_THREADS.pop(state["session_id"], None)


def start_microphone_finalization(state: dict | None):
    """Close microphone input immediately and finish the pipeline in background."""
    state = state or _new_live_state()
    state["input_closed"] = True
    session_id = str(state["session_id"])
    with _FINALIZATION_LOCK:
        existing = _FINALIZATION_THREADS.get(session_id)
        if existing is None or not existing.is_alive():
            state["finalization_status"] = "RUNNING"
            state["finalization_message"] = "录音已停止，正在后台处理尾段……"
            worker = threading.Thread(
                target=_finalize_microphone_worker,
                args=(state,),
                name=f"simul-finalize-{session_id}",
                daemon=True,
            )
            _FINALIZATION_THREADS[session_id] = worker
            worker.start()
    return _live_outputs_with_messages(
        state,
        state["finalization_message"],
        "输入已关闭；ASR 尾段、翻译、TTS 与结果保存正在后台执行",
    )


def poll_microphone_finalization(state: dict | None):
    if not state or state.get("finalization_status") == "IDLE":
        return tuple(gr.skip() for _ in range(9))
    audio_path = _consume_live_tts(state)
    status = state.get("finalization_message", "正在处理录音尾段……")
    if state.get("finalization_status") == "DONE" and not state.get("audio_queue"):
        status += "；全部英文语音片段已就绪"
    return _live_outputs_with_messages(
        state,
        status,
        (
            "后台尾段处理 → 质量门 → Piper TTS 队列"
            if state.get("finalization_status") == "RUNNING"
            else "录音会话已安全结束"
        ),
        audio_path=audio_path,
    )


with gr.Blocks(title="进点双子星队 · 实时同传 Agent") as demo:
    gr.HTML(
        """
        <header id="v3-topbar">
          <div class="v3-brand">
            <strong>进点双子星 · 实时同传</strong>
            <span>Chinese → English · Local interpretation agent</span>
          </div>
          <div class="v3-local"><i></i><span>本地运行 · 数据不出机</span><span id="v3-clock">--:--:--</span></div>
        </header>
        """
    )

    live_state = gr.State(_new_live_state())
    live_finalization_timer = gr.Timer(0.5, active=True)

    with gr.Tabs():
        with gr.Tab("实时工作台"):
            with gr.Row(elem_id="v3-hero"):
                with gr.Column(scale=3, elem_id="v3-hero-copy"):
                    gr.HTML(
                        """
                        <div class="v3-live-line"><i></i><span>等待输入 · 先点火预热模型</span></div>
                        <h1>让英文，跟上演讲。</h1>
                        <p>只保留现场真正需要的信息：输入状态、已经确认的中文，以及正在输出的英文。</p>
                        <div id="v3-wave" aria-label="实时音频活动波形"></div>
                        """
                    )
                    live_input_mode = gr.Radio(
                        choices=["麦克风实时输入", "上传 WAV 流式推送"],
                        value="麦克风实时输入",
                        label="输入方式",
                        elem_id="v3-input-mode",
                    )
                with gr.Column(scale=1, min_width=300, elem_id="v3-ignition-panel"):
                    gr.HTML(
                        '<div class="v3-rings" aria-hidden="true"></div>',
                        elem_id="v3-rings-layer",
                    )
                    live_warmup = gr.Button(
                        "⏻\n模型启动",
                        variant="secondary",
                        elem_id="v3-warmup",
                    )
                    gr.HTML('<div class="v3-ignition-caption">启动本地 ASR · 翻译 · TTS</div>')

            with gr.Column(elem_id="v3-status-card"):
                live_mode_help = gr.Markdown(
                    "麦克风模式：模型就绪后点击开始同声传译，系统持续接收音频并边说边译。",
                    elem_id="v3-mode-help",
                )
                with gr.Row():
                    live_status = gr.Textbox(
                        label="实时状态",
                        value="等待选择输入并开始",
                        interactive=False,
                    )
                    live_pipeline = gr.Textbox(
                        label="调度链路",
                        value="Zipformer → LocalAgreement → Qwen 翻译 → Piper TTS",
                        interactive=False,
                    )

            # The native recorder is a transport only.  It remains mounted so
            # Gradio can stream browser microphone chunks, while the hero's
            # single session button is the only visible microphone control.
            live_mic = gr.Audio(
                label="麦克风实时输入",
                sources=["microphone"],
                type="numpy",
                streaming=True,
                recording=False,
                buttons=[],
                waveform_options={"show_recording_waveform": True},
                elem_id="v3-mic-transport",
            )
            microphone_group = live_mic
            with gr.Column(visible=False, elem_id="v3-source-card-file") as upload_group:
                gr.HTML(
                    '<div class="v3-section-title"><h2>流式音频输入</h2>'
                    '<p>左侧载入源演讲，右侧按真实进度同步播放</p></div>'
                )
                with gr.Row(elem_id="v3-file-audio-grid"):
                    file_upload = gr.Audio(
                        label="上传源演讲音频（WAV）",
                        type="filepath",
                        sources=["upload"],
                        buttons=["download"],
                        elem_id="v3-upload-audio",
                    )
                    file_source_playback = gr.Audio(
                        label="源演讲同步播放",
                        type="filepath",
                        autoplay=True,
                        interactive=False,
                        buttons=[],
                        elem_id="v3-source-playback",
                    )
                with gr.Row(elem_id="v3-file-controls"):
                    file_start = gr.Button(
                        "▶ 开始文件流式同传",
                        variant="primary",
                        elem_id="v3-file-start",
                    )
                    file_stop = gr.Button(
                        "■ 停止",
                        variant="secondary",
                        elem_id="v3-file-stop",
                    )

            with gr.Column(elem_id="v3-semantic-card"):
                gr.HTML(
                    '<div class="v3-section-title"><h2>实时语义对照</h2>'
                    '<p>中英文按照同一个时间片成对呈现，点击英文单元即可独立回放</p>'
                    '<span class="v3-section-status">当前播放与字幕同步</span></div>'
                )
                live_translation = gr.HTML(
                    value=_render_live_translation(_new_live_state()),
                    elem_id="live-translation",
                )
                with gr.Accordion("识别详情", open=False, elem_id="v3-recognition-details"):
                    live_stable = gr.Textbox(
                        label="中文稳定转录（文本备份）",
                        lines=3,
                        interactive=False,
                        elem_id="live-stable",
                    )
                    live_raw = gr.Textbox(
                        label="中文原始增量转录（幻觉过滤后）",
                        lines=7,
                        interactive=False,
                    )

            with gr.Column(elem_id="v3-timeline-card"):
                gr.HTML(
                    '<div class="v3-section-title v3-player-title"><h2>连续英文同传语音</h2>'
                    '<p>播放队列与上方翻译单元同步，高亮表示当前语音片段</p></div>'
                )
                live_subtitles = gr.HTML(
                    '<div class="live-subtitles-empty">等待语音输入……</div>',
                    elem_id="live-subtitles",
                )
                live_audio_output = gr.Audio(
                    label="当前英文同传语音",
                    type="filepath",
                    autoplay=True,
                    interactive=False,
                    buttons=["download"],
                    visible=False,
                    elem_id="live-audio-player",
                )
                with gr.Accordion("实时窗口指标", open=False, elem_id="v3-metrics-details"):
                    live_metrics = gr.Dataframe(
                        headers=["语音窗口 / VAD", "处理耗时", "临时片段数", "稳定片段总数"],
                        datatype=["str", "str", "str", "str"],
                        interactive=False,
                    )

            # Stable bridge between custom visible rows and Gradio callbacks.
            # CSS moves it off-screen; unlike visible=False, the controls stay
            # mounted and therefore remain addressable from browser JavaScript.
            with gr.Column(elem_id="v3-ui-transport"):
                selected_segment = gr.Textbox(value="", elem_id="v3-selected-segment")
                play_segment = gr.Button("播放片段", elem_id="v3-play-segment")

        with gr.Tab("结果回放"):
            with gr.Column(elem_id="v3-results-card"):
                gr.HTML(
                    '<div class="v3-section-title"><h2>结果回放</h2>'
                    '<p>核对最终英文文本、完整语音和本次运行指标</p></div>'
                )
                refresh_results = gr.Button("刷新最新结果", variant="secondary")
                result_summary = gr.Textbox(label="最新结果", value="等待检查", interactive=False)
                result_table = gr.Dataframe(
                    headers=["指标", "数值"], datatype=["str", "str"],
                    label="最新完整运行指标", interactive=False,
                )
                full_audio = gr.Audio(
                    label="最新完整英文合成语音", type="filepath",
                    interactive=False, buttons=["download"],
                )
                with gr.Row():
                    translation_download = gr.File(label="英文翻译 TXT", interactive=False)
                    tts_download = gr.File(label="完整英文语音 WAV", interactive=False)
                    metrics_download = gr.File(label="运行指标 JSON", interactive=False)

        with gr.Tab("系统状态"):
            with gr.Column(elem_id="v3-system-card"):
                gr.HTML(
                    '<div class="v3-section-title"><h2>系统状态</h2>'
                    '<p>本地模型、GPU、稳定器和输出目录的真实健康状态</p></div>'
                )
                refresh_system = gr.Button("刷新系统状态", variant="secondary")
                system_table = gr.Dataframe(
                    headers=["模块", "状态", "说明"], datatype=["str", "str", "str"],
                    label="系统与模型状态", interactive=False,
                )

    gr.HTML(
        '<footer id="v3-footer"><span>进点双子星队 · 中文演讲到英文文本与语音</span>'
        '<span>所有真实推理均在本地执行</span></footer>'
    )

    live_outputs = [
        live_state,
        live_raw,
        live_stable,
        live_translation,
        live_audio_output,
        live_subtitles,
        live_status,
        live_pipeline,
        live_metrics,
    ]
    live_warmup.click(
        warmup_live_models,
        outputs=[live_status, live_pipeline, live_warmup],
        queue=True,
        concurrency_limit=1,
        concurrency_id="live_pipeline",
    )
    live_input_mode.change(
        switch_live_input_mode,
        inputs=live_input_mode,
        outputs=[microphone_group, upload_group, live_mode_help],
        queue=False,
    )
    live_mic.start_recording(
        reset_live_session,
        outputs=live_outputs,
        queue=True,
        concurrency_limit=1,
        concurrency_id="live_pipeline",
    )
    live_mic.stream(
        process_live_chunk,
        inputs=[live_mic, live_state],
        outputs=live_outputs,
        stream_every=0.4,
        concurrency_limit=1,
        concurrency_id="live_pipeline",
        queue=True,
    )
    live_mic.stop_recording(
        start_microphone_finalization,
        inputs=live_state,
        outputs=live_outputs,
        queue=False,
    )
    live_finalization_timer.tick(
        poll_microphone_finalization,
        inputs=live_state,
        outputs=live_outputs,
        queue=False,
    )
    file_start.click(
        stream_uploaded_audio,
        inputs=file_upload,
        outputs=[file_source_playback, *live_outputs],
        queue=True,
        concurrency_limit=1,
        concurrency_id="live_pipeline",
    )
    file_source_playback.pause(
        pause_uploaded_stream,
        inputs=live_state,
        queue=False,
        js="""
        (state) => {
            const tts = document.querySelector('#live-audio-player audio');
            if (tts && !tts.paused) tts.pause();
        }
        """,
    )
    file_source_playback.play(
        resume_uploaded_stream,
        inputs=live_state,
        queue=False,
        js="""
        (state) => {
            const tts = document.querySelector('#live-audio-player audio');
            if (tts && tts.paused && tts.currentTime > 0 && !tts.ended) {
                tts.play().catch(() => {});
            }
        }
        """,
    )
    file_source_playback.stop(
        pause_uploaded_stream,
        inputs=live_state,
        queue=False,
        js="""
        (state) => {
            const tts = document.querySelector('#live-audio-player audio');
            if (tts) tts.pause();
        }
        """,
    )
    file_stop.click(stop_uploaded_stream, inputs=live_state, queue=False)
    play_segment.click(
        play_translation_segment,
        inputs=[live_state, selected_segment],
        outputs=live_audio_output,
        queue=False,
    )

    status_outputs = [
        system_table,
        result_summary,
        result_table,
        translation_download,
        tts_download,
        metrics_download,
        full_audio,
    ]
    refresh_results.click(refresh_results_page, outputs=status_outputs)
    refresh_system.click(refresh_results_page, outputs=status_outputs)
    demo.load(fn=None, js=V3_APP_JS)
    demo.load(refresh_results_page, outputs=status_outputs)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("WEB_PORT", "7860")),
        inbrowser=os.environ.get("INBROWSER", "1") == "1",
        show_error=True,
        css=V3_APP_CSS,
    )
