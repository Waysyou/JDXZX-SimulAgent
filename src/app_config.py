from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Validated process settings shared by Gradio and the V4 API."""

    project_root: Path
    sample_rate: int = 16_000
    stream_block_seconds: float = 0.4
    max_audio_queue: int = 4
    max_sessions: int = 12
    max_chunk_bytes: int = 2_000_000
    max_file_upload_bytes: int = 256_000_000
    result_list_limit: int = 24
    worker_join_seconds: float = 2.0
    log_max_bytes: int = 2_000_000
    log_backup_count: int = 3

    @classmethod
    def from_env(cls, project_root: str | Path) -> "RuntimeSettings":
        return cls(
            project_root=Path(project_root).resolve(),
            # The bundled streaming ASR model is fixed at 16 kHz. Keeping this
            # non-configurable prevents an environment override from silently
            # degrading recognition or corrupting timestamps.
            sample_rate=16_000,
            stream_block_seconds=_env_float(
                "SIMUL_STREAM_BLOCK_SECONDS", 0.4, minimum=0.05
            ),
            max_audio_queue=_env_int("SIMUL_MAX_AUDIO_QUEUE", 4),
            max_sessions=_env_int("SIMUL_MAX_SESSIONS", 12),
            max_chunk_bytes=_env_int("SIMUL_MAX_CHUNK_BYTES", 2_000_000),
            max_file_upload_bytes=_env_int(
                "SIMUL_MAX_FILE_UPLOAD_BYTES", 256_000_000
            ),
            result_list_limit=_env_int("SIMUL_RESULT_LIST_LIMIT", 24),
            worker_join_seconds=_env_float(
                "SIMUL_WORKER_JOIN_SECONDS", 2.0, minimum=0.1
            ),
            log_max_bytes=_env_int("SIMUL_LOG_MAX_BYTES", 2_000_000),
            log_backup_count=_env_int("SIMUL_LOG_BACKUP_COUNT", 3),
        )

    @property
    def cache_root(self) -> Path:
        return self.project_root / ".cache"

    @property
    def runtime_root(self) -> Path:
        return self.cache_root / "runtime"

    @property
    def output_root(self) -> Path:
        return self.project_root / "outputs"

    @property
    def live_output_dir(self) -> Path:
        return self.output_root / "live"

    @property
    def sherpa_model_dir(self) -> Path:
        return Path(
            os.environ.get(
                "SHERPA_MODEL_DIR",
                self.cache_root
                / "models"
                / "sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30",
            )
        )

    @property
    def qwen_model_dir(self) -> Path:
        return Path(
            os.environ.get(
                "QWEN_MODEL_DIR",
                self.cache_root / "models" / "qwen2.5-1.5b-instruct",
            )
        )

    @property
    def piper_voice_model(self) -> Path:
        return Path(
            os.environ.get(
                "PIPER_VOICE_MODEL",
                self.cache_root
                / "models"
                / "piper-en_US-lessac-medium"
                / "en_US-lessac-medium.onnx",
            )
        )

    @property
    def piper_espeak_data(self) -> Path:
        return Path(
            os.environ.get(
                "PIPER_ESPEAK_DATA_DIR",
                self.project_root / "runtime" / "piper_espeak_data",
            )
        )
