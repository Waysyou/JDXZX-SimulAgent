from __future__ import annotations

import logging
import re
import shutil
import tempfile
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TypedDict

from .app_config import RuntimeSettings


_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class StartupCheck(TypedDict):
    name: str
    status: str
    detail: str
    required: bool


class SimulAgentError(RuntimeError):
    """Base error carrying a stable code and pipeline stage."""

    def __init__(self, message: str, *, code: str, stage: str) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "stage": self.stage, "message": str(self)}


def configure_logging(settings: RuntimeSettings) -> logging.Logger:
    logger = logging.getLogger("simul_agent")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if any(getattr(handler, "_simul_agent_handler", False) for handler in logger.handlers):
        return logger
    log_dir = settings.runtime_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "agent.log",
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s"
        )
    )
    handler._simul_agent_handler = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return logger


def session_directory(
    settings: RuntimeSettings,
    category: str,
    session_id: str,
) -> Path:
    if not _SESSION_ID.fullmatch(session_id):
        raise ValueError(f"Invalid session id: {session_id!r}")
    if not _SESSION_ID.fullmatch(category):
        raise ValueError(f"Invalid runtime category: {category!r}")
    return settings.runtime_root / category / session_id


def remove_runtime_path(settings: RuntimeSettings, path: str | Path) -> bool:
    """Remove one generated path only when it is below .cache/runtime."""
    target = Path(path).resolve(strict=False)
    runtime_root = settings.runtime_root.resolve(strict=False)
    try:
        target.relative_to(runtime_root)
    except ValueError:
        return False
    if target == runtime_root or not target.exists():
        return False
    if target.is_file() or target.is_symlink():
        target.unlink(missing_ok=True)
    else:
        shutil.rmtree(target, ignore_errors=True)
    return not target.exists()


def _check(
    name: str,
    condition: bool,
    detail: str,
    *,
    required: bool = True,
) -> StartupCheck:
    return {
        "name": name,
        "status": "ok" if condition else "error" if required else "warning",
        "detail": detail,
        "required": required,
    }


def _directory_is_writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, delete=True):
            return True
    except OSError:
        return False


def run_startup_checks(
    settings: RuntimeSettings,
    *,
    ui_root: str | Path | None = None,
) -> list[StartupCheck]:
    sherpa = settings.sherpa_model_dir
    qwen = settings.qwen_model_dir
    piper = settings.piper_voice_model
    checks = [
        _check(
            "Sherpa Zipformer",
            sherpa.is_dir()
            and bool(list(sherpa.glob("encoder*.onnx")))
            and bool(list(sherpa.glob("decoder*.onnx")))
            and bool(list(sherpa.glob("joiner*.onnx")))
            and (sherpa / "tokens.txt").is_file(),
            str(sherpa),
        ),
        _check(
            "Qwen translator",
            qwen.is_dir()
            and (qwen / "config.json").is_file()
            and bool(list(qwen.glob("*.safetensors"))),
            str(qwen),
        ),
        _check(
            "Piper voice",
            piper.is_file() and piper.with_suffix(piper.suffix + ".json").is_file(),
            str(piper),
        ),
        _check(
            "Piper eSpeak data",
            (settings.piper_espeak_data / "phontab").is_file(),
            str(settings.piper_espeak_data),
        ),
        _check(
            "Runtime directory",
            _directory_is_writable(settings.runtime_root),
            str(settings.runtime_root),
        ),
        _check(
            "Output directory",
            _directory_is_writable(settings.live_output_dir),
            str(settings.live_output_dir),
        ),
    ]
    if ui_root is not None:
        ui_path = Path(ui_root)
        checks.append(
            _check("V4 UI", (ui_path / "index.html").is_file(), str(ui_path))
        )
    return checks


def checks_ok(checks: list[StartupCheck]) -> bool:
    return all(item["status"] == "ok" for item in checks if item["required"])
