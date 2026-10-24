"""Standalone V2 entry point. Checks never load ASR, translation or TTS weights."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATHS = {
    "SHERPA_MODEL_DIR": ".cache/models/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30",
    "QWEN_MODEL_DIR": ".cache/models/qwen2.5-1.5b-instruct",
    "PIPER_VOICE_MODEL": ".cache/models/piper-en_US-lessac-medium/en_US-lessac-medium.onnx",
}
OPTIONAL_PATHS = {"PIPER_ESPEAK_DATA_DIR", "PIPER_ESPEAK_CACHE_DIR"}
ALLOWED_SETTINGS = set(MODEL_PATHS) | OPTIONAL_PATHS | {"TRANSLATION_DEVICE"}
PIPELINE_DEFAULTS = {
    "TRANSLATION_BACKEND": "qwen", "ASR_LIVE_BACKEND": "sherpa",
    "SHERPA_NUM_THREADS": "4", "SHERPA_DECODING_METHOD": "greedy_search",
    "SHERPA_MAX_ACTIVE_PATHS": "4", "SHERPA_LOCAL_AGREEMENT_HOLDBACK": "2",
    "SHERPA_COMMIT_MIN_CHARS": "14", "SHERPA_COMMIT_MAX_CHARS": "22",
    "SHERPA_COMMIT_MIN_SECONDS": "1.0", "ASR_VAD_ENDPOINT_SECONDS": "1.00",
    "ASR_PARTIAL_MIN_SECONDS": "2.00", "ASR_PARTIAL_EVERY_SECONDS": "1.20",
    "GRADIO_ANALYTICS_ENABLED": "False", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def _absolute(root: Path, value: str) -> str:
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else root / path).resolve())


def configure_environment(root: Path = ROOT) -> None:
    """Explicit env > local JSON > project defaults, with no parent-folder fallback."""
    root = root.resolve()
    settings_file = root / "config" / "local_settings.json"
    settings = json.loads(settings_file.read_text(encoding="utf-8")) if settings_file.is_file() else {}
    if not isinstance(settings, dict) or set(settings) - ALLOWED_SETTINGS:
        raise ValueError("local_settings.json contains unsupported settings")
    if any(not isinstance(v, str) or not v.strip() for v in settings.values()):
        raise ValueError("Local setting values must be non-empty strings")
    for name, value in settings.items():
        os.environ.setdefault(name, value)
    for name, default in MODEL_PATHS.items():
        os.environ[name] = _absolute(root, os.environ.get(name) or default)
    espeak = os.environ.get("PIPER_ESPEAK_DATA_DIR")
    if not espeak:
        # Prefer a deliberate project copy, otherwise use the installed Piper resource.
        data = root / "runtime" / "piper_espeak_data"
        if not (data / "phontab").is_file():
            spec = importlib.util.find_spec("piper")
            if spec and spec.origin:
                packaged = Path(spec.origin).parent / "espeak-ng-data"
                if (packaged / "phontab").is_file():
                    data = packaged
        espeak = str(data)
    os.environ["PIPER_ESPEAK_DATA_DIR"] = _absolute(root, espeak)
    for name in OPTIONAL_PATHS - {"PIPER_ESPEAK_DATA_DIR"}:
        if os.environ.get(name):
            os.environ[name] = _absolute(root, os.environ[name])
    for name, value in PIPELINE_DEFAULTS.items():
        os.environ.setdefault(name, value)
    os.environ.setdefault("HF_HOME", str(root / ".cache" / "hf"))


def startup_report(root: Path = ROOT) -> list[dict]:
    from src.app_config import RuntimeSettings
    from src.runtime_support import run_startup_checks
    return run_startup_checks(RuntimeSettings.from_env(root), ui_root=root / "ui_connected_v4")


def available_port(requested: int | None) -> int:
    ports = [requested] if requested is not None else range(7861, 7870)
    for port in ports:
        if port is None or not 1 <= port <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise ValueError("Requested local port is unavailable; choose another --port")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local speech interpretation V2")
    parser.add_argument("--check", action="store_true", help="Check resources without loading models")
    parser.add_argument("--json", action="store_true", help="Print checks as JSON")
    parser.add_argument("--ui-only", action="store_true", help="Allow UI/API inspection with missing models")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    try:
        configure_environment()
        checks = startup_report()
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    complete = all(c["status"] == "ok" for c in checks if c["required"])
    if args.check:
        if args.json:
            print(json.dumps({"ok": complete, "checks": checks, "models_loaded": False}, ensure_ascii=False, indent=2))
        else:
            for check in checks:
                print(f"[{check['status']}] {check['name']}: {check['detail']}")
        return 0 if complete else 2
    if not complete and not args.ui_only:
        print("Model resources are missing. Run --check and read docs/MODELS.md.", file=sys.stderr)
        return 2
    if args.ui_only and not complete:
        print("UI inspection only: real recognition/translation needs local model resources.")
    try:
        port = available_port(args.port)
        import uvicorn
        import v4_server
    except (ImportError, ValueError) as exc:
        print(f"Startup error: {exc}. See README.md for installation.", file=sys.stderr)
        return 2
    url = f"http://127.0.0.1:{port}"
    print(f"Local UI: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(v4_server.application, host="127.0.0.1", port=port, log_level="info")
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
