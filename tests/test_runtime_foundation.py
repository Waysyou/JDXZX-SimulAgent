import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
from src.app_config import RuntimeSettings
from src.contracts import (
    audio_queue_item,
    segment_key,
    translation_record,
    tts_record,
    validate_segment,
)
from src.runtime_support import (
    checks_ok,
    configure_logging,
    remove_runtime_path,
    run_startup_checks,
    session_directory,
)
from src.stream_asr import StreamingASR


class RuntimeFoundationTests(unittest.TestCase):
    def test_settings_parse_validated_environment_values(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {
                    "SIMUL_STREAM_BLOCK_SECONDS": "0.25",
                    "SIMUL_MAX_AUDIO_QUEUE": "6",
                    "SIMUL_MAX_SESSIONS": "8",
                    "SIMUL_MAX_FILE_UPLOAD_BYTES": "12345678",
                },
                clear=False,
            ):
                settings = RuntimeSettings.from_env(directory)
        self.assertEqual(settings.stream_block_seconds, 0.25)
        self.assertEqual(settings.max_audio_queue, 6)
        self.assertEqual(settings.max_sessions, 8)
        self.assertEqual(settings.max_file_upload_bytes, 12_345_678)

    def test_settings_reject_invalid_numeric_values(self):
        with patch.dict(
            os.environ,
            {"SIMUL_STREAM_BLOCK_SECONDS": "not-a-number"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "SIMUL_STREAM_BLOCK_SECONDS"):
                RuntimeSettings.from_env(".")

    def test_segment_contract_helpers_keep_ids_aligned(self):
        segment = validate_segment(
            {
                "segment_id": "7",
                "start": 1.25,
                "end": 2.5,
                "text": "你好",
                "status": "FINAL",
            }
        )
        self.assertEqual(segment_key(segment["segment_id"]), "7")
        self.assertEqual(
            translation_record("你好", "Hello.", "ACCEPTED")["source"],
            segment["text"],
        )
        self.assertEqual(audio_queue_item(7, "segment.wav")["segment_id"], 7)
        self.assertEqual(tts_record("segment.wav", 1.0, 2.0)["audio_end"], 2.0)

    def test_segment_contract_rejects_invalid_timestamps(self):
        with self.assertRaisesRegex(ValueError, "earlier"):
            validate_segment(
                {
                    "segment_id": 1,
                    "start": 2.0,
                    "end": 1.0,
                    "text": "bad",
                    "status": "FINAL",
                }
            )
        with self.assertRaisesRegex(ValueError, "earlier"):
            tts_record("segment.wav", 2.0, 1.0)

    def test_runtime_path_removal_cannot_escape_cache_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = RuntimeSettings(project_root=root)
            inside = session_directory(settings, "live_tts", "session_1")
            inside.mkdir(parents=True)
            (inside / "segment.wav").write_bytes(b"wav")
            outside = root / "do-not-delete.txt"
            outside.write_text("keep", encoding="utf-8")

            self.assertTrue(remove_runtime_path(settings, inside))
            self.assertFalse(inside.exists())
            self.assertFalse(remove_runtime_path(settings, outside))
            self.assertTrue(outside.exists())

    def test_startup_checks_are_readable_and_model_free(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sherpa = root / "models" / "sherpa"
            qwen = root / "models" / "qwen"
            piper = root / "models" / "voice.onnx"
            espeak = root / "runtime" / "espeak"
            ui = root / "ui"
            sherpa.mkdir(parents=True)
            qwen.mkdir(parents=True)
            espeak.mkdir(parents=True)
            ui.mkdir(parents=True)
            for name in (
                "encoder-test.onnx",
                "decoder-test.onnx",
                "joiner-test.onnx",
                "tokens.txt",
            ):
                (sherpa / name).write_bytes(b"test")
            (qwen / "config.json").write_text("{}", encoding="utf-8")
            (qwen / "model.safetensors").write_bytes(b"test")
            piper.write_bytes(b"test")
            piper.with_suffix(".onnx.json").write_text("{}", encoding="utf-8")
            (espeak / "phontab").write_bytes(b"test")
            (ui / "index.html").write_text("ok", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "SHERPA_MODEL_DIR": str(sherpa),
                    "QWEN_MODEL_DIR": str(qwen),
                    "PIPER_VOICE_MODEL": str(piper),
                    "PIPER_ESPEAK_DATA_DIR": str(espeak),
                },
                clear=False,
            ):
                settings = RuntimeSettings.from_env(root)
                checks = run_startup_checks(settings, ui_root=ui)

            self.assertTrue(checks_ok(checks))
            self.assertTrue(all(item["status"] == "ok" for item in checks))

    def test_logger_configuration_is_idempotent(self):
        first = configure_logging(app.SETTINGS)
        before = len(first.handlers)
        second = configure_logging(app.SETTINGS)
        self.assertIs(first, second)
        self.assertEqual(len(second.handlers), before)

    def test_streaming_asr_releases_native_session_without_loading_model(self):
        recognizer = StreamingASR()
        state = recognizer.new_incremental_state()
        session_id = state["sherpa_session_id"]
        recognizer._sherpa_sessions[session_id] = object()

        recognizer.release_incremental_state(state)

        self.assertNotIn(session_id, recognizer._sherpa_sessions)
        self.assertTrue(state["finalized"])

    def test_app_session_cleanup_releases_transport_and_generated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = RuntimeSettings(project_root=Path(directory))
            with patch.object(app, "SETTINGS", settings):
                state = app._new_live_state()
                control = app._register_file_transport(state["session_id"])
                source = Path(state["source_audio"])
                source.parent.mkdir(parents=True)
                source.write_bytes(b"source")
                tts_dir = session_directory(
                    settings, "live_tts", state["session_id"]
                )
                tts_dir.mkdir(parents=True)
                (tts_dir / "segment.wav").write_bytes(b"tts")

                app.release_live_session_resources(state, remove_cache=True)

                self.assertTrue(control["stopped"].is_set())
                self.assertIsNone(app._file_transport(state["session_id"]))
                self.assertFalse(source.parent.exists())
                self.assertFalse(tts_dir.exists())
                self.assertTrue(state["asr_state"]["finalized"])


if __name__ == "__main__":
    unittest.main()
