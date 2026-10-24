from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from src.file_pipeline import run_file_pipeline
from src.tts import DEFAULT_ESPEAK_DATA_DIR, PiperTTS, _native_safe_espeak_data_dir


class _FakeStreamingASR:
    live_backend = "sherpa"

    def load_live(self):
        self.loaded = True

    def new_incremental_state(self, **_kwargs):
        return {"emitted": False}

    def transcribe_incremental(self, _audio, state, *, final=False):
        new_stable = []
        if not final and not state["emitted"]:
            state["emitted"] = True
            new_stable = [
                {
                    "segment_id": 1,
                    "start": 0.0,
                    "end": 0.4,
                    "text": "测试中文片段",
                    "status": "FINAL",
                }
            ]
        stable = [
            {
                "segment_id": 1,
                "start": 0.0,
                "end": 0.4,
                "text": "测试中文片段",
                "status": "FINAL",
            }
        ] if state["emitted"] else []
        return {
            "state": state,
            "items": stable,
            "stable": stable,
            "new_stable": new_stable,
            "window_start": 0.0,
            "window_end": 0.4,
            "window_latency": 0.01,
        }


class _FakeTranslator:
    device = "cpu"
    load_seconds = 0.0

    def translate_many(self, texts):
        return ["This is a test segment." for _text in texts]


class _FakeTTS:
    length_scale = 0.75

    def synthesize(self, _text, output_path):
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 160)
        return Path(output_path)


class FilePipelineTests(unittest.TestCase):
    def test_submission_artifacts_use_streaming_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "speech.wav"
            sf.write(source, np.zeros(8000, dtype=np.float32), 16000)
            output = root / "out"
            with (
                patch("src.file_pipeline.StreamingASR", _FakeStreamingASR),
                patch("src.file_pipeline.build_translator", _FakeTranslator),
                patch("src.file_pipeline.PiperTTS", _FakeTTS),
            ):
                metadata = run_file_pipeline(source, output, block_seconds=0.4)

            translation = output / "speech_translation.txt"
            tts = output / "speech_translation_tts.wav"
            self.assertEqual(
                translation.read_text(encoding="utf-8"),
                "This is a test segment.\n",
            )
            self.assertTrue(tts.exists())
            self.assertEqual(metadata["asr_backend"], "sherpa")
            self.assertEqual(metadata["stream_block_seconds"], 0.4)

    def test_piper_default_espeak_data_is_inside_project(self):
        with patch.dict("os.environ", {"PIPER_ESPEAK_DATA_DIR": ""}):
            self.assertEqual(PiperTTS().espeak_data_dir, DEFAULT_ESPEAK_DATA_DIR)

    @unittest.skip(
        "Official-reference transcript mode is intentionally excluded from "
        "the frozen competition runtime and V2 inference chain"
    )
    def test_reference_transcript_generates_timed_submission_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "speech.wav"
            sf.write(source, np.zeros(32000, dtype=np.float32), 16000)
            reference = root / "speech_transcription_unchunk.txt"
            reference.write_text("第一句。第二句，后半句。第三句！", encoding="utf-8")
            output = root / "out"
            with (
                patch("src.file_pipeline.build_translator", _FakeTranslator),
                patch("src.file_pipeline.PiperTTS", _FakeTTS),
            ):
                metadata = run_file_pipeline(
                    source,
                    output,
                    block_seconds=0.4,
                    reference_transcript_path=reference,
                )

            translation = (output / "speech_translation.txt").read_text(encoding="utf-8")
            self.assertEqual(
                translation,
                "English: 第一句。\nEnglish: 第二句，后半句。\nEnglish: 第三句！\n",
            )
            self.assertEqual(metadata["asr_backend"], "official_reference_transcript")
            self.assertEqual(metadata["stable_segments"], 3)
            self.assertEqual(metadata["reference_transcript"], str(reference))

    def test_piper_non_ascii_project_path_is_staged_to_ascii_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "语音资源"
            source.mkdir()
            (source / "phontab").write_bytes(b"test")
            cache = root / "piper_runtime_cache"
            with patch.dict("os.environ", {"PIPER_ESPEAK_CACHE_DIR": str(cache)}):
                resolved = _native_safe_espeak_data_dir(source)
            self.assertEqual(resolved, cache)
            self.assertTrue((resolved / "phontab").exists())
            self.assertTrue(str(resolved).isascii())


if __name__ == "__main__":
    unittest.main()
