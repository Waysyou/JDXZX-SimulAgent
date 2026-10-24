import threading
import queue
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import app


class AppControlTests(unittest.TestCase):
    def test_live_translation_rows_keep_bilingual_unit_aligned(self):
        state = app._new_live_state()
        state["asr_state"]["segments"] = [
            {
                "segment_id": 7,
                "start": 1.25,
                "end": 3.5,
                "text": "这是中文",
                "status": "FINAL",
            }
        ]
        state["translations"]["7"] = {"text": "This is English."}

        self.assertEqual(
            app._live_translation_rows(state),
            [["1.25s", "这是中文", "This is English.", 7]],
        )

    def test_live_translation_html_is_stable_and_escaped(self):
        state = app._new_live_state()
        state["asr_state"]["segments"] = [
            {
                "segment_id": 3,
                "start": 0.5,
                "end": 1.5,
                "text": "中文 <测试>",
                "status": "FINAL",
            }
        ]
        state["translations"]["3"] = {"text": "English <unit>."}
        state["tts_paths"]["3"] = {"path": "segment.wav"}

        rendered = app._render_live_translation(state)

        self.assertIn('class="v3-unit-row has-audio"', rendered)
        self.assertIn('data-segment-id="3"', rendered)
        self.assertIn("中文 &lt;测试&gt;", rendered)
        self.assertIn("English &lt;unit&gt;.", rendered)
        self.assertNotIn("<测试>", rendered)

    def test_live_translation_html_has_actionable_empty_state(self):
        rendered = app._render_live_translation(app._new_live_state())
        self.assertIn("等待开始同声传译", rendered)
        self.assertNotIn("<table", rendered)

    def test_uploaded_transport_pause_resume_and_stop(self):
        session_id = "transport-test"
        state = {"session_id": session_id}
        control = app._register_file_transport(session_id)
        try:
            app.pause_uploaded_stream(state)
            self.assertTrue(control["paused"].is_set())
            app.resume_uploaded_stream(state)
            self.assertFalse(control["paused"].is_set())
            app.stop_uploaded_stream(state)
            self.assertTrue(control["stopped"].is_set())
        finally:
            app._release_file_transport(session_id)

    def test_stop_recording_callback_returns_before_worker_finishes(self):
        state = app._new_live_state()
        worker_started = threading.Event()
        worker_release = threading.Event()

        def slow_worker(worker_state):
            worker_started.set()
            worker_release.wait(2.0)
            worker_state["finalization_status"] = "DONE"

        try:
            with patch.object(app, "_finalize_microphone_worker", slow_worker):
                started = time.perf_counter()
                outputs = app.start_microphone_finalization(state)
                elapsed = time.perf_counter() - started
                self.assertTrue(worker_started.wait(0.5))
                self.assertLess(elapsed, 0.25)
                self.assertEqual(len(outputs), 9)
                self.assertEqual(state["finalization_status"], "RUNNING")
        finally:
            worker_release.set()
            with app._FINALIZATION_LOCK:
                thread = app._FINALIZATION_THREADS.pop(state["session_id"], None)
            if thread:
                thread.join(timeout=2.0)

    def test_background_finalizer_contains_errors(self):
        state = app._new_live_state()
        with (
            patch.object(
                app.asr,
                "transcribe_incremental",
                side_effect=RuntimeError("synthetic ASR failure"),
            ),
            patch.object(
                app,
                "_write_finalization_log",
                return_value=Path("synthetic.log"),
            ),
        ):
            app._finalize_microphone_worker(state)

        self.assertEqual(state["finalization_status"], "ERROR")
        self.assertIn("synthetic ASR failure", state["last_error"])
        self.assertIn("synthetic.log", state["finalization_message"])

    def test_tts_backpressure_never_drops_confirmed_segments(self):
        state = app._new_live_state()
        state["max_audio_queue"] = 1
        state["audio_queue"] = [{"segment_id": 99, "path": "existing.wav"}]
        segments = [
            {"segment_id": 1, "text": "你好", "status": "FINAL"},
            {"segment_id": 2, "text": "欢迎", "status": "FINAL"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(app, "PROJECT_ROOT", Path(temp_dir)),
                patch.object(
                    app.translator,
                    "translate_many",
                    return_value=["Hello.", "Welcome."],
                ),
                patch.object(app.tts, "synthesize"),
                patch.object(app, "_audio_duration", return_value=0.5),
            ):
                app._queue_live_tts_unlocked(state, segments)

        self.assertEqual(
            [item["segment_id"] for item in state["audio_queue"]],
            [99, 1, 2],
        )
        self.assertEqual(state["dropped_audio_ids"], [])
        self.assertEqual(state["audio_queue_high_watermark"], 3)

    def test_rejected_translation_can_retry_with_a_bounded_attempt_count(self):
        state = app._new_live_state()
        segments = [
            {"segment_id": 1, "text": "这是一个稳定片段", "status": "FINAL"}
        ]
        batch_calls = 0
        single_calls = 0

        def translate_many(texts):
            nonlocal batch_calls
            batch_calls += 1
            if batch_calls == 1:
                return ["这不是英文"]
            return ["This is a stable segment."]

        def translate_one(text):
            nonlocal single_calls
            single_calls += 1
            return "仍然不是英文"

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    app,
                    "session_directory",
                    side_effect=lambda _settings, category, session_id: (
                        Path(temp_dir) / category / session_id
                    ),
                ),
                patch.object(
                    app.translator,
                    "translate_many",
                    side_effect=translate_many,
                ),
                patch.object(
                    app.translator,
                    "translate",
                    side_effect=translate_one,
                ),
                patch.object(app.tts, "synthesize"),
                patch.object(app, "_audio_duration", return_value=0.5),
            ):
                app._queue_live_tts_unlocked(state, segments)
                self.assertEqual(state["translations"]["1"]["quality_status"], "REJECTED")
                self.assertNotIn("1", state["translation_sources"])
                self.assertIn("1", state["translation_retry_segments"])
                self.assertEqual(state["translation_attempts"]["1"], 2)

                app._queue_live_tts_unlocked(state, segments)

        self.assertEqual(state["translations"]["1"]["quality_status"], "ACCEPTED")
        self.assertEqual(state["translation_sources"]["1"], "这是一个稳定片段")
        self.assertNotIn("1", state["translation_retry_segments"])
        self.assertNotIn(1, state["failed_translation_ids"])
        self.assertEqual(state["translation_attempts"]["1"], 3)
        self.assertEqual(batch_calls, 2)
        self.assertEqual(single_calls, 1)
        self.assertIn("1", state["tts_paths"])

    def test_exhausted_translation_uses_english_degraded_fallback(self):
        state = app._new_live_state()
        segments = [
            {"segment_id": 8, "text": "无法通过质量校验的稳定片段", "status": "FINAL"}
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    app,
                    "session_directory",
                    side_effect=lambda _settings, category, session_id: (
                        Path(temp_dir) / category / session_id
                    ),
                ),
                patch.object(
                    app.translator,
                    "translate_many",
                    return_value=["这不是英文"],
                ),
                patch.object(
                    app.translator,
                    "translate",
                    return_value="仍然不是英文",
                ),
                patch.object(app.tts, "synthesize"),
                patch.object(app, "_audio_duration", return_value=0.5),
            ):
                app._queue_live_tts_unlocked(state, segments)
                self.assertEqual(
                    state["translations"]["8"]["quality_status"],
                    "REJECTED",
                )
                app._queue_live_tts_unlocked(state, segments)

        record = state["translations"]["8"]
        self.assertEqual(record["quality_status"], "DEGRADED")
        self.assertNotIn("8", state["translation_retry_segments"])
        self.assertNotIn(8, state["failed_translation_ids"])
        self.assertIn(8, state["degraded_translation_ids"])
        self.assertEqual(state["translation_sources"]["8"], segments[0]["text"])
        self.assertIn("The speaker", record["text"])
        self.assertIn("8", state["tts_paths"])

    def test_degraded_translation_is_included_in_complete_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "outputs"
            source_audio = root / "speech.wav"
            segment_audio = root / "segment_0001.wav"
            source_audio.touch()
            segment_audio.touch()
            stable_segments = [
                {
                    "segment_id": 1,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "片段",
                    "status": "FINAL",
                }
            ]
            with (
                patch.object(app, "PROJECT_ROOT", root),
                patch.object(app, "LIVE_OUTPUT_DIR", output_dir),
                patch.object(app, "_combine_wavs"),
                patch.object(app, "_audio_duration", return_value=0.5),
            ):
                metadata = app._save_live_results(
                    str(source_audio),
                    stable_segments,
                    {1: ("片段", "The speaker continues with this point.")},
                    {1: str(segment_audio)},
                    [{"window_latency": 0.2}],
                    {"degraded_translation_ids": [1]},
                )
                self.assertTrue((output_dir / "speech_translation.txt").exists())

        self.assertEqual(metadata["stable_segments"], 1)
        self.assertEqual(metadata["degraded_translation_ids"], [1])

    def test_file_inference_worker_returns_results_without_blocking_ingest(self):
        state = app._new_live_state("file", "sample.wav")
        pending = queue.Queue()
        completed = queue.Queue()
        worker_started = threading.Event()
        worker_release = threading.Event()

        def slow_inference(worker_state, segments):
            worker_started.set()
            worker_release.wait(1.0)
            worker_state["translations"]["1"] = {
                "source": "你好",
                "text": "Hello.",
                "quality_status": "ACCEPTED",
            }
            worker_state["translation_sources"]["1"] = "你好"
            worker_state["tts_paths"]["1"] = {
                "path": "segment.wav",
                "audio_start": 0.0,
                "audio_end": 0.5,
            }
            worker_state["audio_queue"].append(
                {"segment_id": 1, "path": "segment.wav"}
            )

        with patch.object(app, "_queue_live_tts", side_effect=slow_inference):
            worker = threading.Thread(
                target=app._file_inference_worker,
                args=(state, pending, completed),
                daemon=True,
            )
            worker.start()
            started = time.perf_counter()
            pending.put(
                [{"segment_id": 1, "text": "你好", "status": "FINAL"}]
            )
            enqueue_elapsed = time.perf_counter() - started
            self.assertTrue(worker_started.wait(0.25))
            self.assertLess(enqueue_elapsed, 0.05)
            self.assertNotIn("1", state["translations"])

            worker_release.set()
            pending.join()
            self.assertEqual(app._apply_file_inference_results(state, completed), 1)
            self.assertEqual(state["translations"]["1"]["text"], "Hello.")
            self.assertEqual(state["audio_cursor"], 0.5)
            self.assertEqual(state["audio_queue"][0]["segment_id"], 1)

            pending.put(None)
            pending.join()
            worker.join(timeout=1.0)
            self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main()
