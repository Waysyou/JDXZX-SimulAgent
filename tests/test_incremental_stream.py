import unittest
import threading
import time
import tempfile
from unittest.mock import patch
from pathlib import Path

import numpy as np
import soundfile as sf

from src.realtime_input import iter_audio_blocks, load_audio_16k
from src.stream_asr import StreamingASR, _to_mono_16k


class _Segment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


class _FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio, **kwargs):
        self.calls += 1
        seconds = len(audio) / 16000
        if seconds < 0.2:
            return iter(()), None
        text = "今天我们介绍实时同传"
        return iter([_Segment(0.1, min(seconds, 1.1), text)]), None


class _FakeHallucinationModel(_FakeModel):
    def transcribe(self, audio, **kwargs):
        self.calls += 1
        seconds = len(audio) / 16000
        return iter(
            [_Segment(0.1, min(seconds, 1.1), "请点赞订阅转发")]
        ), None


class IncrementalStreamTests(unittest.TestCase):
    def test_file_input_is_split_into_strict_streaming_blocks(self):
        samples = np.arange(17600, dtype=np.float32)
        blocks = list(iter_audio_blocks(samples, block_seconds=0.4))

        self.assertEqual([len(block) for _, block in blocks], [6400, 6400, 4800])
        rebuilt = np.concatenate([block for _, block in blocks])
        np.testing.assert_array_equal(rebuilt, samples)
        self.assertTrue(all(sample_rate == 16000 for sample_rate, _ in blocks))

    def test_uploaded_stereo_audio_is_loaded_as_mono_16k(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "source.wav"
            left = np.linspace(-0.5, 0.5, 48000, dtype=np.float32)
            right = np.linspace(0.5, -0.5, 48000, dtype=np.float32)
            sf.write(path, np.column_stack([left, right]), 48000)

            normalized = load_audio_16k(path)

        self.assertEqual(normalized.dtype, np.float32)
        self.assertEqual(len(normalized), 16000)
        self.assertLess(float(np.max(np.abs(normalized))), 1e-3)

    def test_sherpa_native_calls_are_serialized(self):
        asr = StreamingASR()
        active = 0
        max_active = 0
        counter_lock = threading.Lock()

        def fake_unlocked(audio, state, *, final):
            nonlocal active, max_active
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.04)
            with counter_lock:
                active -= 1
            return {"state": state, "final": final}

        asr._transcribe_sherpa_incremental_unlocked = fake_unlocked
        state = asr.new_incremental_state()
        threads = [
            threading.Thread(
                target=asr._transcribe_sherpa_incremental,
                args=(None, state),
                kwargs={"final": index == 1},
            )
            for index in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(max_active, 1)

    def test_late_sherpa_chunk_after_finalization_is_ignored(self):
        asr = StreamingASR()
        asr.live_backend = "sherpa"
        state = asr.new_incremental_state()
        state["finalized"] = True
        state["total_samples"] = 16000

        with patch.object(
            asr,
            "load_live",
            side_effect=AssertionError("must not recreate a Sherpa stream"),
        ):
            result = asr.transcribe_incremental(
                (16000, np.ones(6400, dtype=np.float32)),
                state,
            )

        self.assertEqual(result["new_stable"], [])
        self.assertEqual(result["elapsed"], 1.0)
        self.assertTrue(result["state"]["finalized"])

    def test_integer_microphone_pcm_is_normalized(self):
        mono = np.array([-32768, 0, 32767], dtype=np.int16)
        normalized = _to_mono_16k(mono, 16000)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertAlmostEqual(float(normalized[0]), -1.0, places=5)
        self.assertAlmostEqual(float(normalized[1]), 0.0, places=5)
        self.assertAlmostEqual(float(normalized[2]), 32767 / 32768, places=5)

        stereo = np.array([[-32768, -32768], [32767, 32767]], dtype=np.int16)
        stereo_normalized = _to_mono_16k(stereo, 16000)
        self.assertAlmostEqual(float(stereo_normalized[0]), -1.0, places=5)
        self.assertAlmostEqual(float(stereo_normalized[1]), 32767 / 32768, places=5)

    def test_float_file_audio_is_preserved_and_sanitized(self):
        audio = np.array([-0.5, np.nan, 0.5, np.inf], dtype=np.float32)
        normalized = _to_mono_16k(audio, 16000)
        np.testing.assert_allclose(normalized, [-0.5, 0.0, 0.5, 1.0])

    def test_coerced_audio_reports_the_resampled_rate(self):
        audio_48k = np.zeros(48000, dtype=np.float32)
        sample_rate, audio_16k = StreamingASR._coerce_stream_audio(
            (48000, audio_48k)
        )
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(len(audio_16k), 16000)

        # The web callback normalizes once before the ASR adapter sees the
        # block. A second safety normalization must therefore be idempotent.
        repeated_rate, repeated_audio = StreamingASR._coerce_stream_audio(
            (sample_rate, audio_16k)
        )
        self.assertEqual(repeated_rate, 16000)
        self.assertEqual(len(repeated_audio), 16000)

    def test_microphone_blocks_are_processed_incrementally(self):
        asr = StreamingASR()
        asr.live_backend = "whisper"
        asr._model = _FakeModel()
        state = asr.new_incremental_state(process_every_seconds=0.5)

        speech = np.full(8000, 0.05, dtype=np.float32)
        first = asr.transcribe_incremental((16000, speech), state)
        for _ in range(3):
            asr.transcribe_incremental((16000, speech), state)
        partial = asr.transcribe_incremental((16000, speech), state)

        self.assertEqual(first["items"], [])
        self.assertEqual(len(partial["items"]), 1)
        self.assertEqual(partial["items"][0]["status"], "PARTIAL")
        self.assertEqual(partial["new_stable"], [])

        endpoint = asr.transcribe_incremental(
            (16000, np.zeros(19200, dtype=np.float32)),
            state,
        )
        self.assertEqual(len(endpoint["items"]), 1)
        self.assertEqual(endpoint["items"][0]["status"], "FINAL")
        self.assertEqual(len(endpoint["new_stable"]), 1)
        self.assertEqual(state["endpoint_count"], 1)
        self.assertGreaterEqual(len(state["window_records"]), 1)

    def test_overlapping_windows_reuse_the_same_segment_id(self):
        asr = StreamingASR()
        asr.live_backend = "whisper"
        asr._model = _FakeModel()
        state = asr.new_incremental_state(process_every_seconds=0.5)
        speech = np.full(8000, 0.05, dtype=np.float32)

        for _ in range(5):
            result = asr.transcribe_incremental((16000, speech), state)

        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["segment_id"], 1)
        self.assertEqual(state["next_segment_id"], 2)

        final = asr.transcribe_incremental(None, state, final=True)
        self.assertTrue(final["state"]["finalized"])
        self.assertEqual(final["state"]["segments"][0]["status"], "FINAL")
        self.assertEqual(final["state"]["segments"][0]["segment_id"], 1)
        self.assertEqual(len(final["new_stable"]), 1)

    def test_silent_windows_do_not_invoke_whisper(self):
        asr = StreamingASR()
        asr.live_backend = "whisper"
        model = _FakeModel()
        asr._model = model
        state = asr.new_incremental_state(process_every_seconds=0.5)

        for _ in range(4):
            result = asr.transcribe_incremental(
                (16000, np.zeros(8000, dtype=np.float32)),
                state,
            )

        self.assertEqual(model.calls, 0)
        self.assertEqual(result["items"], [])
        self.assertTrue(state["window_records"][-1]["skipped_as_silence"])

    def test_known_subtitle_hallucinations_are_rejected(self):
        self.assertTrue(
            StreamingASR._looks_like_hallucination(
                "请不吝点赞订阅转发打赏支持明镜与点点栏目"
            )
        )
        self.assertFalse(StreamingASR._looks_like_hallucination("不不不"))
        self.assertFalse(StreamingASR._looks_like_hallucination("喂喂喂"))
        self.assertFalse(StreamingASR._looks_like_hallucination("一二三四五六七"))

    def test_rejected_candidate_remains_visible_but_is_not_committed(self):
        asr = StreamingASR()
        asr.live_backend = "whisper"
        asr._model = _FakeHallucinationModel()
        state = asr.new_incremental_state()
        speech = np.full(24000, 0.05, dtype=np.float32)
        asr.transcribe_incremental((16000, speech), state)
        result = asr.transcribe_incremental(
            (16000, np.zeros(19200, dtype=np.float32)),
            state,
        )

        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["status"], "REJECTED")
        self.assertEqual(result["new_stable"], [])
        self.assertEqual(state["rejected_utterances"], 1)

    def test_local_agreement_only_confirms_shared_prefix(self):
        state = StreamingASR.new_incremental_state()
        first = StreamingASR._update_local_agreement(
            state,
            "今天我们介绍",
            final=False,
        )
        second = StreamingASR._update_local_agreement(
            state,
            "今天我们介绍新的",
            final=False,
        )
        third = StreamingASR._update_local_agreement(
            state,
            "今天我们介绍新的系统",
            final=False,
        )
        final = StreamingASR._update_local_agreement(
            state,
            "今天我们介绍新的系统",
            final=True,
        )

        self.assertEqual(first, "")
        self.assertTrue("今天我们介绍".startswith(second))
        self.assertTrue(third.startswith(second))
        self.assertEqual(final, "今天我们介绍新的系统")

    def test_sherpa_commit_policy_prefers_semantic_boundaries(self):
        with patch.dict(
            "os.environ",
            {
                "SHERPA_COMMIT_MIN_CHARS": "14",
                "SHERPA_COMMIT_MAX_CHARS": "22",
                "SHERPA_COMMIT_MIN_SECONDS": "1.0",
            },
        ):
            self.assertTrue(
                StreamingASR._should_commit_sherpa_delta(
                    "今天我们介绍新的系统。",
                    endpoint=False,
                    elapsed_since_emit=0.4,
                )
            )
            self.assertFalse(
                StreamingASR._should_commit_sherpa_delta(
                    "今天我们介绍新的系统和",
                    endpoint=False,
                    elapsed_since_emit=0.8,
                )
            )
            self.assertTrue(
                StreamingASR._should_commit_sherpa_delta(
                    "今天我们介绍新的系统和相关研究成果",
                    endpoint=False,
                    elapsed_since_emit=1.2,
                )
            )
            self.assertTrue(
                StreamingASR._should_commit_sherpa_delta(
                    "短句",
                    endpoint=True,
                    elapsed_since_emit=0.2,
                )
            )

    def test_audio_quality_reports_low_level_and_clipping(self):
        state = StreamingASR.new_incremental_state()
        quality = StreamingASR._update_audio_quality(
            state, np.full(6400, 0.001, dtype=np.float32)
        )
        self.assertEqual(quality["label"], "音量过低")

        state = StreamingASR.new_incremental_state()
        quality = StreamingASR._update_audio_quality(
            state, np.ones(6400, dtype=np.float32)
        )
        self.assertEqual(quality["label"], "存在削波")

    def test_adaptive_commit_uses_larger_units_under_tts_pressure(self):
        with patch.dict(
            "os.environ",
            {
                "SHERPA_COMMIT_MIN_CHARS": "14",
                "SHERPA_COMMIT_MAX_CHARS": "22",
                "SHERPA_COMMIT_MIN_SECONDS": "1.0",
            },
        ):
            normal = StreamingASR._adaptive_commit_thresholds(
                speech_rate_cps=4.0, queue_pressure=0.0
            )
            congested = StreamingASR._adaptive_commit_thresholds(
                speech_rate_cps=4.0, queue_pressure=1.0
            )
        self.assertGreater(congested[0], normal[0])
        self.assertGreater(congested[1], normal[1])
        self.assertGreater(congested[2], normal[2])


if __name__ == "__main__":
    unittest.main()
