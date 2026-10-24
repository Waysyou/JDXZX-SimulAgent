from __future__ import annotations

import os
import threading
import time
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Generator

import numpy as np
import soundfile as sf
from .text_stabilizer import normalize_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / ".cache" / "models" / "faster-whisper-large-v3-turbo"
DEFAULT_FUNASR_MODEL_DIR = PROJECT_ROOT / ".cache" / "models" / "funasr-paraformer-zh-streaming"
DEFAULT_SHERPA_MODEL_DIR = (
    PROJECT_ROOT
    / ".cache"
    / "models"
    / "sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30"
)


def _to_mono_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert an input waveform to mono float32 at 16 kHz."""
    source_dtype = audio.dtype
    if np.issubdtype(source_dtype, np.signedinteger):
        info = np.iinfo(source_dtype)
        scale = float(max(abs(info.min), info.max))
        audio = audio.astype(np.float32, copy=False) / scale
    elif np.issubdtype(source_dtype, np.unsignedinteger):
        info = np.iinfo(source_dtype)
        midpoint = float(info.max + 1) / 2.0
        audio = (audio.astype(np.float32, copy=False) - midpoint) / midpoint
    else:
        audio = audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)
    if sample_rate == 16000:
        return audio

    from scipy.signal import resample_poly

    import math

    divisor = math.gcd(sample_rate, 16000)
    up = 16000 // divisor
    down = sample_rate // divisor
    return resample_poly(audio, up, down).astype(np.float32, copy=False)


class StreamingASR:
    """Chunked and incremental ASR wrapper.

    ``transcribe_stream`` is retained for deterministic file validation.  The
    live web path uses ``transcribe_incremental``: it accepts one microphone
    block at a time, keeps only a short tail for context, and reconciles
    overlapping Whisper hypotheses into stable timestamped segments.
    """

    def __init__(self, model_dir: str | os.PathLike[str] | None = None) -> None:
        self.model_dir = Path(model_dir or os.environ.get("ASR_MODEL_DIR", DEFAULT_MODEL_DIR))
        # Keep the legacy Whisper fallback optional.  The production path uses
        # Sherpa streaming ASR, so importing faster-whisper at module import
        # time would make an otherwise unused package a release dependency.
        self._model: Any | None = None
        self.live_backend = os.environ.get("ASR_LIVE_BACKEND", "whisper").lower()
        self.live_model_dir = Path(
            os.environ.get("ASR_LIVE_MODEL_DIR", DEFAULT_FUNASR_MODEL_DIR)
        )
        self.sherpa_model_dir = Path(
            os.environ.get("SHERPA_MODEL_DIR", DEFAULT_SHERPA_MODEL_DIR)
        )
        self._live_model = None
        self._sherpa_sessions: dict[str, object] = {}
        # sherpa-onnx's OnlineStream/OnlineRecognizer native objects are not
        # safe to enter concurrently from Gradio's stream and stop callbacks.
        # A process-wide lock prevents a native 0xc0000005 access violation.
        self._sherpa_native_lock = threading.RLock()

    def load(self) -> Any:
        if self._model is None:
            if not self.model_dir.exists():
                raise FileNotFoundError(f"ASR model directory does not exist: {self.model_dir}")
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "The optional Whisper fallback is not installed. "
                    "Use the default Sherpa live path, or install faster-whisper."
                ) from exc
            self._model = WhisperModel(
                str(self.model_dir),
                device=os.environ.get("ASR_DEVICE", "cuda"),
                # INT8 weights with FP16 activations reduce VRAM pressure on the
                # 8 GB laptop GPU. Set ASR_COMPUTE_TYPE=float16 for quality
                # comparison after the streaming path is stable.
                compute_type=os.environ.get("ASR_COMPUTE_TYPE", "int8_float16"),
            )
        return self._model

    def load_live(self):
        """Load the configured live ASR backend, with an automatic fallback."""
        if self.live_backend == "sherpa":
            if self._live_model is None:
                import sherpa_onnx

                required = {
                    "tokens": self.sherpa_model_dir / "tokens.txt",
                    "encoder": self.sherpa_model_dir / "encoder.int8.onnx",
                    "decoder": self.sherpa_model_dir / "decoder.onnx",
                    "joiner": self.sherpa_model_dir / "joiner.int8.onnx",
                }
                missing = [str(path) for path in required.values() if not path.exists()]
                if missing:
                    raise FileNotFoundError(
                        "Sherpa streaming model files are missing: "
                        + ", ".join(missing)
                    )
                self._live_model = sherpa_onnx.OnlineRecognizer.from_transducer(
                    tokens=str(required["tokens"]),
                    encoder=str(required["encoder"]),
                    decoder=str(required["decoder"]),
                    joiner=str(required["joiner"]),
                    num_threads=int(os.environ.get("SHERPA_NUM_THREADS", "4")),
                    decoding_method=os.environ.get(
                        "SHERPA_DECODING_METHOD", "greedy_search"
                    ),
                    max_active_paths=int(
                        os.environ.get("SHERPA_MAX_ACTIVE_PATHS", "4")
                    ),
                    enable_endpoint_detection=True,
                    rule1_min_trailing_silence=float(
                        os.environ.get("SHERPA_RULE1_SILENCE", "2.4")
                    ),
                    rule2_min_trailing_silence=float(
                        os.environ.get("SHERPA_RULE2_SILENCE", "1.2")
                    ),
                    rule3_min_utterance_length=float(
                        os.environ.get("SHERPA_RULE3_LENGTH", "20.0")
                    ),
                    provider=os.environ.get("SHERPA_PROVIDER", "cpu"),
                )
            return self._live_model
        if self.live_backend == "auto":
            if self.live_model_dir.exists():
                try:
                    from funasr import AutoModel  # type: ignore

                    self.live_backend = "funasr"
                    self._live_model = AutoModel(
                        model=str(self.live_model_dir),
                        device=os.environ.get("ASR_DEVICE", "cuda"),
                        disable_update=True,
                    )
                    return self._live_model
                except Exception:
                    self.live_backend = "whisper"
            else:
                self.live_backend = "whisper"
        if self.live_backend == "funasr":
            if self._live_model is None:
                from funasr import AutoModel  # type: ignore

                if not self.live_model_dir.exists():
                    raise FileNotFoundError(
                        f"FunASR model directory does not exist: {self.live_model_dir}"
                    )
                self._live_model = AutoModel(
                    model=str(self.live_model_dir),
                    device=os.environ.get("ASR_DEVICE", "cuda"),
                    disable_update=True,
                )
            return self._live_model
        return self.load()

    def transcribe_stream(
        self,
        audio_path: str | os.PathLike[str],
        chunk_seconds: float = 10.0,
        overlap_seconds: float = 0.0,
    ) -> Generator[dict, None, None]:
        """Yield one result per simulated streaming audio window."""
        model = self.load()
        audio, sample_rate = sf.read(str(audio_path), dtype="float32")
        audio = _to_mono_16k(audio, sample_rate)
        sample_rate = 16000
        total_seconds = len(audio) / sample_rate
        chunk_size = int(chunk_seconds * sample_rate)
        overlap = int(overlap_seconds * sample_rate)
        step = max(1, chunk_size - overlap)

        started = time.perf_counter()
        segment_id = 0
        for window_start in range(0, len(audio), step):
            window_end = min(len(audio), window_start + chunk_size)
            window = audio[window_start:window_end]
            # A tiny overlap-only tail can make an autoregressive decoder
            # hallucinate for much longer than the audio itself. The preceding
            # window already covers most of this tail; the final version will
            # replace this guard with the text stabilizer/flush mechanism.
            if window_start > 0 and len(window) < int(1.5 * sample_rate):
                break

            recognition_started = time.perf_counter()
            segments, _ = model.transcribe(
                window,
                language="zh",
                beam_size=int(os.environ.get("ASR_BEAM_SIZE", "1")),
                best_of=int(os.environ.get("ASR_BEST_OF", "1")),
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                max_new_tokens=96,
                repetition_penalty=1.05,
            )
            items = []
            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                segment_id += 1
                start = window_start / sample_rate + float(segment.start)
                end = window_start / sample_rate + float(segment.end)
                items.append(
                    {
                        "segment_id": segment_id,
                        "start": round(start, 2),
                        "end": round(end, 2),
                        "text": text,
                    }
                )

            elapsed = time.perf_counter() - started
            yield {
                "window_start": round(window_start / sample_rate, 2),
                "window_end": round(window_end / sample_rate, 2),
                "total_seconds": round(total_seconds, 2),
                "items": items,
                "window_latency": round(time.perf_counter() - recognition_started, 2),
                "elapsed": round(elapsed, 2),
            }

    @staticmethod
    def new_incremental_state(
        *,
        context_seconds: float = 4.0,
        tail_seconds: float = 6.0,
        process_every_seconds: float = 0.8,
    ) -> dict:
        """Return a deepcopy/JSON-friendly state for one microphone session."""
        return {
            "sample_rate": 16000,
            "context_seconds": float(context_seconds),
            "tail_seconds": float(tail_seconds),
            "process_every_seconds": float(process_every_seconds),
            "tail": [],
            "total_samples": 0,
            "last_processed_samples": 0,
            "next_segment_id": 1,
            "segments": [],
            "window_records": [],
            "finalized": False,
            "funasr_cache": {},
            "funasr_pending": [],
            "funasr_transcript": "",
            # Endpoint controller. PARTIAL hypotheses are display-only; only a
            # completed utterance may become FINAL and enter translation/TTS.
            "vad_state": "IDLE",
            "noise_rms": 0.002,
            "pre_roll": [],
            "speech_run_frames": 0,
            "utterance_audio": [],
            "utterance_start_sample": None,
            "utterance_last_voiced_sample": 0,
            "utterance_voiced_samples": 0,
            "utterance_trailing_silence": 0,
            "active_utterance_id": None,
            "last_partial_decode_samples": 0,
            "endpoint_count": 0,
            "rejected_utterances": 0,
            "last_endpoint_reason": "",
            "audio_quality": {
                "sample_count": 0,
                "rms": 0.0,
                "peak": 0.0,
                "clip_ratio": 0.0,
                "dc_offset": 0.0,
                "label": "等待输入",
            },
            "tts_queue_pressure": 0.0,
            "commit_policy": {
                "speech_rate_cps": 0.0,
                "soft_chars": int(os.environ.get("SHERPA_COMMIT_MIN_CHARS", "14")),
                "hard_chars": int(os.environ.get("SHERPA_COMMIT_MAX_CHARS", "22")),
                "min_context_seconds": float(
                    os.environ.get("SHERPA_COMMIT_MIN_SECONDS", "1.0")
                ),
            },
            # Native Sherpa online recognizer state. The pybind OnlineStream is
            # held in StreamingASR._sherpa_sessions because Gradio state must
            # remain deepcopy/JSON friendly.
            "sherpa_session_id": uuid.uuid4().hex,
            "sherpa_previous_text": "",
            "sherpa_confirmed_text": "",
            "sherpa_emitted_text": "",
            "sherpa_partial_segment_id": None,
            "sherpa_utterance_start_sample": 0,
            "sherpa_last_emit_sample": 0,
        }

    def release_incremental_state(self, state: dict | None) -> None:
        """Drop native per-session objects without loading or resetting a model."""
        if not state:
            return
        session_id = state.get("sherpa_session_id")
        if session_id:
            with self._sherpa_native_lock:
                self._sherpa_sessions.pop(str(session_id), None)
        state["finalized"] = True

    @staticmethod
    def _coerce_stream_audio(
        audio: tuple[int, np.ndarray] | np.ndarray | None,
    ) -> tuple[int, np.ndarray] | None:
        if audio is None:
            return None
        if isinstance(audio, tuple) and len(audio) == 2:
            sample_rate, data = audio
        else:
            sample_rate, data = 16000, audio
        if data is None:
            return None
        array = np.asarray(data)
        if array.size == 0:
            return None
        # The returned waveform has already been converted to 16 kHz.  Keep
        # the tuple metadata consistent so callers can safely pass it through
        # this normalizer again without resampling the same block twice.
        return 16000, _to_mono_16k(array, int(sample_rate))

    @staticmethod
    def _find_incremental_match(
        existing: list[dict],
        candidate: dict,
    ) -> dict | None:
        """Find the old hypothesis represented by an overlapping new window."""
        best: dict | None = None
        best_score = -1.0
        for item in existing:
            overlap = min(item["end"], candidate["end"]) - max(
                item["start"], candidate["start"]
            )
            close_start = abs(item["start"] - candidate["start"]) <= 1.0
            close_end = abs(item["end"] - candidate["end"]) <= 1.2
            if overlap < -0.35 and not (close_start or close_end):
                continue
            text_score = 0.0
            old_text = normalize_text(item.get("text", ""))
            new_text = normalize_text(candidate.get("text", ""))
            if old_text == new_text:
                text_score = 2.0
            elif old_text and new_text and (
                old_text in new_text or new_text in old_text
            ):
                text_score = 1.0
            score = max(overlap, 0.0) + text_score
            if score > best_score:
                best, best_score = item, score
        return best

    @staticmethod
    def _window_speech_stats(audio: np.ndarray) -> tuple[float, float]:
        """Return full-window RMS and voiced-frame ratio for a 16 kHz signal."""
        if audio.size == 0:
            return 0.0, 0.0
        frame_size = 320  # 20 ms at 16 kHz
        usable = len(audio) - (len(audio) % frame_size)
        if usable <= 0:
            rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
            return rms, float(rms >= 0.008)
        frames = audio[:usable].reshape(-1, frame_size).astype(np.float64)
        frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
        return float(np.sqrt(np.mean(frames * frames))), float(np.mean(frame_rms >= 0.008))

    @staticmethod
    def _update_audio_quality(state: dict, audio: np.ndarray) -> dict:
        """Accumulate conservative input diagnostics without altering speech."""
        quality = state.setdefault("audio_quality", {})
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return quality
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        peak = float(np.max(np.abs(samples)))
        dc_offset = float(np.mean(samples, dtype=np.float64))
        clipped = int(np.count_nonzero(np.abs(samples) >= 0.995))
        old_count = int(quality.get("sample_count", 0))
        new_count = old_count + int(samples.size)
        old_clip_count = int(round(float(quality.get("clip_ratio", 0.0)) * old_count))
        previous_rms = float(quality.get("rms", 0.0))
        previous_dc = float(quality.get("dc_offset", 0.0))
        smoothing = 1.0 if old_count == 0 else 0.15
        smoothed_rms = (1.0 - smoothing) * previous_rms + smoothing * rms
        smoothed_dc = (1.0 - smoothing) * previous_dc + smoothing * dc_offset
        clip_ratio = (old_clip_count + clipped) / max(1, new_count)
        if smoothed_rms < 0.003:
            label = "音量过低"
        elif clip_ratio >= 0.005 or peak >= 0.999:
            label = "存在削波"
        elif smoothed_rms > 0.25:
            label = "音量过高"
        else:
            label = "音量正常"
        quality.update(
            {
                "sample_count": new_count,
                "rms": round(smoothed_rms, 5),
                "peak": round(max(float(quality.get("peak", 0.0)), peak), 5),
                "clip_ratio": round(clip_ratio, 6),
                "dc_offset": round(smoothed_dc, 6),
                "label": label,
            }
        )
        return quality

    @staticmethod
    def _hypothesis_similarity(left: str, right: str) -> float:
        left = normalize_text(left)
        right = normalize_text(right)
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        if left in right or right in left:
            return min(len(left), len(right)) / max(len(left), len(right))
        return SequenceMatcher(None, left, right).ratio()

    @staticmethod
    def _finalize_best_hypotheses(state: dict) -> None:
        """Choose the strongest observed text before committing on stop."""
        for item in state.get("segments", []):
            history = item.get("hypothesis_history", [])
            if history:
                best = max(
                    history,
                    key=lambda value: (
                        float(value.get("confidence", -10.0)),
                        float(value.get("voiced_ratio", 0.0)),
                        -len(value.get("text", "")),
                    ),
                )
                item["text"] = best["text"]
                item["raw_text"] = best.get("raw_text", best["text"])
            item["status"] = "FINAL"
        state["finalized"] = True

    @staticmethod
    def _looks_like_hallucination(text: str) -> bool:
        """Reject recurring subtitle boilerplate and degenerate repetitions."""
        compact = normalize_text(text).replace(" ", "")
        if not compact:
            return True
        known_boilerplate = (
            "请不吝点赞",
            "点赞订阅转发",
            "字幕志愿者",
            "感谢观看",
            "明镜与点点栏目",
        )
        if any(phrase in compact for phrase in known_boilerplate):
            return True
        # Repetition alone is not a reliable hallucination signal. A speaker
        # may genuinely say "喂喂喂" or repeat a word while self-correcting;
        # those cases belong to the later semantic cleanup stage.
        return False

    def _decode_live_utterance(
        self,
        audio: np.ndarray,
        utterance_start_sample: int,
        *,
        is_final: bool,
    ) -> tuple[dict | None, float, str]:
        """Decode one speech-bounded utterance and return one translation unit."""
        if audio.size < int(0.30 * 16000):
            return None, 0.0, "too_short"
        window_rms, voiced_ratio = self._window_speech_stats(audio)
        if voiced_ratio < float(os.environ.get("ASR_UTTERANCE_MIN_VOICED_RATIO", "0.10")):
            return None, 0.0, "low_voice_ratio"

        recognition_started = time.perf_counter()
        live_model = self.load_live()
        if self.live_backend == "funasr":
            decoded = live_model.generate(
                input=audio,
                cache={},
                is_final=True,
                chunk_size=[0, 10, 5],
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1,
            )
            raw_text = (decoded[0] if decoded else {}).get("text", "")
            text = normalize_text(raw_text)
            decoded_segments = []
            if text:
                decoded_segments.append(
                    {
                        "start": 0.0,
                        "end": len(audio) / 16000.0,
                        "text": text,
                        "raw_text": raw_text,
                        "confidence": -0.15,
                    }
                )
        else:
            segments, _ = live_model.transcribe(
                audio,
                language="zh",
                beam_size=int(os.environ.get("ASR_LIVE_BEAM_SIZE", "3")),
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 350,
                    "speech_pad_ms": 120,
                },
                max_new_tokens=128,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                compression_ratio_threshold=2.2,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.55,
                word_timestamps=True,
                hallucination_silence_threshold=0.8,
                initial_prompt=(
                    "以下是普通话单人演讲。请忠实转写实际听到的内容，"
                    "保留人名、数字、否定和因果关系；不要生成字幕说明。"
                ),
                hotwords="零一二三四五六七八九十百千万",
            )
            decoded_segments = []
            for segment in segments:
                text = normalize_text(segment.text)
                if not text:
                    continue
                avg_logprob = float(getattr(segment, "avg_logprob", -0.5))
                no_speech_prob = float(getattr(segment, "no_speech_prob", 0.0))
                compression_ratio = float(getattr(segment, "compression_ratio", 0.0))
                if (
                    no_speech_prob > 0.70
                    or avg_logprob < -1.20
                    or compression_ratio > 2.60
                ):
                    continue
                decoded_segments.append(
                    {
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": text,
                        "raw_text": segment.text.strip(),
                        "confidence": avg_logprob - no_speech_prob,
                    }
                )

        latency = time.perf_counter() - recognition_started
        if not decoded_segments:
            return None, latency, "decoder_empty"

        text = normalize_text("".join(item["text"] for item in decoded_segments))
        raw_text = "".join(item["raw_text"] for item in decoded_segments).strip()
        start_offset = max(0.0, float(decoded_segments[0]["start"]))
        end_offset = max(start_offset, float(decoded_segments[-1]["end"]))
        confidence = float(
            np.mean([item["confidence"] for item in decoded_segments])
        )
        absolute_start = utterance_start_sample / 16000.0 + start_offset
        absolute_end = utterance_start_sample / 16000.0 + end_offset
        rejected = self._looks_like_hallucination(text)
        return (
            {
                "start": round(absolute_start, 2),
                "end": round(absolute_end, 2),
                "text": text,
                "raw_text": raw_text,
                "status": (
                    "REJECTED"
                    if rejected
                    else ("FINAL" if is_final else "PARTIAL")
                ),
                "rejection_reason": "hallucination_guard" if rejected else "",
                "confirmations": 1,
                "confidence": confidence,
                "voiced_ratio": voiced_ratio,
                "window_rms": window_rms,
                "hypothesis_history": [
                    {
                        "text": text,
                        "raw_text": raw_text,
                        "confidence": confidence,
                        "voiced_ratio": voiced_ratio,
                    }
                ],
            },
            latency,
            "hallucination_guard" if rejected else "accepted",
        )

    @staticmethod
    def _upsert_utterance_segment(
        state: dict,
        segment_id: int,
        decoded: dict | None,
        *,
        is_final: bool,
    ) -> dict | None:
        existing = next(
            (
                item
                for item in state.get("segments", [])
                if int(item["segment_id"]) == int(segment_id)
            ),
            None,
        )
        if decoded is None:
            if existing is not None and existing.get("status") != "FINAL":
                state["segments"].remove(existing)
            return None
        decoded = {**decoded, "segment_id": int(segment_id)}
        if decoded.get("status") != "REJECTED":
            decoded["status"] = "FINAL" if is_final else "PARTIAL"
        if existing is None:
            state["segments"].append(decoded)
            existing = decoded
        elif existing.get("status") != "FINAL":
            history = list(existing.get("hypothesis_history", []))
            history.extend(decoded.get("hypothesis_history", []))
            decoded["hypothesis_history"] = history[-6:]
            existing.update(decoded)
        state["segments"].sort(key=lambda item: (item["start"], item["segment_id"]))
        return existing

    @staticmethod
    def _common_text_prefix(left: str, right: str) -> str:
        limit = min(len(left), len(right))
        index = 0
        while index < limit and left[index] == right[index]:
            index += 1
        return left[:index]

    @classmethod
    def _update_local_agreement(
        cls,
        state: dict,
        current_text: str,
        *,
        final: bool,
    ) -> str:
        """Confirm only prefixes shared by consecutive online hypotheses."""
        current_text = normalize_text(current_text)
        previous = normalize_text(state.get("sherpa_previous_text", ""))
        confirmed = normalize_text(state.get("sherpa_confirmed_text", ""))
        if final:
            candidate = current_text or previous
        elif previous and current_text:
            candidate = cls._common_text_prefix(previous, current_text)
            holdback = int(os.environ.get("SHERPA_LOCAL_AGREEMENT_HOLDBACK", "2"))
            if holdback > 0 and len(candidate) > holdback:
                candidate = candidate[:-holdback]
            elif holdback > 0:
                candidate = ""
        else:
            candidate = ""

        if candidate.startswith(confirmed):
            state["sherpa_confirmed_text"] = candidate
        state["sherpa_previous_text"] = current_text
        return normalize_text(state.get("sherpa_confirmed_text", ""))

    @staticmethod
    def _remove_partial_segment(state: dict) -> dict | None:
        partial_id = state.get("sherpa_partial_segment_id")
        if partial_id is None:
            return None
        partial = next(
            (
                item
                for item in state.get("segments", [])
                if int(item["segment_id"]) == int(partial_id)
                and item.get("status") == "PARTIAL"
            ),
            None,
        )
        if partial is not None:
            state["segments"].remove(partial)
        state["sherpa_partial_segment_id"] = None
        return partial

    def _emit_sherpa_stable(
        self,
        state: dict,
        text: str,
        end_sample: int,
    ) -> dict | None:
        text = normalize_text(text)
        if not text:
            return None
        partial = self._remove_partial_segment(state)
        if partial is not None:
            segment_id = int(partial["segment_id"])
        else:
            segment_id = int(state["next_segment_id"])
            state["next_segment_id"] += 1
        item = {
            "segment_id": segment_id,
            "start": round(int(state["sherpa_last_emit_sample"]) / 16000, 2),
            "end": round(end_sample / 16000, 2),
            "text": text,
            "raw_text": text,
            "status": "FINAL",
            "confirmations": 2,
            "confidence": 0.0,
            "hypothesis_history": [{"text": text, "raw_text": text}],
        }
        state["segments"].append(item)
        state["segments"].sort(key=lambda value: (value["start"], value["segment_id"]))
        state["sherpa_last_emit_sample"] = int(end_sample)
        return item

    @staticmethod
    def _adaptive_commit_thresholds(
        *,
        speech_rate_cps: float,
        queue_pressure: float,
    ) -> tuple[int, int, float]:
        """Choose translation-unit size from speech rate and TTS backpressure."""
        soft_chars = max(1, int(os.environ.get("SHERPA_COMMIT_MIN_CHARS", "14")))
        hard_chars = max(
            soft_chars,
            int(os.environ.get("SHERPA_COMMIT_MAX_CHARS", "22")),
        )
        min_context = max(
            0.0,
            float(os.environ.get("SHERPA_COMMIT_MIN_SECONDS", "1.0")),
        )
        if speech_rate_cps >= 6.0:
            soft_chars = max(10, soft_chars - 2)
            hard_chars = max(soft_chars, hard_chars - 3)
            min_context = max(0.85, min_context - 0.10)
        elif 0.0 < speech_rate_cps <= 2.8:
            soft_chars += 2
            hard_chars += 2
            min_context += 0.20
        if queue_pressure >= 0.75:
            # Produce fewer, more complete translation units while speech is
            # already waiting for synthesis/playback.
            soft_chars += 4
            hard_chars += 6
            min_context += 0.20
        return soft_chars, hard_chars, min_context

    @classmethod
    def _should_commit_sherpa_delta(
        cls,
        delta: str,
        *,
        endpoint: bool,
        elapsed_since_emit: float,
        speech_rate_cps: float = 0.0,
        queue_pressure: float = 0.0,
    ) -> bool:
        """Balance translation context against bounded streaming latency.

        Natural punctuation and ASR endpoints are always safe boundaries.
        Without punctuation, a soft threshold waits for a small amount of
        acoustic context, while a hard threshold prevents unbounded delay.
        """
        delta = normalize_text(delta)
        if not delta:
            return False
        if endpoint:
            return True
        if delta.endswith(("，", "。", "！", "？", "；", "：", ",", ".", "!", "?", ";", ":")):
            return True
        soft_chars, hard_chars, min_context_seconds = cls._adaptive_commit_thresholds(
            speech_rate_cps=speech_rate_cps,
            queue_pressure=queue_pressure,
        )
        return len(delta) >= hard_chars or (
            len(delta) >= soft_chars
            and elapsed_since_emit >= min_context_seconds
        )

    def _update_sherpa_partial(
        self,
        state: dict,
        text: str,
        end_sample: int,
    ) -> None:
        text = normalize_text(text)
        partial_id = state.get("sherpa_partial_segment_id")
        partial = None
        if partial_id is not None:
            partial = next(
                (
                    item
                    for item in state.get("segments", [])
                    if int(item["segment_id"]) == int(partial_id)
                    and item.get("status") == "PARTIAL"
                ),
                None,
            )
        if not text:
            self._remove_partial_segment(state)
            return
        if partial is None:
            partial_id = int(state["next_segment_id"])
            state["next_segment_id"] += 1
            partial = {
                "segment_id": partial_id,
                "start": round(int(state["sherpa_last_emit_sample"]) / 16000, 2),
                "end": round(end_sample / 16000, 2),
                "text": text,
                "raw_text": text,
                "status": "PARTIAL",
                "confirmations": 1,
                "confidence": 0.0,
                "hypothesis_history": [{"text": text, "raw_text": text}],
            }
            state["segments"].append(partial)
            state["sherpa_partial_segment_id"] = partial_id
        else:
            partial.update(
                {
                    "end": round(end_sample / 16000, 2),
                    "text": text,
                    "raw_text": text,
                    "hypothesis_history": (
                        list(partial.get("hypothesis_history", []))
                        + [{"text": text, "raw_text": text}]
                    )[-6:],
                }
            )
        state["segments"].sort(key=lambda value: (value["start"], value["segment_id"]))

    def _transcribe_sherpa_incremental(
        self,
        audio: tuple[int, np.ndarray] | np.ndarray | None,
        state: dict,
        *,
        final: bool,
    ) -> dict:
        with self._sherpa_native_lock:
            return self._transcribe_sherpa_incremental_unlocked(
                audio,
                state,
                final=final,
            )

    def _transcribe_sherpa_incremental_unlocked(
        self,
        audio: tuple[int, np.ndarray] | np.ndarray | None,
        state: dict,
        *,
        final: bool,
    ) -> dict:
        """Run native streaming Zipformer plus LocalAgreement prefix commits."""
        # A late Gradio stream callback can arrive just after stop_recording.
        # Never recreate a native Sherpa stream after the session was flushed.
        if state.get("finalized"):
            total_samples = int(state.get("total_samples", 0))
            stable = [
                item
                for item in state.get("segments", [])
                if item.get("status") == "FINAL"
            ]
            return {
                "items": list(state.get("segments", [])),
                "stable": stable,
                "new_stable": [],
                "window_start": round(total_samples / 16000, 2),
                "window_end": round(total_samples / 16000, 2),
                "window_latency": 0.0,
                "elapsed": round(total_samples / 16000, 2),
                "state": state,
            }
        recognizer = self.load_live()
        session_id = str(state["sherpa_session_id"])
        stream = self._sherpa_sessions.get(session_id)
        if stream is None:
            if len(self._sherpa_sessions) >= 16:
                self._sherpa_sessions.pop(next(iter(self._sherpa_sessions)))
            stream = recognizer.create_stream()
            self._sherpa_sessions[session_id] = stream

        coerced = self._coerce_stream_audio(audio)
        chunk_start_sample = int(state["total_samples"])
        chunk = np.asarray([], dtype=np.float32)
        if coerced is not None:
            _, chunk = coerced
            state["total_samples"] += int(len(chunk))
            self._update_audio_quality(state, chunk)
            stream.accept_waveform(16000, chunk)

        recognition_started = time.perf_counter()
        if final:
            stream.accept_waveform(16000, np.zeros(8000, dtype=np.float32))
            stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        latency = time.perf_counter() - recognition_started

        total_samples = int(state["total_samples"])
        current_text = normalize_text(recognizer.get_result(stream))
        endpoint = bool(final or recognizer.is_endpoint(stream))
        confirmed = self._update_local_agreement(
            state,
            current_text,
            final=endpoint,
        )
        emitted = normalize_text(state.get("sherpa_emitted_text", ""))
        new_stable: list[dict] = []
        if endpoint:
            target = current_text or confirmed
        else:
            target = confirmed
        delta = target[len(emitted) :] if target.startswith(emitted) else ""
        elapsed_since_emit = max(
            0.0,
            (total_samples - int(state["sherpa_last_emit_sample"])) / 16000,
        )
        speech_rate_cps = (
            len(delta) / elapsed_since_emit if elapsed_since_emit > 0 else 0.0
        )
        queue_pressure = min(1.0, max(0.0, float(state.get("tts_queue_pressure", 0.0))))
        soft_chars, hard_chars, min_context_seconds = self._adaptive_commit_thresholds(
            speech_rate_cps=speech_rate_cps,
            queue_pressure=queue_pressure,
        )
        state["commit_policy"] = {
            "speech_rate_cps": round(speech_rate_cps, 2),
            "queue_pressure": round(queue_pressure, 2),
            "soft_chars": soft_chars,
            "hard_chars": hard_chars,
            "min_context_seconds": round(min_context_seconds, 2),
        }
        should_emit = self._should_commit_sherpa_delta(
            delta,
            endpoint=endpoint,
            elapsed_since_emit=elapsed_since_emit,
            speech_rate_cps=speech_rate_cps,
            queue_pressure=queue_pressure,
        )
        if should_emit:
            item = self._emit_sherpa_stable(state, delta, total_samples)
            if item is not None:
                new_stable.append(item)
                state["sherpa_emitted_text"] = target
                emitted = target

        remainder = (
            current_text[len(emitted) :]
            if current_text.startswith(emitted)
            else current_text
        )
        if endpoint:
            self._remove_partial_segment(state)
        else:
            self._update_sherpa_partial(state, remainder, total_samples)

        if endpoint:
            if current_text:
                state["endpoint_count"] += 1
                state["last_endpoint_reason"] = (
                    "stop" if final else "sherpa_endpoint"
                )
            if not final:
                recognizer.reset(stream)
            state["sherpa_previous_text"] = ""
            state["sherpa_confirmed_text"] = ""
            state["sherpa_emitted_text"] = ""
            state["sherpa_utterance_start_sample"] = total_samples
            state["sherpa_last_emit_sample"] = total_samples

        state["vad_state"] = (
            "IDLE" if endpoint or not current_text else "ONLINE_SPEECH"
        )
        state["last_processed_samples"] = total_samples
        state["finalized"] = bool(final)
        stable = [
            item for item in state["segments"] if item.get("status") == "FINAL"
        ]
        partial_count = len(
            [item for item in state["segments"] if item.get("status") == "PARTIAL"]
        )
        chunk_rms, chunk_voiced = self._window_speech_stats(chunk)
        state["window_records"].append(
            {
                "window_start": round(chunk_start_sample / 16000, 2),
                "window_end": round(total_samples / 16000, 2),
                "window_latency": round(latency, 3),
                "segment_count": partial_count,
                "stable_count": len(stable),
                "window_rms": round(chunk_rms, 5),
                "voiced_ratio": round(chunk_voiced, 3),
                "skipped_as_silence": not current_text and chunk_voiced < 0.08,
                "vad_state": state["vad_state"],
                "endpoint_count": int(state["endpoint_count"]),
                "rejected_utterances": 0,
                "endpoint_reason": (
                    state["last_endpoint_reason"] if endpoint else ""
                ),
            }
        )
        if final:
            self._sherpa_sessions.pop(session_id, None)
        return {
            "items": list(state["segments"]),
            "stable": stable,
            "new_stable": new_stable,
            "window_start": round(chunk_start_sample / 16000, 2),
            "window_end": round(total_samples / 16000, 2),
            "window_latency": round(latency, 3),
            "elapsed": round(total_samples / 16000, 2),
            "state": state,
        }

    def transcribe_incremental(
        self,
        audio: tuple[int, np.ndarray] | np.ndarray | None,
        state: dict | None = None,
        *,
        final: bool = False,
    ) -> dict:
        """Consume microphone blocks with endpoint-controlled stable commits.

        Energy VAD runs on 20 ms frames. Whisper/FunASR may emit display-only
        PARTIAL text while speech is active, but FINAL text is produced only
        after a real pause, the maximum utterance length, or an explicit stop.
        """
        state = state or self.new_incremental_state()
        if self.live_backend == "sherpa":
            return self._transcribe_sherpa_incremental(
                audio,
                state,
                final=final,
            )
        coerced = self._coerce_stream_audio(audio)
        completed: list[tuple[int, np.ndarray, int, str]] = []
        latest_latency = 0.0
        current_endpoint_reason = ""
        chunk_start_sample = int(state["total_samples"])

        if coerced is not None:
            _, chunk = coerced
            state["tail"].extend(chunk.astype(np.float32).tolist())
            max_tail = int(state["tail_seconds"] * 16000)
            if len(state["tail"]) > max_tail:
                state["tail"] = state["tail"][-max_tail:]
            state["total_samples"] += int(len(chunk))
            frame_size = 320
            pre_roll_limit = int(
                float(os.environ.get("ASR_VAD_PRE_ROLL_SECONDS", "0.24")) * 16000
            )
            endpoint_silence = int(
                float(os.environ.get("ASR_VAD_ENDPOINT_SECONDS", "1.00")) * 16000
            )
            max_utterance = int(
                float(os.environ.get("ASR_MAX_UTTERANCE_SECONDS", "10.0")) * 16000
            )
            minimum_voiced = int(
                float(os.environ.get("ASR_MIN_UTTERANCE_VOICED_SECONDS", "0.30"))
                * 16000
            )
            speech_start_frames = int(
                os.environ.get("ASR_VAD_START_FRAMES", "3")
            )
            static_threshold = float(
                os.environ.get("ASR_VAD_RMS_THRESHOLD", "0.008")
            )

            for offset in range(0, len(chunk), frame_size):
                frame = chunk[offset : offset + frame_size]
                if frame.size == 0:
                    continue
                frame_rms = float(
                    np.sqrt(np.mean(np.square(frame, dtype=np.float64)))
                )
                threshold = max(static_threshold, float(state["noise_rms"]) * 3.0)
                is_speech = frame_rms >= threshold
                frame_end_sample = chunk_start_sample + offset + len(frame)

                if state["vad_state"] == "IDLE":
                    state["pre_roll"].extend(frame.tolist())
                    if len(state["pre_roll"]) > pre_roll_limit:
                        state["pre_roll"] = state["pre_roll"][-pre_roll_limit:]
                    if is_speech:
                        state["speech_run_frames"] += 1
                    else:
                        state["speech_run_frames"] = 0
                        state["noise_rms"] = (
                            0.97 * float(state["noise_rms"]) + 0.03 * frame_rms
                        )
                    if state["speech_run_frames"] < speech_start_frames:
                        continue

                    utterance_id = int(state["next_segment_id"])
                    state["next_segment_id"] += 1
                    state["active_utterance_id"] = utterance_id
                    state["utterance_audio"] = list(state["pre_roll"])
                    state["utterance_start_sample"] = max(
                        0, frame_end_sample - len(state["utterance_audio"])
                    )
                    state["utterance_last_voiced_sample"] = len(
                        state["utterance_audio"]
                    )
                    state["utterance_voiced_samples"] = min(
                        len(state["utterance_audio"]),
                        speech_start_frames * frame_size,
                    )
                    state["utterance_trailing_silence"] = 0
                    state["last_partial_decode_samples"] = 0
                    state["pre_roll"] = []
                    state["vad_state"] = "SPEECH"
                    continue

                state["utterance_audio"].extend(frame.tolist())
                if is_speech:
                    state["utterance_voiced_samples"] += len(frame)
                    state["utterance_trailing_silence"] = 0
                    state["utterance_last_voiced_sample"] = len(
                        state["utterance_audio"]
                    )
                    state["vad_state"] = "SPEECH"
                else:
                    state["utterance_trailing_silence"] += len(frame)
                    state["vad_state"] = "SILENCE_PENDING"

                endpoint_reason = ""
                if state["utterance_trailing_silence"] >= endpoint_silence:
                    endpoint_reason = "pause"
                elif len(state["utterance_audio"]) >= max_utterance:
                    endpoint_reason = "max_length"
                if not endpoint_reason:
                    continue

                post_roll = int(0.12 * 16000)
                cut = min(
                    len(state["utterance_audio"]),
                    int(state["utterance_last_voiced_sample"]) + post_roll,
                )
                utterance = np.asarray(
                    state["utterance_audio"][:cut], dtype=np.float32
                )
                utterance_id = int(state["active_utterance_id"])
                utterance_start = int(state["utterance_start_sample"])
                if int(state["utterance_voiced_samples"]) >= minimum_voiced:
                    completed.append(
                        (utterance_id, utterance, utterance_start, endpoint_reason)
                    )
                else:
                    self._upsert_utterance_segment(
                        state, utterance_id, None, is_final=True
                    )

                state["pre_roll"] = (
                    state["utterance_audio"][-pre_roll_limit:]
                    if endpoint_reason == "pause"
                    else []
                )
                state["vad_state"] = "IDLE"
                state["speech_run_frames"] = 0
                state["utterance_audio"] = []
                state["utterance_start_sample"] = None
                state["utterance_last_voiced_sample"] = 0
                state["utterance_voiced_samples"] = 0
                state["utterance_trailing_silence"] = 0
                state["active_utterance_id"] = None
                state["last_partial_decode_samples"] = 0

        if final and state.get("active_utterance_id") is not None:
            post_roll = int(0.12 * 16000)
            cut = min(
                len(state["utterance_audio"]),
                int(state["utterance_last_voiced_sample"]) + post_roll,
            )
            utterance = np.asarray(state["utterance_audio"][:cut], dtype=np.float32)
            utterance_id = int(state["active_utterance_id"])
            if int(state["utterance_voiced_samples"]) >= int(0.30 * 16000):
                completed.append(
                    (
                        utterance_id,
                        utterance,
                        int(state["utterance_start_sample"]),
                        "stop",
                    )
                )
            else:
                self._upsert_utterance_segment(
                    state, utterance_id, None, is_final=True
                )
            state["vad_state"] = "IDLE"
            state["utterance_audio"] = []
            state["utterance_start_sample"] = None
            state["utterance_last_voiced_sample"] = 0
            state["utterance_voiced_samples"] = 0
            state["utterance_trailing_silence"] = 0
            state["active_utterance_id"] = None
            state["last_partial_decode_samples"] = 0

        new_stable: list[dict] = []
        for utterance_id, utterance, utterance_start, endpoint_reason in completed:
            decoded, latency, decode_reason = self._decode_live_utterance(
                utterance,
                utterance_start,
                is_final=True,
            )
            latest_latency += latency
            committed = self._upsert_utterance_segment(
                state, utterance_id, decoded, is_final=True
            )
            if committed is not None and committed.get("status") == "FINAL":
                new_stable.append(committed)
                state["endpoint_count"] += 1
                state["last_endpoint_reason"] = endpoint_reason
                current_endpoint_reason = endpoint_reason
            else:
                state["rejected_utterances"] += 1
                state["last_endpoint_reason"] = decode_reason
                current_endpoint_reason = decode_reason

        active_audio = np.asarray(state.get("utterance_audio", []), dtype=np.float32)
        partial_min = int(
            float(os.environ.get("ASR_PARTIAL_MIN_SECONDS", "2.00")) * 16000
        )
        partial_step = int(
            float(os.environ.get("ASR_PARTIAL_EVERY_SECONDS", "1.20")) * 16000
        )
        if (
            not final
            and state.get("active_utterance_id") is not None
            and len(active_audio) >= partial_min
            and len(active_audio) - int(state["last_partial_decode_samples"])
            >= partial_step
        ):
            decoded, latency, _ = self._decode_live_utterance(
                active_audio,
                int(state["utterance_start_sample"]),
                is_final=False,
            )
            latest_latency += latency
            self._upsert_utterance_segment(
                state,
                int(state["active_utterance_id"]),
                decoded,
                is_final=False,
            )
            state["last_partial_decode_samples"] = len(active_audio)

        total_samples = int(state["total_samples"])
        state["last_processed_samples"] = total_samples
        state["finalized"] = bool(final)
        stable = [
            item for item in state["segments"] if item.get("status") == "FINAL"
        ]
        chunk_rms, chunk_voiced = self._window_speech_stats(
            coerced[1] if coerced is not None else np.asarray([], dtype=np.float32)
        )
        record = {
            "window_start": round(chunk_start_sample / 16000, 2),
            "window_end": round(total_samples / 16000, 2),
            "window_latency": round(latest_latency, 3),
            "segment_count": len(
                [item for item in state["segments"] if item.get("status") == "PARTIAL"]
            ),
            "stable_count": len(stable),
            "window_rms": round(chunk_rms, 5),
            "voiced_ratio": round(chunk_voiced, 3),
            "skipped_as_silence": (
                state["vad_state"] == "IDLE" and not completed and chunk_voiced < 0.08
            ),
            "vad_state": state["vad_state"],
            "endpoint_count": int(state["endpoint_count"]),
            "rejected_utterances": int(state["rejected_utterances"]),
            "endpoint_reason": current_endpoint_reason,
        }
        state["window_records"].append(record)
        return {
            "items": list(state["segments"]),
            "stable": stable,
            "new_stable": new_stable,
            "window_start": record["window_start"],
            "window_end": record["window_end"],
            "window_latency": record["window_latency"],
            "elapsed": round(total_samples / 16000, 2),
            "state": state,
        }
