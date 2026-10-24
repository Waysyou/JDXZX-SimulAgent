from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

from .realtime_input import iter_audio_blocks, load_audio_16k
from .stream_asr import StreamingASR
from .translation import build_translator, validate_translation
from .tts import PiperTTS


def _combine_wavs(
    inputs: list[Path],
    output: Path,
    gap_ms: int = 80,
) -> None:
    """Concatenate mono PCM WAV files with a short inter-segment gap."""
    if not inputs:
        raise ValueError("No TTS segments were generated")

    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(inputs[0]), "rb") as first:
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]
    gap_frames = int(params.framerate * gap_ms / 1000)
    silence = b"\x00" * gap_frames * params.nchannels * params.sampwidth

    for path in inputs[1:]:
        with wave.open(str(path), "rb") as current:
            if (
                current.getnchannels() != params.nchannels
                or current.getsampwidth() != params.sampwidth
                or current.getframerate() != params.framerate
                or current.getcomptype() != params.comptype
            ):
                raise ValueError(f"Incompatible Piper WAV format: {path}")
            frames.extend([silence, current.readframes(current.getnframes())])

    with wave.open(str(output), "wb") as merged:
        merged.setparams(params)
        merged.writeframes(b"".join(frames))


def run_file_pipeline(
    audio_path: str | Path,
    output_dir: str | Path,
    *,
    block_seconds: float = 0.4,
) -> dict[str, Any]:
    """Generate submission artifacts through the same Sherpa stream as the UI.

    The input file is exposed as consecutive real-time-sized audio blocks. A
    translation unit is emitted only after the online recognizer has confirmed
    it, so the generated TXT/WAV use the same segmentation semantics as the
    recorded streaming demonstration video.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Input audio does not exist: {audio_path}")
    if block_seconds <= 0:
        raise ValueError("block_seconds must be greater than zero")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem

    asr = StreamingASR()
    # ``StreamingASR`` defaults to the legacy file backend when imported by a
    # standalone script.  Select Sherpa for submission generation without
    # mutating process-wide environment variables (important for test and UI
    # isolation in the same Python process).
    if not asr.live_backend:
        asr.live_backend = "sherpa"
    elif asr.live_backend == "whisper":
        asr.live_backend = "sherpa"
    translator = build_translator()
    tts = PiperTTS()
    tts_dir = output_dir / f".{stem}_tts_segments"

    translated_by_id: dict[int, tuple[str, str]] = {}
    tts_by_id: dict[int, Path] = {}
    windows: list[dict[str, Any]] = []

    def commit(items: list[dict]) -> None:
        if not items:
            return
        translations = translator.translate_many([item["text"] for item in items])
        if len(translations) != len(items):
            raise RuntimeError("Translation count does not match stable ASR segments")
        translations = [
            validate_translation(item["text"], translated)
            for item, translated in zip(items, translations)
        ]
        tts_dir.mkdir(parents=True, exist_ok=True)
        for item, translated in zip(items, translations):
            segment_id = int(item["segment_id"])
            translated_by_id[segment_id] = (item["text"], translated)
            segment_wav = tts_dir / f"segment_{segment_id:04d}.wav"
            tts.synthesize(translated, segment_wav)
            tts_by_id[segment_id] = segment_wav

    asr.load_live()
    state = asr.new_incremental_state(process_every_seconds=block_seconds)
    samples = load_audio_16k(audio_path)
    if samples.size == 0:
        raise ValueError(f"Input audio is empty: {audio_path}")

    stable: list[dict] = []
    for sample_rate, block in iter_audio_blocks(samples, block_seconds=block_seconds):
        result = asr.transcribe_incremental((sample_rate, block), state)
        state = result["state"]
        commit(result.get("new_stable", []))
        stable = list(result.get("stable", []))
        windows.append(
            {
                "window_start": result["window_start"],
                "window_end": result["window_end"],
                "window_latency": result["window_latency"],
                "segment_count": len(
                    [item for item in result["items"] if item.get("status") == "PARTIAL"]
                ),
                "stable_count": len(stable),
            }
        )

    result = asr.transcribe_incremental(None, state, final=True)
    commit(result.get("new_stable", []))
    stable = list(result.get("stable", []))
    windows.append(
        {
            "window_start": result["window_start"],
            "window_end": result["window_end"],
            "window_latency": result["window_latency"],
            "segment_count": 0,
            "stable_count": len(stable),
        }
    )

    if not stable:
        raise RuntimeError("Streaming ASR did not produce a confirmed segment")
    missing = [item["segment_id"] for item in stable if item["segment_id"] not in translated_by_id]
    if missing:
        raise RuntimeError(f"Missing translations for stable segments: {missing}")

    translation_path = output_dir / f"{stem}_translation.txt"
    translation_path.write_text(
        "\n".join(translated_by_id[item["segment_id"]][1] for item in stable) + "\n",
        encoding="utf-8",
    )

    segment_paths = [tts_by_id[item["segment_id"]] for item in stable]
    tts_path = output_dir / f"{stem}_translation_tts.wav"
    _combine_wavs(segment_paths, tts_path)

    metadata_path = output_dir / f"{stem}_pipeline_metrics.json"
    metadata = {
        "source_audio": str(audio_path),
        "translation_file": str(translation_path),
        "tts_file": str(tts_path),
        "stable_segments": len(stable),
        "windows": windows,
        "translation_backend": translator.__class__.__name__,
        "translation_device": translator.device,
        "translation_load_seconds": translator.load_seconds,
        "tts_length_scale": tts.length_scale,
        "asr_backend": asr.live_backend,
        "stream_block_seconds": block_seconds,
        "audio_quality": state.get("audio_quality", {}),
        "final_commit_policy": state.get("commit_policy", {}),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata
