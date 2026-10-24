from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from src.stream_asr import StreamingASR


def load_audio_16k(audio_path: str | Path) -> np.ndarray:
    """Load an audio file once and normalize it to mono 16 kHz float32."""
    samples, sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=False,
    )
    coerced = StreamingASR._coerce_stream_audio((sample_rate, samples))
    if coerced is None:
        return np.asarray([], dtype=np.float32)
    _, normalized = coerced
    return np.asarray(normalized, dtype=np.float32)


def iter_audio_blocks(
    samples: np.ndarray,
    *,
    sample_rate: int = 16000,
    block_seconds: float = 0.4,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield consecutive real-time-sized blocks without looking ahead."""
    block_size = max(1, int(round(sample_rate * block_seconds)))
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    for start in range(0, len(mono), block_size):
        yield sample_rate, mono[start : start + block_size]

