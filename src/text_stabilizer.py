from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

try:
    from opencc import OpenCC
except ImportError:  # pragma: no cover - optional dependency fallback
    OpenCC = None


_SPACE_RE = re.compile(r"\s+")
_FILLER_RE = re.compile(r"(?:^|\s)(?:嗯+|呃+|啊+|额+)(?=\s|$)")
_PUNCTUATION = str.maketrans({"，": ",", "。": ".", "？": "?", "！": "!"})
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
_CORRECTION_PATH = (
    _CONFIG_DIR / "asr_corrections.json"
)


@lru_cache(maxsize=8)
def _load_correction_file(path_text: str) -> dict[str, str]:
    path = Path(path_text)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ASR correction file must contain a JSON object: {path}")
    return {str(source): str(target) for source, target in payload.items()}


def _active_corrections() -> dict[str, str]:
    """Load generic corrections plus an explicitly selected scene profile.

    The default correction table must remain domain-neutral. Development-set
    or presentation-specific terms are only loaded when the operator sets
    ``ASR_CORRECTIONS_PROFILE`` to a JSON filename or absolute path.
    """
    corrections = dict(_load_correction_file(str(_CORRECTION_PATH)))
    profile = os.environ.get("ASR_CORRECTIONS_PROFILE", "").strip()
    if not profile:
        return corrections
    profile_path = Path(profile)
    if not profile_path.is_absolute():
        profile_path = _CONFIG_DIR / profile_path
    corrections.update(_load_correction_file(str(profile_path.resolve())))
    return corrections


def normalize_text(text: str) -> str:
    """Normalize ASR text without changing its semantic content."""
    text = text.strip()
    if not text:
        return ""
    if OpenCC is not None:
        text = OpenCC("t2s").convert(text)
    text = text.translate(_PUNCTUATION)
    text = _FILLER_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    for source, target in _active_corrections().items():
        text = text.replace(source, target)
    return text


def _longest_suffix_prefix_overlap(left: str, right: str, minimum: int = 2) -> int:
    max_size = min(len(left), len(right))
    for size in range(max_size, minimum - 1, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


@dataclass
class StableSegment:
    segment_id: int
    start: float
    end: float
    text: str
    raw_text: str
    status: str = "FINAL"

    def as_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "text": self.text,
            "raw_text": self.raw_text,
            "status": self.status,
        }


class TextStabilizer:
    """Merge ASR window outputs and suppress repeated text.

    The first prototype uses FINAL ASR segments. The same API can later accept
    PARTIAL/STABLE updates without changing the translation or UI layers.
    """

    def __init__(self) -> None:
        self._segments: list[StableSegment] = []
        self._next_id = 1

    @property
    def segments(self) -> list[StableSegment]:
        return list(self._segments)

    def feed(self, raw_text: str, start: float, end: float, status: str = "FINAL") -> list[dict]:
        normalized = normalize_text(raw_text)
        if not normalized:
            return []

        if self._segments:
            previous = self._segments[-1]
            if normalized == previous.text or normalized in previous.text:
                previous.end = max(previous.end, end)
                return [previous.as_dict()]

            overlap = _longest_suffix_prefix_overlap(previous.text, normalized)
            if overlap:
                previous.text += normalized[overlap:]
                previous.end = max(previous.end, end)
                previous.status = "UPDATED"
                return [previous.as_dict()]

        segment = StableSegment(
            segment_id=self._next_id,
            start=start,
            end=end,
            text=normalized,
            raw_text=raw_text.strip(),
            status=status,
        )
        self._next_id += 1
        self._segments.append(segment)
        return [segment.as_dict()]

    def feed_many(self, items: Iterable[dict]) -> list[dict]:
        output: list[dict] = []
        for item in items:
            output.extend(self.feed(item.get("text", ""), float(item.get("start", 0)), float(item.get("end", 0))))
        return output

    def flush(self) -> list[dict]:
        return [segment.as_dict() for segment in self._segments]
