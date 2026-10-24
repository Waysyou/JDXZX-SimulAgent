from __future__ import annotations

from typing import Literal, TypedDict


SegmentStatus = Literal["PARTIAL", "FINAL", "REJECTED"]
QualityStatus = Literal["ACCEPTED", "REJECTED", "DEGRADED"]


class SegmentRecord(TypedDict):
    segment_id: int
    start: float
    end: float
    text: str
    status: SegmentStatus


class TranslationRecord(TypedDict):
    source: str
    text: str
    quality_status: QualityStatus


class TTSRecord(TypedDict):
    path: str
    audio_start: float
    audio_end: float


class AudioQueueItem(TypedDict):
    segment_id: int
    path: str


def segment_key(segment_id: int | str) -> str:
    return str(int(segment_id))


def translation_record(
    source: str,
    text: str,
    quality_status: QualityStatus,
) -> TranslationRecord:
    return {
        "source": str(source),
        "text": str(text),
        "quality_status": quality_status,
    }


def tts_record(path: str, audio_start: float, audio_end: float) -> TTSRecord:
    start = float(audio_start)
    end = float(audio_end)
    if end < start:
        raise ValueError("TTS audio_end cannot be earlier than audio_start")
    return {"path": str(path), "audio_start": start, "audio_end": end}


def audio_queue_item(segment_id: int | str, path: str) -> AudioQueueItem:
    return {"segment_id": int(segment_id), "path": str(path)}


def validate_segment(record: dict) -> SegmentRecord:
    required = {"segment_id", "start", "end", "text", "status"}
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"Segment is missing fields: {', '.join(missing)}")
    start = float(record["start"])
    end = float(record["end"])
    if end < start:
        raise ValueError("Segment end cannot be earlier than start")
    status = str(record["status"])
    if status not in {"PARTIAL", "FINAL", "REJECTED"}:
        raise ValueError(f"Unknown segment status: {status}")
    return {
        "segment_id": int(record["segment_id"]),
        "start": start,
        "end": end,
        "text": str(record["text"]),
        "status": status,  # type: ignore[typeddict-item]
    }
