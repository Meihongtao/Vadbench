from __future__ import annotations

import numpy as np

from vadbench.types import SpeechSegment

DEFAULT_FRAME_MS = 10.0


def segments_to_frames(
    segments: list[SpeechSegment],
    duration_s: float,
    frame_ms: float = DEFAULT_FRAME_MS,
) -> np.ndarray:
    """Rasterize speech segments to a binary frame array (1=speech)."""
    if duration_s < 0:
        raise ValueError("duration_s must be >= 0")
    n = int(np.ceil(duration_s * 1000.0 / frame_ms)) if duration_s > 0 else 0
    frames = np.zeros(n, dtype=np.uint8)
    if n == 0:
        return frames
    for seg in segments:
        start = max(0.0, float(seg.start))
        end = min(duration_s, float(seg.end))
        if end <= start:
            continue
        i0 = int(start * 1000.0 / frame_ms)
        i1 = int(np.ceil(end * 1000.0 / frame_ms))
        i0 = max(0, min(n, i0))
        i1 = max(0, min(n, i1))
        if i1 > i0:
            frames[i0:i1] = 1
    return frames


def union_segments(segments: list[SpeechSegment]) -> list[SpeechSegment]:
    """Merge overlapping/adjacent speech intervals."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    merged: list[SpeechSegment] = [ordered[0]]
    for seg in ordered[1:]:
        last = merged[-1]
        if seg.start <= last.end:
            merged[-1] = SpeechSegment(start=last.start, end=max(last.end, seg.end))
        else:
            merged.append(seg)
    return merged
