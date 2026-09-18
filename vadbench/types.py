from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechSegment:
    """Speech interval in seconds (inclusive start, exclusive end preferred)."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")

    @property
    def duration(self) -> float:
        return self.end - self.start
