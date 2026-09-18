from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from vadbench.audio import float_to_pcm16, load_audio
from vadbench.base import BaseVad
from vadbench.registry import register
from vadbench.types import SpeechSegment

AudioInput = Union[str, Path, np.ndarray]


@register("webrtc")
class WebRtcVad(BaseVad):
    """Google WebRTC VAD via py-webrtcvad. No separate model weights."""

    name = "webrtc"

    def __init__(
        self,
        aggressiveness: int = 2,
        frame_ms: int = 30,
        min_speech_ms: float = 100.0,
        min_silence_ms: float = 100.0,
    ) -> None:
        try:
            import pkg_resources  # noqa: F401
        except ImportError:
            # setuptools>=82 may omit pkg_resources; webrtcvad still imports it.
            import sys
            import types

            stub = types.ModuleType("pkg_resources")
            stub.get_distribution = lambda _name: types.SimpleNamespace(
                version="2.0.10"
            )
            sys.modules["pkg_resources"] = stub
        try:
            import webrtcvad
        except ImportError as e:
            raise ImportError(
                "webrtcvad is required: pip install webrtcvad"
            ) from e
        if aggressiveness not in (0, 1, 2, 3):
            raise ValueError("aggressiveness must be 0..3")
        if frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20, or 30")
        self._vad = webrtcvad.Vad(aggressiveness)
        self.frame_ms = frame_ms
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms

    def detect(
        self,
        audio: AudioInput,
        sample_rate: int | None = None,
    ) -> list[SpeechSegment]:
        wav, sr = load_audio(audio, sample_rate)
        pcm = float_to_pcm16(wav)
        frame_bytes = int(sr * self.frame_ms / 1000) * 2  # int16
        if frame_bytes <= 0:
            return []

        flags: list[bool] = []
        for i in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
            frame = pcm[i : i + frame_bytes]
            flags.append(self._vad.is_speech(frame, sr))

        return _flags_to_segments(
            flags,
            frame_ms=self.frame_ms,
            min_speech_ms=self.min_speech_ms,
            min_silence_ms=self.min_silence_ms,
        )


def _flags_to_segments(
    flags: list[bool],
    frame_ms: int,
    min_speech_ms: float,
    min_silence_ms: float,
) -> list[SpeechSegment]:
    if not flags:
        return []

    # Merge with hysteresis: drop short speech / fill short silence.
    min_speech_frames = max(1, int(round(min_speech_ms / frame_ms)))
    min_silence_frames = max(1, int(round(min_silence_ms / frame_ms)))

    # First pass: raw runs
    runs: list[tuple[bool, int, int]] = []
    cur = flags[0]
    start = 0
    for i, f in enumerate(flags[1:], start=1):
        if f != cur:
            runs.append((cur, start, i))
            cur = f
            start = i
    runs.append((cur, start, len(flags)))

    # Fill short non-speech gaps between speech
    merged: list[tuple[bool, int, int]] = []
    for speech, s, e in runs:
        if (
            not speech
            and merged
            and (e - s) < min_silence_frames
            and merged[-1][0]
        ):
            ps, pe = merged[-1][1], merged[-1][2]
            merged[-1] = (True, ps, e)
        else:
            if merged and merged[-1][0] == speech:
                merged[-1] = (speech, merged[-1][1], e)
            else:
                merged.append((speech, s, e))

    segs: list[SpeechSegment] = []
    for speech, s, e in merged:
        if not speech:
            continue
        if (e - s) < min_speech_frames:
            continue
        segs.append(
            SpeechSegment(
                start=s * frame_ms / 1000.0,
                end=e * frame_ms / 1000.0,
            )
        )
    return segs
