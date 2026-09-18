from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from vadbench.audio import load_audio
from vadbench.base import BaseVad
from vadbench.perf.ort_utils import MODELS, make_ort_session
from vadbench.registry import register
from vadbench.types import SpeechSegment

AudioInput = Union[str, Path, np.ndarray]


def _default_onnx() -> Path:
    for p in (
        MODELS / "silero" / "onnx" / "silero_vad.onnx",
        MODELS / "silero" / "silero_vad.onnx",
    ):
        if p.is_file():
            return p
    raise FileNotFoundError("Silero ONNX not found; run vadbench.perf.export_onnx")


@register("silero_onnx")
class SileroOnnxVad(BaseVad):
    """Silero VAD via ONNXRuntime with configurable thread count."""

    name = "silero_onnx"

    def __init__(
        self,
        model_path: str | Path | None = None,
        threads: int = 1,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
    ) -> None:
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as e:
            raise ImportError("silero-vad is required: pip install silero-vad") from e

        path = Path(model_path) if model_path else _default_onnx()
        self._model = load_silero_vad(onnx=True)
        self._model.session = make_ort_session(path, threads=threads)
        self._onnx_path = path
        self._get_speech_timestamps = get_speech_timestamps
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.threads = int(threads)

    def detect(
        self,
        audio: AudioInput,
        sample_rate: int | None = None,
    ) -> list[SpeechSegment]:
        import torch

        wav, sr = load_audio(audio, sample_rate)
        tensor = torch.from_numpy(wav)
        ts = self._get_speech_timestamps(
            tensor,
            self._model,
            sampling_rate=sr,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            return_seconds=False,
        )
        return [
            SpeechSegment(start=t["start"] / float(sr), end=t["end"] / float(sr))
            for t in ts
        ]
