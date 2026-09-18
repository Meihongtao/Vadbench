from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from vadbench.audio import load_audio
from vadbench.base import BaseVad
from vadbench.registry import register
from vadbench.types import SpeechSegment

AudioInput = Union[str, Path, np.ndarray]


def _default_model_path() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    for name in ("silero_vad.onnx", "silero_vad.jit"):
        p = root / "models" / "silero" / name
        if p.is_file():
            return p
    return None


@register("silero")
class SileroVad(BaseVad):
    """Silero VAD (ONNX preferred when local model exists)."""

    name = "silero"

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
        device: str = "cpu",
        onnx: bool | None = None,
    ) -> None:
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as e:
            raise ImportError(
                "silero-vad is required: pip install silero-vad"
            ) from e

        path = Path(model_path) if model_path else _default_model_path()
        use_onnx = onnx if onnx is not None else (
            path is not None and path.suffix.lower() == ".onnx"
        )

        if path is not None and path.is_file():
            # Official loader still downloads/registers utils; pass onnx flag.
            self._model = load_silero_vad(onnx=use_onnx)
            # If a local ONNX path is provided, try to swap session when possible.
            if use_onnx and path.suffix.lower() == ".onnx":
                try:
                    import onnxruntime as ort

                    self._model.session = ort.InferenceSession(
                        str(path),
                        providers=["CPUExecutionProvider"],
                    )
                except Exception:
                    pass
        else:
            self._model = load_silero_vad(onnx=bool(use_onnx))

        self._get_speech_timestamps = get_speech_timestamps
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.device = device

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
