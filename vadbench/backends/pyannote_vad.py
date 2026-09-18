from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np

from vadbench.audio import load_audio
from vadbench.base import BaseVad
from vadbench.registry import register
from vadbench.types import SpeechSegment

AudioInput = Union[str, Path, np.ndarray]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _find_local_pyannote() -> Path | None:
    root = _project_root() / "models" / "pyannote"
    if not root.is_dir():
        return None
    # Prefer segmentation checkpoint for pure VAD.
    preferred = [
        "segmentation-3.0",
        "voice-activity-detection",
        "speaker-diarization-3.1",
    ]
    for name in preferred:
        p = root / name
        if p.is_dir() and any(p.iterdir()):
            return p
    # Any subdirectory with config / pytorch weights
    for p in sorted(root.iterdir()):
        if p.is_dir() and any(p.iterdir()):
            return p
    return None


def _looks_like_segmentation(path: Path) -> bool:
    names = {x.name.lower() for x in path.iterdir()}
    return any(
        n.startswith("pytorch_model") or n.endswith(".bin") or n.endswith(".safetensors")
        for n in names
    ) or "config.yaml" in names or "config.json" in names


@register("pyannote")
class PyannoteVad(BaseVad):
    """pyannote.audio VAD; prefers local ModelScope weights under models/pyannote/."""

    name = "pyannote"

    def __init__(
        self,
        model: str | Path | None = None,
        token: str | None = None,
        device: str = "cpu",
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
    ) -> None:
        try:
            from pyannote.audio import Model, Pipeline
            from pyannote.audio.pipelines import VoiceActivityDetection
        except ImportError as e:
            raise ImportError(
                "pyannote.audio is required: pip install pyannote.audio"
            ) from e

        hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGINGFACE_TOKEN"
        )
        local = Path(model) if model else _find_local_pyannote()

        pipeline = None
        if local is not None and local.is_dir():
            origin = str(local)
            # Try as full pipeline directory first (diarization), then segmentation model.
            try:
                pipeline = Pipeline.from_pretrained(origin, token=hf_token)
            except Exception:
                pipeline = None
            if pipeline is None and _looks_like_segmentation(local):
                segmentation = Model.from_pretrained(origin, token=hf_token)
                pipeline = VoiceActivityDetection(segmentation=segmentation)
                pipeline.instantiate(
                    {
                        "min_duration_on": min_duration_on,
                        "min_duration_off": min_duration_off,
                    }
                )
            if pipeline is None:
                raise RuntimeError(
                    f"Could not load pyannote from local dir: {local}. "
                    "Run: python -m vadbench.download --engine pyannote"
                )
        else:
            origin = "pyannote/voice-activity-detection"
            try:
                pipeline = Pipeline.from_pretrained(origin, token=hf_token)
            except Exception:
                pipeline = None
            if pipeline is None:
                segmentation = Model.from_pretrained(
                    "pyannote/segmentation-3.0", token=hf_token
                )
                pipeline = VoiceActivityDetection(segmentation=segmentation)
                pipeline.instantiate(
                    {
                        "min_duration_on": min_duration_on,
                        "min_duration_off": min_duration_off,
                    }
                )

        if device and device != "cpu":
            import torch

            pipeline.to(torch.device(device))

        self._pipeline = pipeline

    def detect(
        self,
        audio: AudioInput,
        sample_rate: int | None = None,
    ) -> list[SpeechSegment]:
        # Always pass waveform dict — avoids torchcodec/ffmpeg path issues on Windows.
        wav, sr = load_audio(audio, sample_rate)
        import torch

        waveform = torch.from_numpy(wav).unsqueeze(0)
        annotation = self._pipeline({"waveform": waveform, "sample_rate": sr})

        segs: list[SpeechSegment] = []
        if hasattr(annotation, "get_timeline"):
            timeline = annotation.get_timeline().support()
        else:
            timeline = annotation
        for speech in timeline:
            segs.append(SpeechSegment(start=float(speech.start), end=float(speech.end)))
        return segs
