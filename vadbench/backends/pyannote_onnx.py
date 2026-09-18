from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from vadbench.audio import load_audio
from vadbench.base import BaseVad
from vadbench.perf.ort_utils import MODELS, make_ort_session
from vadbench.registry import register
from vadbench.types import SpeechSegment

AudioInput = Union[str, Path, np.ndarray]


def _seg_dir() -> Path:
    p = MODELS / "pyannote" / "segmentation-3.0"
    if not p.is_dir():
        raise FileNotFoundError(p)
    return p


def _default_onnx() -> Path:
    p = MODELS / "pyannote" / "onnx" / "segmentation-3.0.onnx"
    if not p.is_file():
        raise FileNotFoundError(
            "Pyannote ONNX not found; run vadbench.perf.export_onnx"
        )
    return p


class OrtSegmentationModel(nn.Module):
    """ORT-backed stand-in for pyannote segmentation Model (metadata from template)."""

    def __init__(self, session, template: nn.Module):
        super().__init__()
        self._session = session
        self._in_name = session.get_inputs()[0].name
        self.specifications = template.specifications
        self.receptive_field = template.receptive_field
        self.audio = template.audio
        self._device = torch.device("cpu")

    @property
    def device(self):
        return self._device

    def to(self, device):  # noqa: A003
        self._device = torch.device(device) if not isinstance(device, torch.device) else device
        return self

    def eval(self):
        return self

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        x = waveforms.detach().cpu().numpy().astype(np.float32)
        y = self._session.run(None, {self._in_name: x})[0]
        return torch.from_numpy(np.asarray(y)).to(waveforms.device)


@register("pyannote_onnx")
class PyannoteOnnxVad(BaseVad):
    """Pyannote VAD: ONNX segmentation backbone + same VoiceActivityDetection postprocess."""

    name = "pyannote_onnx"

    def __init__(
        self,
        model_path: str | Path | None = None,
        onnx_path: str | Path | None = None,
        threads: int = 1,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
    ) -> None:
        try:
            from pyannote.audio import Model
            from pyannote.audio.pipelines import VoiceActivityDetection
        except ImportError as e:
            raise ImportError(
                "pyannote.audio is required: pip install pyannote.audio"
            ) from e

        seg_dir = Path(model_path) if model_path else _seg_dir()
        onnx = Path(onnx_path) if onnx_path else _default_onnx()
        self._onnx_path = onnx
        self.threads = int(threads)

        template = Model.from_pretrained(str(seg_dir))
        template.eval()
        pipeline = VoiceActivityDetection(segmentation=template)
        pipeline.instantiate(
            {
                "min_duration_on": min_duration_on,
                "min_duration_off": min_duration_off,
            }
        )
        session = make_ort_session(onnx, threads=threads)
        pipeline._segmentation.model = OrtSegmentationModel(session, template)
        self._pipeline = pipeline

    def detect(
        self,
        audio: AudioInput,
        sample_rate: int | None = None,
    ) -> list[SpeechSegment]:
        wav, sr = load_audio(audio, sample_rate)
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
