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


def _default_model_dir() -> Path:
    p = MODELS / "funasr_fsmn"
    if not p.is_dir():
        raise FileNotFoundError(p)
    return p


def _default_onnx() -> Path:
    p = MODELS / "funasr_fsmn" / "onnx" / "model.onnx"
    if not p.is_file():
        raise FileNotFoundError("FunASR ONNX not found; run vadbench.perf.export_onnx")
    return p


class OrtFsmnEncoder(nn.Module):
    """Drop-in replacement for FunASR FSMN encoder using ONNXRuntime.

    Matches ``FSMN.forward(feats, cache=dict)`` so AutoModel postprocess stays identical.
    """

    def __init__(self, session, n_layers: int = 4, proj_dim: int = 128, lookback: int = 19):
        super().__init__()
        self.session = session
        self.n_layers = n_layers
        self.proj_dim = proj_dim
        self.lookback = lookback
        self._in_speech = session.get_inputs()[0].name
        # FunASR AutoModel uses next(model.parameters()).device — keep a Parameter.
        self._device_anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        if cache is None:
            cache = {}
        speech = input.detach().cpu().numpy().astype(np.float32)
        feeds: dict[str, np.ndarray] = {self._in_speech: speech}
        b = int(speech.shape[0])
        for i in range(self.n_layers):
            key = f"cache_layer_{i}"
            if key not in cache or cache[key] is None:
                cache[key] = torch.zeros(
                    b, self.proj_dim, self.lookback, 1, dtype=torch.float32
                )
            feeds[f"in_cache{i}"] = (
                cache[key].detach().cpu().numpy().astype(np.float32)
            )
        outs = self.session.run(None, feeds)
        logits = torch.from_numpy(np.asarray(outs[0])).to(device=input.device, dtype=input.dtype)
        for i in range(self.n_layers):
            cache[f"cache_layer_{i}"] = torch.from_numpy(np.asarray(outs[i + 1]))
        return logits


@register("funasr_fsmn_onnx")
class FunasrFsmnOnnxVad(BaseVad):
    """FunASR FSMN-VAD with ORT encoder + original FunASR segment postprocess."""

    name = "funasr_fsmn_onnx"

    def __init__(
        self,
        model: str | Path | None = None,
        onnx_path: str | Path | None = None,
        threads: int = 1,
        device: str = "cpu",
    ) -> None:
        try:
            from funasr import AutoModel
        except ImportError as e:
            raise ImportError("funasr is required: pip install funasr") from e

        model_dir = Path(model) if model else _default_model_dir()
        onnx = Path(onnx_path) if onnx_path else _default_onnx()
        self._onnx_path = onnx
        self.threads = int(threads)

        self._am = AutoModel(
            model=str(model_dir),
            device=device,
            disable_update=True,
            disable_pbar=True,
        )
        session = make_ort_session(onnx, threads=threads)
        self._am.model.encoder = OrtFsmnEncoder(session)

    def detect(
        self,
        audio: AudioInput,
        sample_rate: int | None = None,
    ) -> list[SpeechSegment]:
        if isinstance(audio, (str, Path)):
            inp = str(audio)
        else:
            wav, _ = load_audio(audio, sample_rate)
            inp = wav

        res = self._am.generate(input=inp)
        if not res:
            return []
        value = res[0].get("value", []) if isinstance(res[0], dict) else res[0]
        segs: list[SpeechSegment] = []
        for item in value or []:
            if not item or len(item) < 2:
                continue
            start_ms, end_ms = item[0], item[1]
            if start_ms < 0 or end_ms < 0:
                continue
            segs.append(SpeechSegment(start=start_ms / 1000.0, end=end_ms / 1000.0))
        return segs
