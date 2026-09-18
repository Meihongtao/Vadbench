from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from vadbench.audio import load_audio
from vadbench.base import BaseVad
from vadbench.registry import register
from vadbench.types import SpeechSegment

AudioInput = Union[str, Path, np.ndarray]


def _default_model_dir() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    p = root / "models" / "funasr_fsmn"
    if p.is_dir() and any(p.iterdir()):
        return p
    return None


@register("funasr_fsmn")
class FunasrFsmnVad(BaseVad):
    """FunASR FSMN-VAD (offline segments in milliseconds)."""

    name = "funasr_fsmn"

    def __init__(
        self,
        model: str | Path | None = None,
        device: str = "cpu",
        hub: str | None = None,
    ) -> None:
        try:
            from funasr import AutoModel
        except ImportError as e:
            raise ImportError("funasr is required: pip install funasr") from e

        local = _default_model_dir()
        if model is not None:
            model_id = str(model)
        elif local is not None:
            model_id = str(local)
        else:
            model_id = "fsmn-vad"

        kwargs = {
            "model": model_id,
            "device": device,
            "disable_update": True,
            "disable_pbar": True,
        }
        if hub:
            kwargs["hub"] = hub
        self._model = AutoModel(**kwargs)

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

        res = self._model.generate(input=inp)
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
