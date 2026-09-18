from __future__ import annotations

import sys
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


def _default_model_dir() -> Path | None:
    root = _project_root()
    candidates = [
        root / "models" / "firered" / "VAD",
        root / "models" / "firered" / "FireRedVAD" / "VAD",
        root / "vendor" / "FireRedVAD" / "pretrained_models" / "FireRedVAD" / "VAD",
    ]
    for p in candidates:
        if p.is_dir() and (p / "model.pth.tar").is_file():
            return p
    return None


def _ensure_firered_importable() -> None:
    try:
        import fireredvad  # noqa: F401

        return
    except ImportError:
        pass
    vendor = _project_root() / "vendor" / "FireRedVAD"
    if vendor.is_dir():
        path = str(vendor)
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        import fireredvad  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "FireRedVAD not found. Run: python -m vadbench.download --engine firered "
            "then pip install -e vendor/FireRedVAD (or keep vendor on PYTHONPATH)."
        ) from e


@register("firered")
class FireRedVadBackend(BaseVad):
    """Xiaohongshu FireRedVAD (non-streaming)."""

    name = "firered"

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str = "cpu",
        speech_threshold: float = 0.4,
        **config_kwargs,
    ) -> None:
        _ensure_firered_importable()
        from fireredvad import FireRedVad, FireRedVadConfig

        mdir = Path(model_dir) if model_dir else _default_model_dir()
        if mdir is None:
            raise FileNotFoundError(
                "FireRedVAD weights not found. "
                "Run: python -m vadbench.download --engine firered"
            )

        use_gpu = device not in ("cpu", "", None)
        cfg = FireRedVadConfig(
            use_gpu=use_gpu,
            speech_threshold=speech_threshold,
            **config_kwargs,
        )
        self._vad = FireRedVad.from_pretrained(str(mdir), cfg)

    def detect(
        self,
        audio: AudioInput,
        sample_rate: int | None = None,
    ) -> list[SpeechSegment]:
        # FireRed fbank expects int16 PCM (file path uses dtype=int16). Decode to
        # mono float first (handles multi-channel FLAC), then scale to int16.
        wav, sr = load_audio(audio, sample_rate)
        pcm16 = (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16)
        result, _probs = self._vad.detect((pcm16, sr), do_postprocess=True)

        if not result:
            return []
        timestamps = result.get("timestamps") or []
        segs: list[SpeechSegment] = []
        for pair in timestamps:
            if pair is None or len(pair) < 2:
                continue
            segs.append(SpeechSegment(start=float(pair[0]), end=float(pair[1])))
        return segs
