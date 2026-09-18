from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

import numpy as np

from vadbench.audio import load_audio
from vadbench.base import BaseVad
from vadbench.perf.ort_utils import MODELS, make_ort_session, project_root
from vadbench.registry import register
from vadbench.types import SpeechSegment

AudioInput = Union[str, Path, np.ndarray]


def _ensure_firered_importable() -> None:
    try:
        import fireredvad  # noqa: F401

        return
    except ImportError:
        pass
    vendor = project_root() / "vendor" / "FireRedVAD"
    if vendor.is_dir():
        path = str(vendor)
        if path not in sys.path:
            sys.path.insert(0, path)
    import fireredvad  # noqa: F401


def _vad_dir() -> Path:
    candidates = [
        MODELS / "firered" / "VAD",
        MODELS / "firered" / "FireRedVAD" / "VAD",
        project_root() / "vendor" / "FireRedVAD" / "pretrained_models" / "FireRedVAD" / "VAD",
    ]
    for p in candidates:
        if p.is_dir() and (p / "cmvn.ark").is_file():
            return p
    raise FileNotFoundError("FireRed VAD dir (cmvn.ark) not found")


def _default_onnx() -> Path:
    for p in (
        MODELS / "firered" / "onnx" / "fireredvad_vad.onnx",
        MODELS / "firered" / "onnx_models" / "fireredvad_vad.onnx",
    ):
        if p.is_file():
            return p
    raise FileNotFoundError("FireRed ONNX not found; run vadbench.perf.export_onnx")


@register("firered_onnx")
class FireRedOnnxVad(BaseVad):
    """FireRedVAD: AudioFeat + ORT + VadPostprocessor."""

    name = "firered_onnx"

    def __init__(
        self,
        model_path: str | Path | None = None,
        vad_dir: str | Path | None = None,
        threads: int = 1,
        speech_threshold: float = 0.4,
        smooth_window_size: int = 5,
        min_speech_frame: int = 20,
        max_speech_frame: int = 2000,
        min_silence_frame: int = 20,
        merge_silence_frame: int = 0,
        extend_speech_frame: int = 0,
        chunk_max_frame: int = 30000,
    ) -> None:
        _ensure_firered_importable()
        from fireredvad.core.audio_feat import AudioFeat
        from fireredvad.core.vad_postprocessor import VadPostprocessor

        mdir = Path(vad_dir) if vad_dir else _vad_dir()
        onnx_path = Path(model_path) if model_path else _default_onnx()
        self._onnx_path = onnx_path
        self.threads = int(threads)
        self.chunk_max_frame = int(chunk_max_frame)
        self._feat = AudioFeat(str(mdir / "cmvn.ark"))
        self._session = make_ort_session(onnx_path, threads=threads)
        self._in_name = self._session.get_inputs()[0].name
        self._post = VadPostprocessor(
            smooth_window_size,
            speech_threshold,
            min_speech_frame,
            max_speech_frame,
            min_silence_frame,
            merge_silence_frame,
            extend_speech_frame,
        )

    def detect(
        self,
        audio: AudioInput,
        sample_rate: int | None = None,
    ) -> list[SpeechSegment]:
        if isinstance(audio, (str, Path)):
            feats, dur = self._feat.extract(str(audio))
        else:
            wav, _ = load_audio(audio, sample_rate)
            feats, dur = self._feat.extract(wav)

        feat_np = feats.detach().cpu().numpy().astype(np.float32)
        if feat_np.ndim == 2:
            # (T, 80) -> chunks
            if feat_np.shape[0] <= self.chunk_max_frame:
                probs = self._session.run(
                    None, {self._in_name: feat_np[np.newaxis, ...]}
                )[0]
                probs = np.asarray(probs).reshape(-1)
            else:
                parts = []
                for i in range(0, feat_np.shape[0], self.chunk_max_frame):
                    chunk = feat_np[i : i + self.chunk_max_frame]
                    p = self._session.run(
                        None, {self._in_name: chunk[np.newaxis, ...]}
                    )[0]
                    parts.append(np.asarray(p).reshape(-1))
                probs = np.concatenate(parts, axis=0)
        else:
            raise ValueError(f"Unexpected feat shape: {feat_np.shape}")

        decisions = self._post.process(probs.tolist())
        starts_ends = self._post.decision_to_segment(decisions, dur)
        return [
            SpeechSegment(start=float(a), end=float(b))
            for a, b in (starts_ends or [])
            if a is not None and b is not None
        ]
