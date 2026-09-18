from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16000


def load_audio(
    audio: str | Path | np.ndarray,
    sample_rate: int | None = None,
    target_sr: int = TARGET_SR,
) -> tuple[np.ndarray, int]:
    """Load mono float32 audio and resample to ``target_sr`` if needed."""
    if isinstance(audio, (str, Path)):
        wav, sr = sf.read(str(audio), always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32, copy=False)
    else:
        if sample_rate is None:
            raise ValueError("sample_rate is required when audio is an ndarray")
        wav = np.asarray(audio, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        sr = int(sample_rate)

    if sr != target_sr:
        wav = _resample_linear(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def float_to_pcm16(wav: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] to little-endian int16 PCM bytes."""
    clipped = np.clip(wav, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    return pcm.tobytes()


def _resample_linear(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or len(wav) == 0:
        return wav.astype(np.float32, copy=False)
    duration = len(wav) / float(orig_sr)
    n_target = max(1, int(round(duration * target_sr)))
    x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_target, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)
