from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"


def project_root() -> Path:
    return ROOT


def make_ort_session(onnx_path: str | Path, threads: int = 1):
    """Create an ONNXRuntime session with fixed thread budget."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = int(threads)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )


def model_size_mb(path: str | Path | None) -> float | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return p.stat().st_size / (1024.0 * 1024.0)


def ensure_clip(clip_s: float = 10.0, out_path: Path | None = None) -> Path:
    """Materialize a fixed-length 16 kHz mono wav for perf/align.

    Prefers a high-energy slice so VAD engines produce non-empty segments.
    """
    import numpy as np
    import soundfile as sf

    out = out_path or (ROOT / "data" / "_perf" / f"clip_{int(clip_s)}s.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        info = sf.info(str(out))
        if abs(float(info.duration) - clip_s) < 0.05 and int(info.samplerate) == 16000:
            wav_chk, _ = sf.read(str(out), always_2d=False)
            if wav_chk.ndim > 1:
                wav_chk = wav_chk.mean(axis=1)
            rms = float(np.sqrt(np.mean(np.asarray(wav_chk, dtype=np.float64) ** 2)))
            if rms >= 0.01:
                return out

    candidates = [
        ROOT / "data" / "ami_ihm_test" / "audio" / "ami_ihm_test_000.wav",
        ROOT / "data" / "alimeeting_far_eval" / "audio",
        ROOT / "data" / "alimeeting_far_test" / "audio",
    ]
    src = None
    for c in candidates:
        if c.is_file():
            src = c
            break
        if c.is_dir():
            wavs = sorted(c.glob("*.wav"))
            if wavs:
                src = wavs[0]
                break
    if src is None:
        raise FileNotFoundError("No source audio found under data/ for perf clip")

    wav, sr = sf.read(str(src), always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype=np.float32)
    n = int(round(clip_s * sr))
    if len(wav) < n:
        wav = np.pad(wav, (0, n - len(wav)))
    else:
        # Prefer mid-file slice (often has speech vs leading silence).
        best_i = max(0, (len(wav) - n) // 2)
        best_rms = -1.0
        step = max(1, n // 2)
        for i in range(0, max(1, len(wav) - n + 1), step):
            chunk = wav[i : i + n]
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            if rms > best_rms:
                best_rms = rms
                best_i = i
        wav = wav[best_i : best_i + n]
    if sr != 16000:
        duration = len(wav) / float(sr)
        n16 = max(1, int(round(duration * 16000)))
        x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n16, endpoint=False)
        wav = np.interp(x_new, x_old, wav).astype(np.float32)
        n = int(round(clip_s * 16000))
        if len(wav) < n:
            wav = np.pad(wav, (0, n - len(wav)))
        else:
            wav = wav[:n]
        sr = 16000
    sf.write(str(out), wav, 16000)
    return out
