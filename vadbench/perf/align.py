from __future__ import annotations

from pathlib import Path

from vadbench.eval.metrics import compute_frame_metrics
from vadbench.eval.rasterize import DEFAULT_FRAME_MS, segments_to_frames
from vadbench.perf.ort_utils import ensure_clip
from vadbench.registry import create

# original backend kwargs / onnx backend name
ALIGN_SPECS: dict[str, tuple[str, dict, str]] = {
    # engine -> (ref_name, ref_kwargs, onnx_name)
    "silero": ("silero", {"onnx": False}, "silero_onnx"),
    "firered": ("firered", {}, "firered_onnx"),
    "funasr_fsmn": ("funasr_fsmn", {}, "funasr_fsmn_onnx"),
    "pyannote": ("pyannote", {}, "pyannote_onnx"),
}

ALIGN_F1_MIN = 0.99


def align_engine(
    engine: str,
    clip: Path | None = None,
    clip_s: float = 10.0,
    frame_ms: float = DEFAULT_FRAME_MS,
    threads: int = 1,
) -> dict:
    """Compare original vs ONNX end-to-end segments on a fixed clip."""
    if engine not in ALIGN_SPECS:
        raise KeyError(f"No align spec for {engine}")
    ref_name, ref_kw, onnx_name = ALIGN_SPECS[engine]
    wav = clip or ensure_clip(clip_s)

    ref = create(ref_name, **ref_kw)
    hyp = create(onnx_name, threads=threads)

    ref_segs = ref.detect(wav)
    hyp_segs = hyp.detect(wav)
    duration = float(clip_s)
    ref_f = segments_to_frames(ref_segs, duration, frame_ms=frame_ms)
    hyp_f = segments_to_frames(hyp_segs, duration, frame_ms=frame_ms)
    if ref_f.shape == hyp_f.shape and (ref_f == hyp_f).all():
        # Identical frames (incl. both-empty) count as perfect align.
        f1 = 1.0
        precision = 1.0
        recall = 1.0
        ok = True
    else:
        m = compute_frame_metrics(ref_f, hyp_f, frame_ms=frame_ms)
        f1, precision, recall = m.f1, m.precision, m.recall
        ok = f1 >= ALIGN_F1_MIN
    return {
        "engine": engine,
        "clip": str(wav),
        "ref_segments": [(s.start, s.end) for s in ref_segs],
        "onnx_segments": [(s.start, s.end) for s in hyp_segs],
        "align_f1": f1,
        "align_precision": precision,
        "align_recall": recall,
        "status": "ok" if ok else "fail",
        "threshold": ALIGN_F1_MIN,
    }


def align_all(
    engines: list[str] | None = None,
    clip: Path | None = None,
    clip_s: float = 10.0,
) -> list[dict]:
    names = engines or list(ALIGN_SPECS)
    out = []
    for name in names:
        if name == "webrtc":
            continue
        print(f"[align] {name} ...", flush=True)
        row = align_engine(name, clip=clip, clip_s=clip_s)
        print(
            f"[align] {name} F1={row['align_f1']:.6f} status={row['status']}",
            flush=True,
        )
        out.append(row)
    return out
