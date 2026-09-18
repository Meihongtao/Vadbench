from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from vadbench.perf.align import ALIGN_SPECS, ALIGN_F1_MIN, align_engine
from vadbench.perf.export_onnx import EXPORTERS, export_all
from vadbench.perf.ort_utils import ensure_clip, model_size_mb, project_root
from vadbench.registry import create

ONNX_BACKEND = {
    "silero": "silero_onnx",
    "firered": "firered_onnx",
    "funasr_fsmn": "funasr_fsmn_onnx",
    "pyannote": "pyannote_onnx",
}

ONNX_PATHS = {
    "silero": lambda: project_root() / "models" / "silero" / "onnx" / "silero_vad.onnx",
    "firered": lambda: project_root()
    / "models"
    / "firered"
    / "onnx"
    / "fireredvad_vad.onnx",
    "funasr_fsmn": lambda: project_root()
    / "models"
    / "funasr_fsmn"
    / "onnx"
    / "model.onnx",
    "pyannote": lambda: project_root()
    / "models"
    / "pyannote"
    / "onnx"
    / "segmentation-3.0.onnx",
}


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))


def _latency_stats(
    detect_fn,
    warmup: int,
    runs: int,
) -> dict[str, float]:
    for _ in range(max(0, warmup)):
        detect_fn()
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        detect_fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "latency_mean_ms": float(np.mean(times)),
        "latency_p95_ms": _percentile(times, 95),
        "latency_p99_ms": _percentile(times, 99),
        "n_runs": float(runs),
    }


def _onnx_file(engine: str) -> Path | None:
    if engine not in ONNX_PATHS:
        return None
    p = ONNX_PATHS[engine]()
    # fall back to known locations
    if not p.is_file():
        alts = {
            "silero": [
                project_root() / "models" / "silero" / "silero_vad.onnx",
            ],
            "firered": [
                project_root()
                / "models"
                / "firered"
                / "onnx_models"
                / "fireredvad_vad.onnx",
            ],
        }
        for a in alts.get(engine, []):
            if a.is_file():
                return a
        return p
    return p


def bench_engine(
    engine: str,
    threads: int | None,
    clip: Path,
    clip_s: float,
    warmup: int,
    runs: int,
    align_f1: float | None = None,
    status: str = "ok",
) -> dict:
    if engine == "webrtc":
        vad = create("webrtc")
        stats = _latency_stats(lambda: vad.detect(clip), warmup=warmup, runs=runs)
        rtf = stats["latency_mean_ms"] / 1000.0 / clip_s
        return {
            "engine": "webrtc",
            "format": "native",
            "threads": "n/a",
            "model_size_mb": None,
            "clip_s": clip_s,
            "n_runs": int(runs),
            "latency_mean_ms": stats["latency_mean_ms"],
            "latency_p95_ms": stats["latency_p95_ms"],
            "latency_p99_ms": stats["latency_p99_ms"],
            "rtf": rtf,
            "align_f1": None,
            "status": "ok",
        }

    if status != "ok":
        onnx_p = _onnx_file(engine)
        return {
            "engine": engine,
            "format": "onnx",
            "threads": threads,
            "model_size_mb": model_size_mb(onnx_p),
            "clip_s": clip_s,
            "n_runs": 0,
            "latency_mean_ms": None,
            "latency_p95_ms": None,
            "latency_p99_ms": None,
            "rtf": None,
            "align_f1": align_f1,
            "status": status,
        }

    backend = ONNX_BACKEND[engine]
    th = 1 if threads is None else int(threads)
    vad = create(backend, threads=th)
    stats = _latency_stats(lambda: vad.detect(clip), warmup=warmup, runs=runs)
    rtf = stats["latency_mean_ms"] / 1000.0 / clip_s
    onnx_p = _onnx_file(engine)
    return {
        "engine": engine,
        "format": "onnx",
        "threads": th,
        "model_size_mb": model_size_mb(onnx_p),
        "clip_s": clip_s,
        "n_runs": int(runs),
        "latency_mean_ms": stats["latency_mean_ms"],
        "latency_p95_ms": stats["latency_p95_ms"],
        "latency_p99_ms": stats["latency_p99_ms"],
        "rtf": rtf,
        "align_f1": align_f1,
        "status": "ok",
    }


def run_perf(
    engines: list[str] | None = None,
    threads: list[int] | None = None,
    clip_s: float = 10.0,
    warmup: int = 5,
    runs: int = 50,
    skip_export: bool = False,
    skip_align: bool = False,
) -> list[dict]:
    names = engines or (list(EXPORTERS) + ["webrtc"])
    th_list = threads or [1, 2, 4]
    clip = ensure_clip(clip_s)

    nn_engines = [e for e in names if e != "webrtc"]
    if not skip_export and nn_engines:
        export_all([e for e in nn_engines if e in EXPORTERS])

    align_map: dict[str, dict] = {}
    if not skip_align:
        for e in nn_engines:
            if e not in ALIGN_SPECS:
                continue
            print(f"[align] {e} ...", flush=True)
            row = align_engine(e, clip=clip, clip_s=clip_s)
            print(
                f"[align] {e} F1={row['align_f1']:.6f} status={row['status']}",
                flush=True,
            )
            align_map[e] = row

    rows: list[dict] = []
    for e in names:
        if e == "webrtc":
            print("[bench] webrtc native ...", flush=True)
            rows.append(
                bench_engine(
                    "webrtc",
                    threads=None,
                    clip=clip,
                    clip_s=clip_s,
                    warmup=warmup,
                    runs=runs,
                )
            )
            continue

        a = align_map.get(e)
        align_f1 = a["align_f1"] if a else None
        status = a["status"] if a else ("ok" if skip_align else "fail")
        if a and a["status"] != "ok":
            print(
                f"[bench] skip {e}: align F1={align_f1:.6f} < {ALIGN_F1_MIN}",
                flush=True,
            )
            rows.append(
                bench_engine(
                    e,
                    threads=th_list[0],
                    clip=clip,
                    clip_s=clip_s,
                    warmup=warmup,
                    runs=runs,
                    align_f1=align_f1,
                    status="fail_align",
                )
            )
            continue

        for th in th_list:
            print(f"[bench] {e} onnx threads={th} ...", flush=True)
            rows.append(
                bench_engine(
                    e,
                    threads=th,
                    clip=clip,
                    clip_s=clip_s,
                    warmup=warmup,
                    runs=runs,
                    align_f1=align_f1,
                    status="ok",
                )
            )
    return rows
