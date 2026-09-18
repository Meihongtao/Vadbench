from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vadbench.eval.datasets import EvalSample, load_dataset
from vadbench.eval.metrics import FrameMetrics, aggregate_metrics, compute_frame_metrics
from vadbench.eval.rasterize import DEFAULT_FRAME_MS, segments_to_frames
from vadbench.registry import create

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def _duration(sample: EvalSample) -> float:
    if sample.duration_s is not None:
        return float(sample.duration_s)
    import soundfile as sf

    return float(sf.info(str(sample.audio_path)).duration)


def eval_one(
    engine: str,
    sample: EvalSample,
    vad: Any,
    frame_ms: float = DEFAULT_FRAME_MS,
) -> FrameMetrics:
    duration = _duration(sample)
    hyp = vad.detect(sample.audio_path)
    ref_f = segments_to_frames(sample.reference, duration, frame_ms=frame_ms)
    hyp_f = segments_to_frames(hyp, duration, frame_ms=frame_ms)
    n = min(len(ref_f), len(hyp_f))
    return compute_frame_metrics(ref_f[:n], hyp_f[:n], frame_ms=frame_ms)


def run_eval(
    engines: list[str],
    dataset: str = "all",
    device: str = "cpu",
    max_files: int | None = None,
    frame_ms: float = DEFAULT_FRAME_MS,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    samples = load_dataset(dataset)
    if max_files is not None:
        # keep per-dataset balance if possible
        by_ds: dict[str, list[EvalSample]] = {}
        for s in samples:
            key = s.dataset or s.lang
            by_ds.setdefault(key, []).append(s)
        capped: list[EvalSample] = []
        for _key, items in by_ds.items():
            capped.extend(items[:max_files])
        samples = capped

    out_dir = out_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frame_ms": frame_ms,
        "dataset": dataset,
        "max_files": max_files,
        "engines": engines,
        "per_file": [],
        "summary": [],
    }

    for engine in engines:
        print(f"\n=== engine={engine} ===", flush=True)
        try:
            kwargs: dict[str, Any] = {}
            if engine in ("funasr_fsmn", "firered", "pyannote", "silero"):
                kwargs["device"] = device
            vad = create(engine, **kwargs)
        except Exception as e:
            print(f"[skip] {engine}: {e}")
            report["summary"].append(
                {
                    "dataset": dataset,
                    "lang": "",
                    "model": engine,
                    "status": "skipped",
                    "error": str(e),
                }
            )
            continue

        by_key_metrics: dict[tuple[str, str], list[FrameMetrics]] = {}
        for sample in samples:
            t0 = time.time()
            try:
                m = eval_one(engine, sample, vad, frame_ms=frame_ms)
            except Exception as e:
                print(f"[error] {engine} {sample.uri}: {e}")
                report["per_file"].append(
                    {
                        "model": engine,
                        "uri": sample.uri,
                        "lang": sample.lang,
                        "dataset": sample.dataset,
                        "status": "error",
                        "error": str(e),
                    }
                )
                continue
            elapsed = time.time() - t0
            key = (sample.dataset or sample.lang, sample.lang)
            by_key_metrics.setdefault(key, []).append(m)
            row = {
                "model": engine,
                "uri": sample.uri,
                "lang": sample.lang,
                "dataset": sample.dataset,
                "status": "ok",
                "elapsed_s": round(elapsed, 3),
                **m.to_dict(),
            }
            report["per_file"].append(row)
            print(
                f"  [{sample.dataset or '-'}] {sample.lang} {sample.uri}: "
                f"f1={m.f1:.4f} far={m.far:.4f} miss={m.miss:.4f} ({elapsed:.1f}s)",
                flush=True,
            )

        for (ds_name, lg), items in by_key_metrics.items():
            w = aggregate_metrics(items, weighted=True)
            u = aggregate_metrics(items, weighted=False)
            report["summary"].append(
                {
                    "dataset": ds_name,
                    "lang": lg,
                    "model": engine,
                    "status": "ok",
                    "n_files": len(items),
                    "hours": w.duration_s / 3600.0,
                    "f1": w.f1,
                    "precision": w.precision,
                    "recall": w.recall,
                    "far": w.far,
                    "miss": w.miss,
                    "f1_unweighted": u.f1,
                    "far_unweighted": u.far,
                    "miss_unweighted": u.miss,
                }
            )

    json_path = out_dir / f"eval_{stamp}.json"
    csv_path = out_dir / f"summary_{stamp}.csv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fieldnames = [
        "dataset",
        "lang",
        "model",
        "status",
        "n_files",
        "hours",
        "f1",
        "precision",
        "recall",
        "far",
        "miss",
        "f1_unweighted",
        "far_unweighted",
        "miss_unweighted",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in report["summary"]:
            writer.writerow(row)

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    return report
