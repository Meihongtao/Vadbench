from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from vadbench.perf.bench import run_perf
from vadbench.perf.ort_utils import project_root

CSV_FIELDS = [
    "engine",
    "format",
    "threads",
    "model_size_mb",
    "clip_s",
    "n_runs",
    "latency_mean_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "rtf",
    "align_f1",
    "status",
]


def _parse_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ONNX end-to-end VAD performance bench")
    p.add_argument(
        "--engines",
        default="silero,firered,funasr_fsmn,pyannote,webrtc",
        help="Comma-separated engines",
    )
    p.add_argument("--threads", default="1,2,4", help="ORT thread counts")
    p.add_argument("--clip-s", type=float, default=10.0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--skip-align", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Results directory (default: results/)",
    )
    args = p.parse_args(argv)

    engines = _parse_list(args.engines)
    threads = [int(x) for x in _parse_list(args.threads)]
    rows = run_perf(
        engines=engines,
        threads=threads,
        clip_s=args.clip_s,
        warmup=args.warmup,
        runs=args.runs,
        skip_export=args.skip_export,
        skip_align=args.skip_align,
    )

    out_dir = args.out_dir or (project_root() / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"perf_{stamp}.csv"
    json_path = out_dir / f"perf_{stamp}.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = dict(row)
            for k in ("model_size_mb", "latency_mean_ms", "latency_p95_ms", "latency_p99_ms", "rtf", "align_f1"):
                if out.get(k) is not None and isinstance(out[k], float):
                    out[k] = f"{out[k]:.6f}"
            w.writerow(out)

    summary = {
        "stamp": stamp,
        "clip_s": args.clip_s,
        "warmup": args.warmup,
        "runs": args.runs,
        "threads": threads,
        "engines": engines,
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] wrote {csv_path}", flush=True)
    print(f"[ok] wrote {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
