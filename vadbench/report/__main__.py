from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from vadbench.report.collect import RESULTS, load_records, load_summary
from vadbench.report.tables import (
    DATASET_LABEL,
    best_per_dataset,
    cross_dataset_f1,
    far_table,
    per_file_table,
)

PER_FILE_FIELDS = [
    "dataset",
    "engine",
    "n_files",
    "hours",
    "f1_min",
    "f1_p25",
    "f1_median",
    "f1_mean",
    "f1_p75",
    "f1_max",
    "n_f1_below",
    "f1_low_threshold",
    "worst_uri",
]

FAR_FIELDS = [
    "dataset",
    "engine",
    "n_files",
    "hours",
    "f1",
    "precision",
    "recall",
    "far",
    "fp_per_frame",
    "speech_ratio",
    "tp",
    "fp",
    "tn",
    "fn",
    "n_frames",
]

CROSS_FIELDS = ["dataset", "best_engine", "best_f1", "runner_up_engine", "runner_up_f1"]


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for k, v in out.items():
                if isinstance(v, float):
                    out[k] = f"{v:.6f}"
            writer.writerow(out)


def _print_per_file(rows: list[dict]) -> None:
    print("\n== per-file F1 distribution ==")
    print(
        f"{'dataset':<22}{'engine':<14}{'files':>6}{'min':>8}{'p25':>8}"
        f"{'median':>8}{'mean':>8}{'max':>8}{'<0.9':>7}"
    )
    current = None
    for r in rows:
        if r["dataset"] != current:
            current = r["dataset"]
            print("-" * 81)
        print(
            f"{DATASET_LABEL.get(r['dataset'], r['dataset']):<22}"
            f"{r['engine']:<14}{r['n_files']:>6}"
            f"{r['f1_min']:>8.3f}{r['f1_p25']:>8.3f}{r['f1_median']:>8.3f}"
            f"{r['f1_mean']:>8.3f}{r['f1_max']:>8.3f}"
            f"{r['n_f1_below']:>7}"
        )


def _print_far(rows: list[dict]) -> None:
    print("\n== false alarms: FAR vs FP/N ==")
    print(
        f"{'dataset':<22}{'engine':<14}{'P':>7}{'R':>7}{'FAR':>9}"
        f"{'FP/N':>9}{'speech%':>9}"
    )
    current = None
    for r in rows:
        if r["dataset"] != current:
            current = r["dataset"]
            print("-" * 77)
        print(
            f"{DATASET_LABEL.get(r['dataset'], r['dataset']):<22}"
            f"{r['engine']:<14}{r['precision']:>7.3f}{r['recall']:>7.3f}"
            f"{r['far']:>9.3f}{r['fp_per_frame']:>9.4f}"
            f"{r['speech_ratio'] * 100:>8.1f}%"
        )


def _print_cross(headers: list[str], rows: list[dict]) -> None:
    print("\n== cross-dataset weighted F1 ==")
    label = [DATASET_LABEL.get(h, h) for h in headers]
    winners = best_per_dataset(headers, rows)
    width = 22
    print(f"{'engine':<14}" + "".join(f"{h[: width - 4]:>{width}}" for h in label))
    for row in rows:
        cells = []
        for h in headers:
            v = row.get(h)
            text = "—" if v is None else f"{v:.3f}"
            if winners.get(h) == row["engine"]:
                text += "*"
            cells.append(f"{text:>{width}}")
        print(f"{row['engine']:<14}" + "".join(cells))
    print("  * = best in column")


def _check_against_summary(records, summary_rows, tol: float = 1e-6) -> list[str]:
    """Verify recomputed metrics match what the runner reported.

    The report re-sums per-file confusion counts instead of trusting the
    ``summary`` block. That is only safe if both paths agree, so compare them
    and surface any mismatch rather than silently publishing a different
    number than ``results/summary_*.csv``.

    A summary built from a *subset* of the files (e.g. a 3-file spot check that
    wrote its own ``eval_*.json``) is not a contradiction, only a different
    scope, so it is skipped via the ``n_files`` guard.
    """
    reported = {(r["dataset"], r["model"]): r for r in summary_rows}

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in records:
        counts[(r.dataset, r.engine)] += 1

    _, recomputed = cross_dataset_f1(records)
    problems: list[str] = []
    for row in recomputed:
        engine = row["engine"]
        for dataset, value in row.items():
            if dataset == "engine" or value is None:
                continue
            ref = reported.get((dataset, engine))
            if ref is None:
                problems.append(f"{dataset}/{engine}: no summary row to compare")
                continue
            if int(ref.get("n_files") or 0) != counts[(dataset, engine)]:
                continue
            ref_f1 = float(ref["f1"])
            if abs(ref_f1 - value) > tol:
                problems.append(
                    f"{dataset}/{engine}: summary f1={ref_f1:.6f} "
                    f"!= recomputed {value:.6f}"
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-analyse results/eval_*.json: per-file F1 spread, FAR divisors, "
            "cross-dataset pivot. Does not re-run any model."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory holding eval_*.json (default: results/)",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset keys to keep (default: all found).",
    )
    parser.add_argument(
        "--engines",
        default=None,
        help="Comma-separated engine names to keep (default: all found).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write report_*.csv (default: results/report/).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Print to stdout only.",
    )
    args = parser.parse_args(argv)

    datasets = [x.strip() for x in args.datasets.split(",")] if args.datasets else None
    engines = [x.strip() for x in args.engines.split(",")] if args.engines else None

    records = load_records(args.results_dir, datasets=datasets, engines=engines)
    if not records:
        print("No per-file records found under results/. Run the eval first:")
        print("  python -m vadbench.eval --engines ... --dataset ...")
        return 1

    summary_rows = load_summary(args.results_dir)
    pf = per_file_table(records, datasets=datasets, engines=engines)
    far = far_table(records)
    headers, cross = cross_dataset_f1(records, datasets=datasets, engines=engines)

    _print_per_file(pf)
    _print_far(far)
    if headers:
        _print_cross(headers, cross)

    problems = _check_against_summary(records, summary_rows)
    if problems:
        print(f"\n[warn] {len(problems)} summary/recompute mismatch(es):")
        for line in problems[:10]:
            print(f"  - {line}")
        if len(problems) > 10:
            print(f"  ... and {len(problems) - 10} more")
    else:
        print("\n[ok] recomputed metrics match results/summary_*.csv")

    if not args.no_csv:
        out_dir = args.out_dir or (RESULTS / "report")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _write_csv(out_dir / f"per_file_f1_{stamp}.csv", PER_FILE_FIELDS, pf)
        _write_csv(out_dir / f"far_detail_{stamp}.csv", FAR_FIELDS, far)

        winners = best_per_dataset(headers, cross) if headers else {}
        cross_rows = []
        for dataset in headers:
            scored = sorted(
                (
                    (row[dataset], str(row["engine"]))
                    for row in cross
                    if row.get(dataset) is not None
                ),
                reverse=True,
            )
            cross_rows.append(
                {
                    "dataset": dataset,
                    "best_engine": winners.get(dataset, ""),
                    "best_f1": scored[0][0] if scored else None,
                    "runner_up_engine": scored[1][1] if len(scored) > 1 else "",
                    "runner_up_f1": scored[1][0] if len(scored) > 1 else None,
                }
            )
        _write_csv(out_dir / f"cross_dataset_{stamp}.csv", CROSS_FIELDS, cross_rows)

        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_records": len(records),
            "per_file": pf,
            "far": far,
            "cross_dataset": {"datasets": headers, "rows": cross},
        }
        json_path = out_dir / f"report_{stamp}.json"
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {out_dir}/per_file_f1_{stamp}.csv")
        print(f"Wrote {out_dir}/far_detail_{stamp}.csv")
        if headers:
            print(f"Wrote {out_dir}/cross_dataset_{stamp}.csv")
        print(f"Wrote {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
