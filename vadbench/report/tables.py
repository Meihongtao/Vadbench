from __future__ import annotations

from collections import defaultdict

from vadbench.report.collect import FileRecord

# Presentation order used by the report tables (roughly hardest -> easiest).
DATASET_ORDER: tuple[str, ...] = (
    "alimeeting_far_test",
    "alimeeting_far_eval",
    "aishell4_test",
    "magicdata_ramc_test",
    "ami_ihm_test",
)

ENGINE_ORDER: tuple[str, ...] = (
    "pyannote",
    "firered",
    "funasr_fsmn",
    "silero",
    "webrtc",
)

DATASET_LABEL: dict[str, str] = {
    "alimeeting_far_test": "AliMeeting test",
    "alimeeting_far_eval": "AliMeeting eval",
    "aishell4_test": "AISHELL-4",
    "magicdata_ramc_test": "MagicData-RAMC",
    "ami_ihm_test": "AMI IHM",
}

F1_LOW_THRESHOLD = 0.9


def _sort_key(value: str, order: tuple[str, ...]) -> tuple[int, str]:
    return (order.index(value) if value in order else len(order), value)


def _percentile(sorted_xs: list[float], q: float) -> float:
    """Linear-interpolated percentile; ``q`` in [0, 100]."""
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = (len(sorted_xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = pos - lo
    return sorted_xs[lo] * (1.0 - frac) + sorted_xs[hi] * frac


def per_file_table(
    records: list[FileRecord],
    *,
    datasets: list[str] | None = None,
    engines: list[str] | None = None,
) -> list[dict]:
    """Per-file F1 distribution for each dataset / engine pair.

    A weighted F1 hides whether a run is uniformly good or mostly good with a
    few collapses; ``min``/``median``/``n_f1_below`` make that visible.
    """
    buckets: dict[tuple[str, str], list[FileRecord]] = defaultdict(list)
    for r in records:
        buckets[(r.dataset, r.engine)].append(r)

    rows: list[dict] = []
    for (dataset, engine), items in buckets.items():
        if datasets and dataset not in datasets:
            continue
        if engines and engine not in engines:
            continue
        f1s = sorted(i.f1 for i in items)
        worst = min(items, key=lambda i: i.f1)
        rows.append(
            {
                "dataset": dataset,
                "engine": engine,
                "n_files": len(items),
                "hours": sum(i.duration_s for i in items) / 3600.0,
                "f1_min": f1s[0],
                "f1_p25": _percentile(f1s, 25),
                "f1_median": _percentile(f1s, 50),
                "f1_mean": sum(f1s) / len(f1s),
                "f1_p75": _percentile(f1s, 75),
                "f1_max": f1s[-1],
                "n_f1_below": sum(1 for f in f1s if f < F1_LOW_THRESHOLD),
                "f1_low_threshold": F1_LOW_THRESHOLD,
                "worst_uri": worst.uri,
            }
        )

    rows.sort(
        key=lambda r: (
            _sort_key(r["dataset"], DATASET_ORDER),
            _sort_key(r["engine"], ENGINE_ORDER),
        )
    )
    return rows


def far_table(records: list[FileRecord]) -> list[dict]:
    """Two readings of false alarms, side by side.

    ``far`` divides by non-speech frames only (the standard FPR). In meeting
    data speech dominates, so a tiny non-speech denominator inflates FAR.
    ``fp_per_frame`` divides by all frames and is the number that maps onto
    "how many junk frames get sent downstream".
    """
    buckets: dict[tuple[str, str], list[FileRecord]] = defaultdict(list)
    for r in records:
        buckets[(r.dataset, r.engine)].append(r)

    rows: list[dict] = []
    for (dataset, engine), items in buckets.items():
        tp = sum(i.tp for i in items)
        fp = sum(i.fp for i in items)
        tn = sum(i.tn for i in items)
        fn = sum(i.fn for i in items)
        n = tp + fp + tn + fn
        if not n:
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "dataset": dataset,
                "engine": engine,
                "n_files": len(items),
                "hours": sum(i.duration_s for i in items) / 3600.0,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "far": fp / (fp + tn) if (fp + tn) else 0.0,
                "fp_per_frame": fp / n,
                "speech_ratio": (tp + fn) / n,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "n_frames": n,
            }
        )

    rows.sort(
        key=lambda r: (
            _sort_key(r["dataset"], DATASET_ORDER),
            _sort_key(r["engine"], ENGINE_ORDER),
        )
    )
    return rows


def cross_dataset_f1(
    records: list[FileRecord],
    *,
    datasets: list[str] | None = None,
    engines: list[str] | None = None,
    metric: str = "f1",
) -> tuple[list[str], list[dict]]:
    """Pivot into ``engine x dataset`` matrix of duration-weighted metrics.

    Counts are re-summed from the per-file records rather than read from
    ``summary`` rows: that keeps the pivot identical to ``far_table`` and
    works for older reports whose summary lacks a ``dataset`` field.

    Returns ``(dataset_headers, rows)`` where each row has ``engine`` plus one
    key per dataset. Missing cells are ``None``.
    """
    buckets: dict[tuple[str, str], list[FileRecord]] = defaultdict(list)
    for r in records:
        if datasets and r.dataset not in datasets:
            continue
        if engines and r.engine not in engines:
            continue
        buckets[(r.dataset, r.engine)].append(r)

    seen_datasets = {d for d, _ in buckets}
    headers = sorted(seen_datasets, key=lambda d: _sort_key(d, DATASET_ORDER))

    grid: dict[str, dict[str, float]] = defaultdict(dict)
    for (dataset, engine), items in buckets.items():
        tp = sum(i.tp for i in items)
        fp = sum(i.fp for i in items)
        tn = sum(i.tn for i in items)
        fn = sum(i.fn for i in items)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        values = {
            "f1": (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            ),
            "precision": precision,
            "recall": recall,
            "far": fp / (fp + tn) if (fp + tn) else 0.0,
            "miss": fn / (fn + tp) if (fn + tp) else 0.0,
        }
        if metric not in values:
            raise KeyError(f"unknown metric={metric!r}; choose from {sorted(values)}")
        grid[engine][dataset] = values[metric]

    out: list[dict] = []
    for engine in sorted(grid, key=lambda e: _sort_key(e, ENGINE_ORDER)):
        entry: dict = {"engine": engine}
        entry.update({d: grid[engine].get(d) for d in headers})
        out.append(entry)
    return headers, out


def best_per_dataset(
    headers: list[str],
    rows: list[dict],
    *,
    lower_is_better: bool = False,
) -> dict[str, str]:
    """Return ``dataset -> engine`` that wins, used to bold the winner."""
    winners: dict[str, str] = {}
    for dataset in headers:
        scored = [
            (row[dataset], str(row["engine"]))
            for row in rows
            if row.get(dataset) is not None
        ]
        if not scored:
            continue
        best = (min if lower_is_better else max)(scored)
        winners[dataset] = best[1]
    return winners
