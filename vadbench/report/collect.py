from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

_STAMP_RE = re.compile(r"eval_(\d{8}T\d{6}Z)\.json$")

# Legacy runs predate the ``dataset`` field; recover it from the uri prefix.
# Longest prefix wins, so keep longer keys first.
_URI_PREFIX_DATASET: tuple[tuple[str, str], ...] = (
    ("ami_ihm_test_", "ami_ihm_test"),
    ("alimeeting_far_test_", "alimeeting_far_test"),
    ("alimeeting_far_eval_", "alimeeting_far_eval"),
    ("aishell4_test_", "aishell4_test"),
    ("magicdata_ramc_test_", "magicdata_ramc_test"),
    ("smoke_", "smoke"),
)

# Not real benchmark sets: the 2-file sanity runs written while wiring up the
# harness. Reporting them next to 20-hour meeting sets distorts every average.
EXCLUDED_DATASETS: frozenset[str] = frozenset({"smoke", "(unknown)"})


@dataclass(frozen=True)
class FileRecord:
    """One (dataset, engine, file) evaluation result."""

    dataset: str
    engine: str
    uri: str
    lang: str
    f1: float
    precision: float
    recall: float
    far: float
    miss: float
    tp: int
    fp: int
    tn: int
    fn: int
    n_frames: int
    duration_s: float
    source: str

    @property
    def speech_ratio(self) -> float:
        """Ground-truth speech frame share ``(TP+FN)/N``.

        This is the quantity that explains why a large FAR can still mean a
        small number of false-alarm frames: FAR divides by the *non-speech*
        subset, which is tiny in meeting data.
        """
        return (self.tp + self.fn) / self.n_frames if self.n_frames else 0.0

    @property
    def fp_per_frame(self) -> float:
        """False-alarm frames as a share of *all* frames (``FP/N``)."""
        return self.fp / self.n_frames if self.n_frames else 0.0


def _infer_dataset(row: dict) -> str:
    named = (row.get("dataset") or "").strip()
    if named:
        return named
    uri = str(row.get("uri") or "")
    for prefix, name in _URI_PREFIX_DATASET:
        if uri.startswith(prefix):
            return name
    return "(unknown)"


def _stamp_of(path: Path) -> str:
    m = _STAMP_RE.search(path.name)
    return m.group(1) if m else ""


def load_records(
    results_dir: Path | None = None,
    *,
    datasets: list[str] | None = None,
    engines: list[str] | None = None,
    include_smoke: bool = False,
) -> list[FileRecord]:
    """Collect per-file results, newest run winning per (dataset, engine, uri).

    Re-running one engine on one dataset writes a fresh ``eval_*.json`` and
    leaves the older one on disk. Reading them in timestamp order and letting
    later rows overwrite earlier ones keeps the newest measurement without
    hand-maintaining a list of "which file is authoritative".

    Everything downstream aggregates these per-file counts, so a dataset-level
    number can never disagree with the per-file evidence it came from.
    """
    root = results_dir or RESULTS
    keep_ds = set(datasets) if datasets else None
    keep_en = set(engines) if engines else None

    # (dataset, engine, uri) -> FileRecord
    latest: dict[tuple[str, str, str], FileRecord] = {}
    for path in sorted(root.glob("eval_*.json"), key=_stamp_of):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"[warn] skipping unreadable {path.name}")
            continue
        for row in payload.get("per_file", []):
            if row.get("status") != "ok":
                continue
            dataset = _infer_dataset(row)
            engine = str(row.get("model") or "")
            uri = str(row.get("uri") or "")
            if not engine or not uri:
                continue
            if not include_smoke and dataset in EXCLUDED_DATASETS:
                continue
            if keep_ds is not None and dataset not in keep_ds:
                continue
            if keep_en is not None and engine not in keep_en:
                continue
            latest[(dataset, engine, uri)] = FileRecord(
                dataset=dataset,
                engine=engine,
                uri=uri,
                lang=str(row.get("lang") or ""),
                f1=float(row["f1"]),
                precision=float(row["precision"]),
                recall=float(row["recall"]),
                far=float(row["far"]),
                miss=float(row["miss"]),
                tp=int(row["tp"]),
                fp=int(row["fp"]),
                tn=int(row["tn"]),
                fn=int(row["fn"]),
                n_frames=int(row["n_frames"]),
                duration_s=float(row["duration_s"]),
                source=path.name,
            )

    return sorted(latest.values(), key=lambda r: (r.dataset, r.engine, r.uri))


def load_summary(
    results_dir: Path | None = None,
    *,
    include_smoke: bool = False,
) -> list[dict]:
    """Collect dataset-level summary rows, newest run winning.

    Early reports left ``summary[].dataset`` empty while ``per_file[].uri``
    still carried the dataset prefix. When the summary row has no dataset we
    recover it from the per-file rows of the same report, so a summary and the
    per-file evidence it describes never get attributed to different places.
    """
    root = results_dir or RESULTS
    latest: dict[tuple[str, str], dict] = {}
    for path in sorted(root.glob("eval_*.json"), key=_stamp_of):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        # engine -> dataset, inferred from this report's own per-file rows.
        by_engine: dict[str, str] = {}
        for row in payload.get("per_file", []):
            engine = str(row.get("model") or "")
            if engine and engine not in by_engine:
                by_engine[engine] = _infer_dataset(row)

        for row in payload.get("summary", []):
            if row.get("status") != "ok":
                continue
            engine = str(row.get("model") or "")
            if not engine:
                continue
            dataset = (row.get("dataset") or "").strip() or by_engine.get(engine, "(unknown)")
            if not include_smoke and dataset in EXCLUDED_DATASETS:
                continue
            latest[(dataset, engine)] = {**row, "dataset": dataset, "source": path.name}
    return sorted(latest.values(), key=lambda r: (r["dataset"], r["model"]))
