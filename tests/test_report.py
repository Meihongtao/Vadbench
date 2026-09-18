from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Allow running this file directly (``python tests/test_report.py``) without an
# editable install, which is how CI invokes it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vadbench.report.collect import load_records, load_summary  # noqa: E402
from vadbench.report.tables import (  # noqa: E402
    best_per_dataset,
    cross_dataset_f1,
    far_table,
    per_file_table,
)


def _make_report(path: Path, rows: list[dict], dataset: str = "") -> None:
    per_file = []
    for i, row in enumerate(rows):
        per_file.append(
            {
                "model": row["model"],
                "uri": row.get("uri", f"{dataset}_file_{i:03d}"),
                "lang": "zh",
                "dataset": dataset,
                "status": "ok",
                "f1": row["f1"],
                "precision": row["precision"],
                "recall": row["recall"],
                "far": row["far"],
                "miss": row["miss"],
                "tp": row["tp"],
                "fp": row["fp"],
                "tn": row["tn"],
                "fn": row["fn"],
                "n_frames": row["tp"] + row["fp"] + row["tn"] + row["fn"],
                "duration_s": 600.0,
            }
        )
    path.write_text(json.dumps({"per_file": per_file, "summary": []}), encoding="utf-8")


def test_counts_are_sums_of_files() -> None:
    """A dataset number must be derivable from its files, not a separate claim."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_report(
            root / "eval_20260101T000000Z.json",
            [
                # short, badly handled file: F1 = 0.5
                {
                    "model": "a",
                    "f1": 0.5,
                    "precision": 0.5,
                    "recall": 0.5,
                    "far": 0.05,
                    "miss": 0.5,
                    "tp": 50,
                    "fp": 50,
                    "tn": 950,
                    "fn": 50,
                },
                # long, clean file: F1 = 1.0
                {
                    "model": "a",
                    "f1": 1.0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "far": 0.0,
                    "miss": 0.0,
                    "tp": 1000,
                    "fp": 0,
                    "tn": 900,
                    "fn": 0,
                },
            ],
            dataset="ds",
        )
        far = far_table(load_records(root))
        assert len(far) == 1, far
        assert (far[0]["tp"], far[0]["fp"], far[0]["fn"]) == (1050, 50, 50), far[0]
        expected = 2 * 1050 / (2 * 1050 + 50 + 50)
        assert abs(far[0]["f1"] - expected) < 1e-12, (far[0]["f1"], expected)
        # duration weighting must beat the 0.75 arithmetic mean of the two files
        assert far[0]["f1"] > 0.75, far[0]["f1"]


def test_newest_run_wins() -> None:
    """Re-running one engine must supersede the stale row, not duplicate it."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = {
            "model": "a",
            "uri": "ds_file_000",
            "precision": 0.5,
            "recall": 0.5,
            "far": 0.1,
            "miss": 0.5,
            "tp": 50,
            "fp": 50,
            "tn": 900,
            "fn": 50,
        }
        _make_report(root / "eval_20260101T000000Z.json", [{**base, "f1": 0.10}], dataset="ds")
        _make_report(root / "eval_20260102T000000Z.json", [{**base, "f1": 0.90}], dataset="ds")
        far = far_table(load_records(root))
        assert len(far) == 1, f"expected one merged row, got {far}"
        assert far[0]["n_files"] == 1, far
        assert abs(far[0]["f1"] - 0.5) < 1e-12, "counts win; f1 is recomputed from them"


def test_smoke_runs_excluded() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        row = {
            "model": "a",
            "f1": 0.9,
            "precision": 0.9,
            "recall": 0.9,
            "far": 0.1,
            "miss": 0.1,
            "tp": 90,
            "fp": 10,
            "tn": 900,
            "fn": 10,
        }
        _make_report(root / "eval_20260101T000000Z.json", [{**row, "uri": "smoke_zh"}])
        assert load_records(root) == []
        assert len(load_records(root, include_smoke=True)) == 1


def test_legacy_summary_dataset_recovery() -> None:
    """Old reports omit summary.dataset; recover it from their per_file rows."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        payload = {
            "per_file": [
                {
                    "model": "a",
                    "uri": "ami_ihm_test_000",
                    "lang": "en",
                    "status": "ok",
                    "f1": 0.9,
                    "precision": 0.9,
                    "recall": 0.9,
                    "far": 0.1,
                    "miss": 0.1,
                    "tp": 90,
                    "fp": 10,
                    "tn": 900,
                    "fn": 10,
                    "n_frames": 1010,
                    "duration_s": 10.1,
                }
            ],
            "summary": [{"dataset": "", "model": "a", "status": "ok", "f1": 0.9, "n_files": 1}],
        }
        (root / "eval_20260101T000000Z.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        summary = load_summary(root)
        assert len(summary) == 1, summary
        assert summary[0]["dataset"] == "ami_ihm_test", summary[0]


def test_far_divisor_trap() -> None:
    """Equal FP counts give very different FAR under different speech shares."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = [
            # 90% speech: only 100 non-speech frames, so 50 FP -> FAR 0.50
            {
                "model": "dense",
                "f1": 0.9,
                "precision": 0.9,
                "recall": 0.9,
                "far": 0.5,
                "miss": 0.1,
                "tp": 850,
                "fp": 50,
                "tn": 50,
                "fn": 50,
            },
            # 50% speech: 500 non-speech frames, same 50 FP -> FAR 0.10
            {
                "model": "sparse",
                "f1": 0.9,
                "precision": 0.9,
                "recall": 0.9,
                "far": 0.1,
                "miss": 0.1,
                "tp": 450,
                "fp": 50,
                "tn": 450,
                "fn": 50,
            },
        ]
        _make_report(root / "eval_20260101T000000Z.json", rows, dataset="ds")
        far = {r["engine"]: r for r in far_table(load_records(root))}
        assert abs(far["dense"]["far"] - 0.5) < 1e-12, far["dense"]
        assert abs(far["sparse"]["far"] - 0.1) < 1e-12, far["sparse"]
        # 5x difference in FAR, identical FP/N -- which is why both are reported
        assert far["dense"]["n_frames"] == far["sparse"]["n_frames"], (
            far["dense"]["n_frames"],
            far["sparse"]["n_frames"],
        )
        assert abs(far["dense"]["fp_per_frame"] - far["sparse"]["fp_per_frame"]) < 1e-12
        assert abs(far["dense"]["fp_per_frame"] - 0.05) < 1e-12
        assert abs(far["dense"]["speech_ratio"] - 0.9) < 1e-12
        assert abs(far["sparse"]["speech_ratio"] - 0.5) < 1e-12


def test_worst_file_survives_aggregation() -> None:
    """The per-file table must expose a collapse that the mean hides."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = [
            {
                "model": "a",
                "uri": f"ds_file_{i:03d}",
                "f1": 0.99,
                "precision": 0.99,
                "recall": 0.99,
                "far": 0.01,
                "miss": 0.01,
                "tp": 990,
                "fp": 10,
                "tn": 990,
                "fn": 10,
            }
            for i in range(9)
        ]
        # One file where nearly all speech is missed: F1 ~ 0.07
        rows.append(
            {
                "model": "a",
                "uri": "ds_file_009",
                "f1": 0.07,
                "precision": 0.29,
                "recall": 0.04,
                "far": 0.011,
                "miss": 0.96,
                "tp": 40,
                "fp": 100,
                "tn": 8900,
                "fn": 960,
            }
        )
        _make_report(root / "eval_20260101T000000Z.json", rows, dataset="ds")
        records = load_records(root)
        far = far_table(records)[0]
        pf = per_file_table(records)[0]
        assert far["f1"] > 0.93, far["f1"]
        assert pf["n_files"] == 10, pf
        assert abs(pf["f1_min"] - 0.0702) < 1e-3, pf
        assert pf["worst_uri"] == "ds_file_009", pf
        assert pf["n_f1_below"] == 1, pf
        assert pf["f1_median"] == 0.99, pf
        assert pf["f1_min"] < far["f1"] - 0.8, (pf["f1_min"], far["f1"])


def test_best_per_dataset_flags_winner() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        records = []
        for name, f1 in (("fast", 0.8), ("good", 0.95)):
            root = Path(tmp) / name
            root.mkdir()
            _make_report(
                root / "eval_20260101T000000Z.json",
                [
                    {
                        "model": name,
                        "f1": f1,
                        "precision": f1,
                        "recall": f1,
                        "far": 0.1,
                        "miss": 1 - f1,
                        "tp": 100,
                        "fp": 10,
                        "tn": 90,
                        "fn": 0,
                    }
                ],
                dataset="ds",
            )
            records.extend(load_records(root))
        headers, cross = cross_dataset_f1(records)
        assert headers == ["ds"], headers
        winners = best_per_dataset(headers, cross)
        assert winners == {"ds": "good"}, winners


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as e:
            failures += 1
            print(f"FAIL {fn.__name__}: {e}")
        else:
            print(f"ok   {fn.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
