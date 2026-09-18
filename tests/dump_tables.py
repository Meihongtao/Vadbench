from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vadbench.report.collect import RESULTS, load_records  # noqa: E402
from vadbench.report.tables import (  # noqa: E402
    DATASET_LABEL,
    far_table,
    per_file_table,
)


def format_markdown(rows: list[dict], columns: list[tuple[str, str]], best: str = "f1") -> str:
    """Render rows as a GitHub table; bold the best value in ``best`` column."""
    out = ["| " + " | ".join(h for h, _ in columns) + " |"]
    out.append("|" + "|".join("---" for _ in columns) + "|")
    top = None
    if best:
        values = [r[best] for r in rows if r.get(best) is not None]
        top = max(values) if values else None
    for r in rows:
        cells = []
        for _, key in columns:
            v = r.get(key)
            if v is None:
                cells.append("—")
            elif isinstance(v, float):
                text = f"{v:.3f}"
                if key == best and top is not None and v == top:
                    text = f"**{text}**"
                cells.append(text)
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> int:
    records = load_records()
    hours = sum(r.duration_s for r in records) / 3600.0
    print(f"# {len(records)} per-file records, {hours:.1f} h, {len({r.dataset for r in records})} datasets")
    print(f"# results dir: {RESULTS}")
    print()

    far = far_table(records)
    for dataset in [d for d in DATASET_LABEL if any(r["dataset"] == d for r in far)]:
        rows = [r for r in far if r["dataset"] == dataset]
        print(f"### {DATASET_LABEL[dataset]}  ({rows[0]['hours']:.2f} h)")
        print(
            format_markdown(
                rows,
                [
                    ("engine", "engine"),
                    ("F1", "f1"),
                    ("P", "precision"),
                    ("R", "recall"),
                    ("FAR", "far"),
                    ("FP/N", "fp_per_frame"),
                ],
                best="f1",
            )
        )
        print()

    print("### per-file spread")
    print(
        format_markdown(
            per_file_table(records),
            [
                ("dataset", "dataset"),
                ("engine", "engine"),
                ("files", "n_files"),
                ("min", "f1_min"),
                ("p25", "f1_p25"),
                ("median", "f1_median"),
                ("mean", "f1_mean"),
                ("max", "f1_max"),
                ("<0.9", "n_f1_below"),
            ],
            best="",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
