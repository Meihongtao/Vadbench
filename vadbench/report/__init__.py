"""Post-hoc analysis of ``results/eval_*.json``.

``vadbench.eval`` produces per-file metrics; this package re-reads those raw
JSON reports to answer the questions a single weighted number cannot:

* how much do files disagree inside one dataset (median / worst-case / spread);
* how large is FAR really, once the non-speech denominator is made explicit;
* what does the same run look like under a different aggregation.

Nothing here re-runs a model: it is pure post-processing of existing results.
"""

from vadbench.report.collect import FileRecord, load_records, load_summary
from vadbench.report.tables import (
    cross_dataset_f1,
    far_table,
    per_file_table,
)

__all__ = [
    "FileRecord",
    "load_records",
    "load_summary",
    "cross_dataset_f1",
    "far_table",
    "per_file_table",
]
