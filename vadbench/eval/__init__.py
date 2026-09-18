"""VAD evaluation package."""

from vadbench.eval.metrics import FrameMetrics, aggregate_metrics, compute_frame_metrics
from vadbench.eval.rasterize import segments_to_frames, union_segments

__all__ = [
    "FrameMetrics",
    "aggregate_metrics",
    "compute_frame_metrics",
    "segments_to_frames",
    "union_segments",
]
