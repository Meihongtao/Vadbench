from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FrameMetrics:
    precision: float
    recall: float
    f1: float
    far: float
    miss: float
    tp: int
    fp: int
    tn: int
    fn: int
    n_frames: int
    duration_s: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_frame_metrics(
    ref: np.ndarray,
    hyp: np.ndarray,
    frame_ms: float = 10.0,
) -> FrameMetrics:
    """Compute Precision / Recall / F1 / FAR / Miss on binary frames."""
    if ref.shape != hyp.shape:
        raise ValueError(f"shape mismatch: ref={ref.shape} hyp={hyp.shape}")
    r = ref.astype(bool)
    h = hyp.astype(bool)
    tp = int(np.logical_and(r, h).sum())
    fp = int(np.logical_and(~r, h).sum())
    tn = int(np.logical_and(~r, ~h).sum())
    fn = int(np.logical_and(r, ~h).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    far = fp / (fp + tn) if (fp + tn) else 0.0
    miss = fn / (fn + tp) if (fn + tp) else 0.0
    duration_s = len(ref) * frame_ms / 1000.0
    return FrameMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        far=far,
        miss=miss,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        n_frames=len(ref),
        duration_s=duration_s,
    )


def aggregate_metrics(
    items: list[FrameMetrics],
    weighted: bool = True,
) -> FrameMetrics:
    """Aggregate per-file metrics from summed confusion counts when weighted."""
    if not items:
        return FrameMetrics(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            far=0.0,
            miss=0.0,
            tp=0,
            fp=0,
            tn=0,
            fn=0,
            n_frames=0,
            duration_s=0.0,
        )
    if weighted:
        tp = sum(m.tp for m in items)
        fp = sum(m.fp for m in items)
        tn = sum(m.tn for m in items)
        fn = sum(m.fn for m in items)
        n_frames = sum(m.n_frames for m in items)
        duration_s = sum(m.duration_s for m in items)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        far = fp / (fp + tn) if (fp + tn) else 0.0
        miss = fn / (fn + tp) if (fn + tp) else 0.0
        return FrameMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            far=far,
            miss=miss,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            n_frames=n_frames,
            duration_s=duration_s,
        )

    n = len(items)
    return FrameMetrics(
        precision=sum(m.precision for m in items) / n,
        recall=sum(m.recall for m in items) / n,
        f1=sum(m.f1 for m in items) / n,
        far=sum(m.far for m in items) / n,
        miss=sum(m.miss for m in items) / n,
        tp=sum(m.tp for m in items),
        fp=sum(m.fp for m in items),
        tn=sum(m.tn for m in items),
        fn=sum(m.fn for m in items),
        n_frames=sum(m.n_frames for m in items),
        duration_s=sum(m.duration_s for m in items),
    )
