from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from vadbench.eval.rasterize import union_segments
from vadbench.types import SpeechSegment

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"

# name -> (subdir under data/, language)
DATASET_REGISTRY: dict[str, tuple[str, str]] = {
    "aishell4": ("aishell4_test", "zh"),
    "alimeeting": ("alimeeting_far_test", "zh"),
    "alimeeting_eval": ("alimeeting_far_eval", "zh"),
    "magicdata_ramc": ("magicdata_ramc_test", "zh"),
    "magicdata_meeting": ("magicdata_meeting", "zh"),
    "ami": ("ami_ihm_test", "en"),
    # Download abandoned (Oxford host too slow); kept for optional retry.
    "voxconverse": ("voxconverse_test", "en"),
}

# Prepared local sets used by zh|en|all (excludes unavailable voxconverse).
# magicdata_* added after download completes.
READY_DATASETS: tuple[str, ...] = (
    "aishell4",
    "alimeeting",
    "alimeeting_eval",
    "magicdata_ramc",
    "ami",
)

LANG_DATASETS: dict[str, tuple[str, ...]] = {
    "zh": ("aishell4", "alimeeting", "alimeeting_eval", "magicdata_ramc"),
    "en": ("ami",),
}


@dataclass(frozen=True)
class EvalSample:
    uri: str
    audio_path: Path
    lang: str
    reference: list[SpeechSegment]
    duration_s: float | None = None
    dataset: str = ""


def parse_rttm(path: Path) -> list[SpeechSegment]:
    """Parse RTTM SPEAKER lines into speech segments (any speaker)."""
    segs: list[SpeechSegment] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        if parts[0].upper() != "SPEAKER":
            continue
        try:
            start = float(parts[3])
            dur = float(parts[4])
        except ValueError:
            continue
        if dur <= 0:
            continue
        segs.append(SpeechSegment(start=start, end=start + dur))
    return union_segments(segs)


def _audio_duration_s(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.duration)


def load_manifest(manifest_path: Path, dataset: str = "") -> list[EvalSample]:
    samples: list[EvalSample] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        audio = Path(row["audio_path"])
        if not audio.is_file():
            audio = ROOT / row["audio_path"]
        if "rttm_path" in row:
            rttm = Path(row["rttm_path"])
            if not rttm.is_file():
                rttm = ROOT / row["rttm_path"]
            ref_segs = parse_rttm(rttm)
        elif "segments" in row:
            ref_segs = union_segments(
                [
                    SpeechSegment(start=float(a), end=float(b))
                    for a, b in row["segments"]
                ]
            )
        else:
            raise ValueError(f"manifest row missing rttm_path/segments: {row.get('uri')}")
        dur = row.get("duration_s")
        if dur is None and audio.is_file():
            dur = _audio_duration_s(audio)
        ds_name = str(row.get("dataset") or dataset or "")
        samples.append(
            EvalSample(
                uri=str(row["uri"]),
                audio_path=audio,
                lang=str(row["lang"]),
                reference=ref_segs,
                duration_s=float(dur) if dur is not None else None,
                dataset=ds_name,
            )
        )
    return samples


def resolve_dataset_names(spec: str) -> list[str]:
    """Expand zh|en|all|comma-list into concrete dataset registry keys."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty dataset spec")
    out: list[str] = []
    for p in parts:
        if p == "all":
            out.extend(READY_DATASETS)
        elif p in LANG_DATASETS:
            out.extend(LANG_DATASETS[p])
        elif p in DATASET_REGISTRY:
            out.append(p)
        else:
            raise ValueError(
                f"Unknown dataset={p!r}; choose from "
                f"{list(DATASET_REGISTRY)} or zh|en|all"
            )
    # preserve order, unique
    seen: set[str] = set()
    uniq: list[str] = []
    for name in out:
        if name not in seen:
            seen.add(name)
            uniq.append(name)
    return uniq


def load_dataset(
    dataset: str = "all",
    data_root: Path | None = None,
) -> list[EvalSample]:
    """Load prepared eval set(s).

    ``dataset`` accepts registry keys (aishell4, alimeeting, ami, voxconverse),
    language aliases (zh, en), comma lists, or ``all``.
    """
    root = data_root or DATA_ROOT
    names = resolve_dataset_names(dataset)
    out: list[EvalSample] = []
    for name in names:
        subdir, _lang = DATASET_REGISTRY[name]
        path = root / subdir / "manifest.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run: python -m vadbench.eval.download_data "
                f"--dataset {name}"
            )
        out.extend(load_manifest(path, dataset=name))
    return out
