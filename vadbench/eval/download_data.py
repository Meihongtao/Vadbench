from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HF_MIRROR = "https://hf-mirror.com"

# Official AliMeeting audio (ModelScope card only hosts metadata; audio is on Aliyun OSS).
ALIMEETING_TEST_URL = (
    "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/"
    "AliMeeting/openlr/Test_Ali.tar.gz"
)
ALIMEETING_EVAL_URL = (
    "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/"
    "AliMeeting/openlr/Eval_Ali.tar.gz"
)
# Official VoxConverse test wavs (Oxford; often slow from CN — prefer AliMeeting Eval).
VOXCONVERSE_TEST_WAV_URL = (
    "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip"
)
# MagicData-RAMC (OpenSLR SLR123); full archive ~15GB, we materialize test only.
MAGICDATA_RAMC_URLS = (
    "https://www.openslr.org/resources/123/MagicData-RAMC.tar.gz",
    "https://openslr.elda.org/resources/123/MagicData-RAMC.tar.gz",
)
MAGICDATA_RAMC_BYTES = 15_470_673_657
MAGICDATA_RAMC_INVALID_SPK = "G00000000"
# MagicData web-meeting corpus (~202MB on Magichub; needs login → local zip import).
MAGICDATA_MEETING_MAGICHUB = (
    "https://magichub.com/datasets/mandarin-chinese-conversational-speech-corpus-web-meeting/"
)
# RTTM shipped with local pyannote ModelScope download (preferred over HuggingFace).
PYANNOTE_REPRO = (
    ROOT
    / "models"
    / "pyannote"
    / "speaker-diarization-3.1"
    / "reproducible_research"
)


def _ensure_hf_mirror() -> None:
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = HF_MIRROR


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _download_file(
    url: str,
    dest: Path,
    label: str,
    *,
    min_bytes: int | None = None,
    expected_bytes: int | None = None,
) -> Path:
    """Resumable HTTP download to dest.

    If ``expected_bytes`` is set, raise when the final size is short.
    If ``min_bytes`` is set and an existing file is smaller, resume/redownload.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.is_file() else 0
    if min_bytes is not None and existing > 0 and existing < min_bytes:
        print(f"[{label}] incomplete local file ({existing} B), will resume")
    headers = {"User-Agent": "vadbench/1.0"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        print(f"[{label}] resume {dest.name} from byte {existing}")
    else:
        print(f"[{label}] downloading {url}")

    req = Request(url, headers=headers)
    with urlopen(req, timeout=300) as resp:
        mode = "ab" if existing > 0 and getattr(resp, "status", 200) == 206 else "wb"
        if mode == "wb" and existing > 0:
            existing = 0
        total = resp.headers.get("Content-Length")
        total_i = int(total) + existing if total else expected_bytes
        written = existing
        chunk = 1024 * 1024
        with dest.open(mode) as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                written += len(buf)
                if total_i:
                    pct = 100.0 * written / total_i
                    print(
                        f"\r[{label}] {written / 1e9:.2f}/{total_i / 1e9:.2f} GB ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
        print()
    final = dest.stat().st_size
    need = expected_bytes or min_bytes
    if need is not None and final < need:
        raise IOError(
            f"[{label}] download incomplete: got {final} B, need >= {need} B "
            f"(resume by re-running the same command)"
        )
    return dest


def _split_rttm(combined: Path, out_dir: Path) -> dict[str, Path]:
    """Split a multi-URI RTTM into per-uri files. Returns uri -> path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[str]] = defaultdict(list)
    for line in combined.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "SPEAKER":
            continue
        buckets[parts[1]].append(line)
    out: dict[str, Path] = {}
    for uri, lines in buckets.items():
        path = out_dir / f"{uri}.rttm"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out[uri] = path
    return out


def _write_mono_wav(src: Path, dst: Path, channel: int = 0) -> None:
    """Write 16 kHz mono float wav from multi-channel source.

    ``channel=-1`` averages all channels (mixdown); otherwise pick channel index.
    """
    import numpy as np
    import soundfile as sf

    arr, sr = sf.read(str(src), always_2d=True)
    if channel < 0:
        mono = arr.mean(axis=1).astype(np.float32)
    else:
        ch = min(channel, arr.shape[1] - 1)
        mono = arr[:, ch].astype(np.float32, copy=False)
    if sr != 16000:
        duration = len(mono) / float(sr)
        n = max(1, int(round(duration * 16000)))
        x_old = np.linspace(0.0, 1.0, num=len(mono), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
        mono = np.interp(x_new, x_old, mono).astype(np.float32)
        sr = 16000
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), mono, sr)


def _materialize_audio_rttm_hf(
    *,
    repo_id: str,
    dest: Path,
    lang: str,
    audio_glob_roots: list[str],
    label: str,
) -> Path:
    """Legacy HF snapshot helper (kept for aishell4 until a ModelScope mirror is wired)."""
    from huggingface_hub import snapshot_download

    _ensure_hf_mirror()
    dest.mkdir(parents=True, exist_ok=True)
    raw = dest / "_hf_raw"
    print(f"[{label}] downloading {repo_id} via HF ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(raw),
    )

    audio_dst = dest / "audio"
    rttm_dst = dest / "rttm"
    audio_dst.mkdir(parents=True, exist_ok=True)
    rttm_dst.mkdir(parents=True, exist_ok=True)

    audio_files: list[Path] = []
    for root_name in audio_glob_roots:
        base = raw / root_name
        if not base.is_dir():
            continue
        audio_files.extend(base.rglob("*.flac"))
        audio_files.extend(base.rglob("*.wav"))

    rttm_files = {p.stem: p for p in raw.rglob("*.rttm")}
    rows = []
    for audio in sorted(set(audio_files)):
        uri = audio.stem
        rttm = rttm_files.get(uri)
        if rttm is None:
            candidates = [
                p for s, p in rttm_files.items() if s.startswith(uri) or uri.startswith(s)
            ]
            rttm = candidates[0] if candidates else None
        if rttm is None:
            print(f"[warn][{label}] no rttm for {audio.name}, skip")
            continue
        a_out = audio_dst / audio.name
        r_out = rttm_dst / f"{uri}.rttm"
        if not a_out.exists():
            shutil.copy2(audio, a_out)
        if not r_out.exists():
            shutil.copy2(rttm, r_out)
        rows.append(
            {
                "uri": uri,
                "lang": lang,
                "dataset": dest.name,
                "audio_path": _rel(a_out),
                "rttm_path": _rel(r_out),
            }
        )

    manifest = dest / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[{label}] wrote {len(rows)} items -> {manifest}")
    return manifest


def download_aishell4_test(dest: Path) -> Path:
    """Download AISHELL-4 test (audio+RTTM). Currently via HF mirror of OpenSLR layout."""
    return _materialize_audio_rttm_hf(
        repo_id="ggfox00000/dia-aishell4-test",
        dest=dest,
        lang="zh",
        audio_glob_roots=["audio"],
        label="aishell4",
    )


def _textgrid_speech_segments(path: Path) -> list[tuple[float, float]]:
    """Parse Praat TextGrid intervals with non-empty text → (start, end) seconds."""
    import re

    text = path.read_text(encoding="utf-8", errors="ignore")
    segs: list[tuple[float, float]] = []
    # Match interval blocks: xmin / xmax / text =
    pattern = re.compile(
        r"xmin\s*=\s*([0-9.]+)\s*"
        r"xmax\s*=\s*([0-9.]+)\s*"
        r'text\s*=\s*"([^"]*)"',
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        start, end, mark = float(m.group(1)), float(m.group(2)), m.group(3).strip()
        if mark and end > start:
            segs.append((start, end))
    return segs


def _write_rttm_from_segments(uri: str, segments: list[tuple[float, float]], path: Path) -> None:
    lines = []
    for start, end in segments:
        dur = end - start
        if dur <= 0:
            continue
        lines.append(
            f"SPEAKER {uri} 1 {start:.3f} {dur:.3f} <NA> <NA> speech <NA> <NA>"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _download_alimeeting_split(
    *,
    dest: Path,
    tar_url: str,
    tar_name: str,
    far_token: str,
    label: str,
    rttm_combined: Path | None,
    min_tar_bytes: int,
) -> Path:
    """Shared AliMeeting far-field materializer (test uses pyannote RTTM; eval uses TextGrid)."""
    dest.mkdir(parents=True, exist_ok=True)
    raw = dest / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    tar_path = raw / tar_name

    try:
        from modelscope.hub.file_download import dataset_file_download

        meta = dest / "_modelscope_meta"
        meta.mkdir(parents=True, exist_ok=True)
        dataset_file_download(
            dataset_id="modelscope/AliMeeting",
            file_path="README.md",
            local_dir=str(meta),
        )
        print(f"[{label}] ModelScope metadata OK (modelscope/AliMeeting)")
    except Exception as e:
        print(f"[warn][{label}] ModelScope metadata skip: {e}")

    if not tar_path.is_file() or tar_path.stat().st_size < min_tar_bytes:
        _download_file(tar_url, tar_path, label)

    extract_dir = raw / "extracted"
    need_extract = not any(extract_dir.rglob("*.wav"))
    if need_extract:
        print(f"[{label}] extracting far-field wavs/TextGrid from tar...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            members = [
                m
                for m in tar.getmembers()
                if (far_token in m.name)
                and (m.name.endswith(".wav") or m.name.lower().endswith(".textgrid"))
            ]
            print(f"[{label}] extracting {len(members)} far members...")
            tar.extractall(path=extract_dir, members=members)

    audio_dirs = list(extract_dir.rglob("audio_dir"))
    if audio_dirs:
        far_dirs = [d for d in audio_dirs if "far" in str(d).lower()]
        search_root = far_dirs[0] if far_dirs else audio_dirs[0]
        wavs = list(search_root.rglob("*.wav"))
    else:
        wavs = list(extract_dir.rglob("*.wav"))

    textgrids = {p.stem: p for p in extract_dir.rglob("*.TextGrid")}
    textgrids.update({p.stem: p for p in extract_dir.rglob("*.textgrid")})

    rttm_map: dict[str, Path] = {}
    if rttm_combined is not None and rttm_combined.is_file():
        rttm_map = _split_rttm(rttm_combined, dest / "rttm")

    audio_dst = dest / "audio"
    rttm_dst = dest / "rttm"
    audio_dst.mkdir(parents=True, exist_ok=True)
    rttm_dst.mkdir(parents=True, exist_ok=True)

    rows = []
    for wav in sorted(wavs):
        uri = wav.stem
        rttm_path = rttm_map.get(uri)
        if rttm_path is None:
            # TextGrid stem is often shorter session id for far (without _MSxxx)
            tg = textgrids.get(uri)
            if tg is None:
                # match by prefix
                for stem, p in textgrids.items():
                    if uri.startswith(stem) or stem.startswith(uri.split("_MS")[0]):
                        tg = p
                        break
            if tg is None:
                print(f"[warn][{label}] no rttm/TextGrid for {uri}, skip")
                continue
            segs = _textgrid_speech_segments(tg)
            if not segs:
                print(f"[warn][{label}] empty TextGrid {tg.name}, skip")
                continue
            rttm_path = rttm_dst / f"{uri}.rttm"
            _write_rttm_from_segments(uri, segs, rttm_path)

        out_wav = audio_dst / f"{uri}.wav"
        if not out_wav.exists():
            _write_mono_wav(wav, out_wav, channel=0)
        rows.append(
            {
                "uri": uri,
                "lang": "zh",
                "dataset": dest.name,
                "audio_path": _rel(out_wav),
                "rttm_path": _rel(rttm_path),
                "source": "aliyun_oss+modelscope_meta",
            }
        )

    manifest = dest / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[{label}] wrote {len(rows)} items -> {manifest}")
    return manifest


def download_alimeeting_far_test(dest: Path) -> Path:
    """AliMeeting test far-field ch0 + pyannote RTTM (Aliyun OSS)."""
    rttm_src = PYANNOTE_REPRO / "AliMeeting.SpeakerDiarization.Benchmark.test.rttm"
    if not rttm_src.is_file():
        raise FileNotFoundError(
            f"Missing {rttm_src}. Run: python -m vadbench.download --engine pyannote"
        )
    return _download_alimeeting_split(
        dest=dest,
        tar_url=ALIMEETING_TEST_URL,
        tar_name="Test_Ali.tar.gz",
        far_token="Test_Ali_far",
        label="alimeeting",
        rttm_combined=rttm_src,
        min_tar_bytes=1_000_000_000,
    )


def download_alimeeting_far_eval(dest: Path) -> Path:
    """AliMeeting eval far-field ch0 + TextGrid-derived RTTM (Aliyun OSS, ~3.4GB)."""
    return _download_alimeeting_split(
        dest=dest,
        tar_url=ALIMEETING_EVAL_URL,
        tar_name="Eval_Ali.tar.gz",
        far_token="Eval_Ali_far",
        label="alimeeting_eval",
        rttm_combined=None,
        min_tar_bytes=500_000_000,
    )


def download_voxconverse_test(dest: Path) -> Path:
    """VoxConverse test wav (official Oxford) + local pyannote RTTM. No HuggingFace."""
    dest.mkdir(parents=True, exist_ok=True)
    raw = dest / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / "voxconverse_test_wav.zip"

    # Official zip is ~4.27 GB; reject tiny/HTML error bodies.
    min_zip_bytes = 3_500_000_000
    if not zip_path.is_file() or zip_path.stat().st_size < min_zip_bytes:
        _download_file(VOXCONVERSE_TEST_WAV_URL, zip_path, "voxconverse")
    if not zip_path.is_file() or zip_path.stat().st_size < min_zip_bytes:
        raise RuntimeError(
            f"VoxConverse zip incomplete: {zip_path} "
            f"size={zip_path.stat().st_size if zip_path.is_file() else 0}"
        )
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f"corrupt member in zip: {bad}")
    except zipfile.BadZipFile as e:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"VoxConverse zip invalid, deleted for re-download: {e}"
        ) from e

    extract_dir = raw / "wav"
    if not extract_dir.is_dir() or not list(extract_dir.rglob("*.wav")):
        print("[voxconverse] extracting zip...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    rttm_src = PYANNOTE_REPRO / "VoxConverse.SpeakerDiarization.Benchmark.test.rttm"
    if not rttm_src.is_file():
        raise FileNotFoundError(
            f"Missing {rttm_src}. Run: python -m vadbench.download --engine pyannote"
        )
    rttm_map = _split_rttm(rttm_src, dest / "rttm")

    audio_dst = dest / "audio"
    audio_dst.mkdir(parents=True, exist_ok=True)
    wavs = {p.stem: p for p in extract_dir.rglob("*.wav")}
    rows = []
    for uri, rttm_path in sorted(rttm_map.items()):
        src = wavs.get(uri)
        if src is None:
            print(f"[warn][voxconverse] no wav for {uri}, skip")
            continue
        out_wav = audio_dst / f"{uri}.wav"
        if not out_wav.exists():
            _write_mono_wav(src, out_wav, channel=0)
        rows.append(
            {
                "uri": uri,
                "lang": "en",
                "dataset": dest.name,
                "audio_path": _rel(out_wav),
                "rttm_path": _rel(rttm_path),
                "source": "oxford_official+pyannote_rttm",
            }
        )

    manifest = dest / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[voxconverse] wrote {len(rows)} items -> {manifest}")
    return manifest


def download_ami_ihm_test(dest: Path) -> Path:
    """Download AMI IHM test split and write wav + segment manifest."""
    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    _ensure_hf_mirror()
    dest.mkdir(parents=True, exist_ok=True)
    audio_dst = dest / "audio"
    audio_dst.mkdir(parents=True, exist_ok=True)

    print("[ami] loading diarizers-community/ami ihm test ...")
    ds = load_dataset("diarizers-community/ami", "ihm", split="test")
    ds = ds.cast_column("audio", Audio(decode=False))

    rows = []
    for i, ex in enumerate(ds):
        uri = f"ami_ihm_test_{i:03d}"
        wav_path = audio_dst / f"{uri}.wav"
        audio = ex["audio"]

        if not wav_path.exists():
            arr, sr = _decode_audio_dict(audio)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if sr != 16000:
                duration = len(arr) / float(sr)
                n = max(1, int(round(duration * 16000)))
                x_old = np.linspace(0.0, 1.0, num=len(arr), endpoint=False)
                x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
                arr = np.interp(x_new, x_old, arr).astype(np.float32)
                sr = 16000
            sf.write(str(wav_path), arr.astype(np.float32), sr)
        else:
            info = sf.info(str(wav_path))
            sr = int(info.samplerate)
            arr = None

        starts = ex.get("timestamps_start") or []
        ends = ex.get("timestamps_end") or []
        segments = []
        for s, e in zip(starts, ends):
            s_f, e_f = float(s), float(e)
            if e_f > s_f:
                segments.append([s_f, e_f])

        if arr is not None:
            duration_s = float(len(arr)) / float(sr)
        else:
            duration_s = float(sf.info(str(wav_path)).duration)

        rows.append(
            {
                "uri": uri,
                "lang": "en",
                "dataset": dest.name,
                "audio_path": _rel(wav_path),
                "segments": segments,
                "duration_s": duration_s,
            }
        )

    manifest = dest / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[ami] wrote {len(rows)} items -> {manifest}")
    return manifest


def _decode_audio_dict(audio: dict) -> tuple:
    """Decode datasets Audio(decode=False) payload with soundfile."""
    import io

    import numpy as np
    import soundfile as sf

    path = audio.get("path")
    raw = audio.get("bytes")
    if path and Path(path).is_file():
        arr, sr = sf.read(path, always_2d=False)
        return np.asarray(arr, dtype=np.float32), int(sr)
    if raw:
        arr, sr = sf.read(io.BytesIO(raw), always_2d=False)
        return np.asarray(arr, dtype=np.float32), int(sr)
    raise RuntimeError(f"Cannot decode audio dict keys={list(audio.keys())}")


def _parse_magicdata_txt(path: Path) -> list[tuple[float, float, str]]:
    """Parse MagicData TXT: ``[start,end]\\tspk\\tmeta\\ttext`` -> (start, end, spk)."""
    import re

    line_re = re.compile(r"^\[([\d.]+),([\d.]+)\]\s+(\S+)\s+")
    segs: list[tuple[float, float, str]] = []
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = line_re.match(line)
        if not m:
            continue
        start, end, spk = float(m.group(1)), float(m.group(2)), m.group(3)
        if spk == MAGICDATA_RAMC_INVALID_SPK:
            continue
        if end <= start:
            continue
        segs.append((start, end, spk))
    return segs


def _parse_magicdata_split_tsv(tsv_path: Path) -> list[str]:
    """Return reco_ids (no .wav) listed in DataPartition/{split}.tsv."""
    names: list[str] = []
    for line in tsv_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        first = line.split("\t")[0].replace("\\", "/")
        base = Path(first).name
        if not base.lower().endswith(".wav"):
            continue
        names.append(base[: -len(".wav")])
    return names


def _find_tar_member(members: list, *suffixes: str) -> object | None:
    for m in members:
        name = getattr(m, "name", "").replace("\\", "/")
        for suf in suffixes:
            if name.endswith(suf) or name.endswith(suf.replace("/", "\\")):
                return m
    return None


def download_magicdata_ramc_test(dest: Path) -> Path:
    """Materialize MagicData-RAMC **test** split for VAD (OpenSLR SLR123).

    Downloads the full ~15GB archive (OpenSLR ships one tarball), then extracts
    only test wav/txt members and writes mono 16 kHz audio + RTTM.
    """
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dest / "manifest.jsonl"
    audio_dst = dest / "audio"
    rttm_dst = dest / "rttm"
    if manifest.is_file() and any(audio_dst.glob("*.wav")):
        n = sum(1 for _ in manifest.open(encoding="utf-8") if _.strip())
        if n > 0:
            print(f"[magicdata_ramc] skip existing manifest ({n} items)")
            return manifest

    raw = dest / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    tar_path = raw / "MagicData-RAMC.tar.gz"
    if not tar_path.is_file() or tar_path.stat().st_size < MAGICDATA_RAMC_BYTES * 0.99:
        last_err: Exception | None = None
        for url in MAGICDATA_RAMC_URLS:
            try:
                _download_file(
                    url,
                    tar_path,
                    "magicdata_ramc",
                    min_bytes=MAGICDATA_RAMC_BYTES,
                    expected_bytes=MAGICDATA_RAMC_BYTES,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"[warn][magicdata_ramc] download failed from {url}: {e}")
        if last_err is not None:
            raise last_err

    extract_root = raw / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    print("[magicdata_ramc] scanning tar for test partition ...", flush=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        tsv_m = _find_tar_member(members, "DataPartition/test.tsv", "DataPartition\\test.tsv")
        if tsv_m is None:
            raise RuntimeError("DataPartition/test.tsv not found in MagicData-RAMC.tar.gz")
        tar.extract(tsv_m, path=extract_root)
        tsv_path = extract_root / tsv_m.name
        if not tsv_path.is_file():
            # tar may strip leading components differently
            found = list(extract_root.rglob("test.tsv"))
            if not found:
                raise FileNotFoundError("extracted test.tsv missing")
            tsv_path = found[0]
        reco_ids = set(_parse_magicdata_split_tsv(tsv_path))
        print(f"[magicdata_ramc] test sessions: {len(reco_ids)}", flush=True)

        by_name = {m.name.replace("\\", "/"): m for m in members}
        want: list = []
        for reco_id in reco_ids:
            for kind, ext in (("WAV", ".wav"), ("TXT", ".txt")):
                suffix = f"MDT2021S003/{kind}/{reco_id}{ext}"
                hit = None
                for name, m in by_name.items():
                    if name.endswith(suffix) or name.endswith(f"{kind}/{reco_id}{ext}"):
                        hit = m
                        break
                if hit is None:
                    print(f"[warn][magicdata_ramc] tar missing {kind}/{reco_id}{ext}")
                    continue
                want.append(hit)
        print(f"[magicdata_ramc] extracting {len(want)} wav/txt members ...", flush=True)
        tar.extractall(path=extract_root, members=want)

    wav_dir = next(extract_root.rglob("WAV"), None)
    txt_dir = next(extract_root.rglob("TXT"), None)
    if wav_dir is None or txt_dir is None:
        raise FileNotFoundError("MDT2021S003/WAV or TXT not found after extract")

    audio_dst.mkdir(parents=True, exist_ok=True)
    rttm_dst.mkdir(parents=True, exist_ok=True)
    rows = []
    for reco_id in sorted(reco_ids):
        wav_src = wav_dir / f"{reco_id}.wav"
        txt_src = txt_dir / f"{reco_id}.txt"
        if not wav_src.is_file() or not txt_src.is_file():
            print(f"[warn][magicdata_ramc] missing {reco_id}, skip")
            continue
        out_wav = audio_dst / f"{reco_id}.wav"
        if not out_wav.is_file():
            _write_mono_wav(wav_src, out_wav, channel=-1)
        segs = [(a, b) for a, b, _spk in _parse_magicdata_txt(txt_src)]
        if not segs:
            print(f"[warn][magicdata_ramc] empty labels {reco_id}, skip")
            continue
        rttm_path = rttm_dst / f"{reco_id}.rttm"
        _write_rttm_from_segments(reco_id, segs, rttm_path)
        rows.append(
            {
                "uri": reco_id,
                "lang": "zh",
                "dataset": "magicdata_ramc_test",
                "audio_path": _rel(out_wav),
                "rttm_path": _rel(rttm_path),
                "source": "openslr_123",
            }
        )

    with manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[magicdata_ramc] wrote {len(rows)} items -> {manifest}")
    # Drop bulky extracted originals after materialization (keep tar for resume).
    shutil.rmtree(extract_root, ignore_errors=True)
    return manifest


def _materialize_magicdata_meeting_from_dir(src_root: Path, dest: Path) -> Path:
    """Build mono 16 kHz + RTTM from an unpacked MagicData meeting corpus tree."""
    audio_dst = dest / "audio"
    rttm_dst = dest / "rttm"
    audio_dst.mkdir(parents=True, exist_ok=True)
    rttm_dst.mkdir(parents=True, exist_ok=True)

    wavs = sorted(src_root.rglob("*.wav")) + sorted(src_root.rglob("*.WAV"))
    if not wavs:
        raise FileNotFoundError(f"No wav under {src_root}")

    rows = []
    for wav in wavs:
        uri = wav.stem
        txt = wav.with_suffix(".txt")
        if not txt.is_file():
            # sibling TXT dir
            cand = list(src_root.rglob(f"{uri}.txt"))
            txt = cand[0] if cand else txt
        if not txt.is_file():
            print(f"[warn][magicdata_meeting] no txt for {uri}, skip")
            continue
        out_wav = audio_dst / f"{uri}.wav"
        if not out_wav.is_file():
            _write_mono_wav(wav, out_wav, channel=-1)
        segs = [(a, b) for a, b, _spk in _parse_magicdata_txt(txt)]
        if not segs:
            # some meeting packs use simple start end without MagicData tags
            print(f"[warn][magicdata_meeting] empty/unparsed labels {uri}, skip")
            continue
        rttm_path = rttm_dst / f"{uri}.rttm"
        _write_rttm_from_segments(uri, segs, rttm_path)
        rows.append(
            {
                "uri": uri,
                "lang": "zh",
                "dataset": "magicdata_meeting",
                "audio_path": _rel(out_wav),
                "rttm_path": _rel(rttm_path),
                "source": "magichub_meeting",
            }
        )

    manifest = dest / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[magicdata_meeting] wrote {len(rows)} items -> {manifest}")
    return manifest


def download_magicdata_meeting(dest: Path) -> Path:
    """Materialize MagicData web-meeting corpus (8 kHz → 16 kHz mono).

    Magichub requires sign-in for the ~202MB zip. Place the downloaded archive at
    ``data/magicdata_meeting/_raw/*.zip`` (or already extracted under ``_raw/extracted``),
    then re-run this downloader.
    """
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dest / "manifest.jsonl"
    if manifest.is_file() and any((dest / "audio").glob("*.wav")):
        n = sum(1 for _ in manifest.open(encoding="utf-8") if _.strip())
        if n > 0:
            print(f"[magicdata_meeting] skip existing manifest ({n} items)")
            return manifest

    raw = dest / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    extracted = raw / "extracted"
    zips = sorted(raw.glob("*.zip")) + sorted(raw.glob("*.tar.gz")) + sorted(raw.glob("*.tgz"))

    if extracted.is_dir() and any(extracted.rglob("*.wav")):
        return _materialize_magicdata_meeting_from_dir(extracted, dest)

    if not zips:
        raise FileNotFoundError(
            "MagicData meeting corpus is not auto-downloadable (Magichub login).\n"
            f"  1) Open {MAGICDATA_MEETING_MAGICHUB}\n"
            "  2) Sign in and download the zip (~202MB)\n"
            f"  3) Place it under {raw}/\n"
            "  4) Re-run: python -m vadbench.eval.download_data --dataset magicdata_meeting"
        )

    archive = zips[0]
    print(f"[magicdata_meeting] extracting {archive.name} ...", flush=True)
    extracted.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(extracted)
    else:
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(extracted)
    return _materialize_magicdata_meeting_from_dir(extracted, dest)


DOWNLOADERS = {
    "aishell4": ("aishell4_test", download_aishell4_test),
    "ami": ("ami_ihm_test", download_ami_ihm_test),
    "alimeeting": ("alimeeting_far_test", download_alimeeting_far_test),
    "alimeeting_eval": ("alimeeting_far_eval", download_alimeeting_far_eval),
    "voxconverse": ("voxconverse_test", download_voxconverse_test),
    "magicdata_ramc": ("magicdata_ramc_test", download_magicdata_ramc_test),
    "magicdata_meeting": ("magicdata_meeting", download_magicdata_meeting),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download VAD eval datasets into repo data/ (prefer ModelScope/OSS)."
    )
    parser.add_argument(
        "--dataset",
        default="alimeeting,alimeeting_eval,ami,aishell4",
        help=(
            "Comma-separated datasets: "
            + ",".join(DOWNLOADERS)
            + ",all  (legacy --lang also supported)"
        ),
    )
    parser.add_argument(
        "--lang",
        default=None,
        choices=("zh", "en", "all"),
        help="Legacy: zh=aishell4+alimeeting, en=ami+voxconverse, all=everything.",
    )
    args = parser.parse_args(argv)
    DATA.mkdir(parents=True, exist_ok=True)

    if args.lang is not None:
        if args.lang == "zh":
            names = ["aishell4", "alimeeting", "alimeeting_eval"]
        elif args.lang == "en":
            names = ["ami"]
            print("[info] voxconverse excluded from --lang en (download abandoned)")
        else:
            names = list(DOWNLOADERS)
    else:
        raw = [x.strip() for x in args.dataset.split(",") if x.strip()]
        if raw == ["all"] or "all" in raw:
            names = list(DOWNLOADERS)
        else:
            names = raw

    unknown = [n for n in names if n not in DOWNLOADERS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}; choose from {list(DOWNLOADERS)}")

    for name in names:
        dirname, fn = DOWNLOADERS[name]
        fn(DATA / dirname)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
