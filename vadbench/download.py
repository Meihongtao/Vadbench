from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
VENDOR = ROOT / "vendor"
MANIFEST_PATH = MODELS / "DOWNLOAD_MANIFEST.json"
HF_MIRROR = "https://hf-mirror.com"

ENGINES = ("webrtc", "silero", "funasr_fsmn", "pyannote", "firered")


def _snapshot_download(repo_id: str, local_dir: Path, token: str | None = None) -> str:
    """Download HF repo; retry with hf-mirror.com on connection errors."""
    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {"repo_id": repo_id, "local_dir": str(local_dir)}
    if token:
        kwargs["token"] = token

    try:
        return snapshot_download(**kwargs)
    except Exception as first:
        print(f"[warn] snapshot_download failed ({first}); retry with {HF_MIRROR}")
        os.environ["HF_ENDPOINT"] = HF_MIRROR
        return snapshot_download(**kwargs)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _git_clone(url: str, dest: Path, depth: int = 1) -> str:
    if dest.exists():
        print(f"[skip] vendor exists: {dest}")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", str(depth), url, str(dest)])
    return _git_rev(dest)


def _git_rev(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def _rel(path: str | Path) -> str:
    """Render a path relative to the repo root, with forward slashes.

    ``DOWNLOAD_MANIFEST.json`` is committed, so it must not record whoever ran
    the download's home directory. Relative paths also stay meaningful for
    anyone reading the manifest on another machine.
    """
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return p.as_posix()


def _write_notes(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def download_webrtc(manifest: dict[str, Any]) -> None:
    rev = _git_clone(
        "https://github.com/wiseman/py-webrtcvad.git",
        VENDOR / "py-webrtcvad",
    )
    manifest["webrtc"] = {
        "vendor": "vendor/py-webrtcvad",
        "repo": "https://github.com/wiseman/py-webrtcvad",
        "commit": rev,
        "model": None,
        "note": "No separate weights; C VAD embedded in extension.",
    }


def download_silero(manifest: dict[str, Any]) -> None:
    rev = _git_clone(
        "https://github.com/snakers4/silero-vad.git",
        VENDOR / "silero-vad",
    )
    dest_dir = MODELS / "silero"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Prefer files shipped in the cloned repo.
    candidates = list((VENDOR / "silero-vad").rglob("silero_vad.onnx"))
    if not candidates:
        # Fallback: hub raw path used by recent releases
        try:
            onnx_dir = dest_dir / "_hf_tmp"
            _snapshot_download("onnx-community/silero-vad", onnx_dir)
            found = list(onnx_dir.rglob("*.onnx"))
            if found:
                shutil.copy2(found[0], dest_dir / "silero_vad.onnx")
                model_rev = "onnx-community/silero-vad"
            else:
                model_rev = None
            shutil.rmtree(onnx_dir, ignore_errors=True)
        except Exception as e:
            print(f"[warn] HF onnx fallback failed: {e}")
            try:
                from silero_vad import load_silero_vad
                import silero_vad

                pkg = Path(silero_vad.__file__).resolve().parent
                found = list(pkg.rglob("*.onnx"))
                if found:
                    shutil.copy2(found[0], dest_dir / "silero_vad.onnx")
                    model_rev = f"pip:{found[0].name}"
                else:
                    load_silero_vad(onnx=True)
                    model_rev = "silero-vad-package-default"
            except Exception as e2:
                print(f"[warn] could not materialize silero onnx: {e2}")
                model_rev = None
    else:
        shutil.copy2(candidates[0], dest_dir / "silero_vad.onnx")
        model_rev = str(candidates[0].relative_to(VENDOR / "silero-vad"))

    manifest["silero"] = {
        "vendor": "vendor/silero-vad",
        "repo": "https://github.com/snakers4/silero-vad",
        "commit": rev,
        "model_dir": "models/silero",
        "model_file": "silero_vad.onnx",
        "model_rev": model_rev,
    }


def download_funasr_fsmn(manifest: dict[str, Any]) -> None:
    dest = MODELS / "funasr_fsmn"
    dest.mkdir(parents=True, exist_ok=True)

    model_rev = None
    try:
        path = _snapshot_download("funasr/fsmn-vad", dest)
        model_rev = f"hf:funasr/fsmn-vad -> {_rel(path)}"
    except Exception as e:
        print(f"[warn] HF download failed ({e}), trying ModelScope id via funasr...")
        try:
            from funasr import AutoModel

            m = AutoModel(model="fsmn-vad", disable_update=True, disable_pbar=True)
            src = Path(m.model_path)
            if src.is_dir():
                for item in src.iterdir():
                    target = dest / item.name
                    if item.is_dir():
                        if target.exists():
                            shutil.rmtree(target)
                        shutil.copytree(item, target)
                    else:
                        shutil.copy2(item, target)
                model_rev = f"funasr-cache:{_rel(src)}"
        except Exception as e2:
            print(f"[error] funasr_fsmn model download failed: {e2}")

    # Sparse notes instead of full FunASR clone
    _write_notes(
        VENDOR / "FunASR.NOTES.md",
        """# FunASR vendor note

Full repo: https://github.com/modelscope/FunASR (large).

This project does not vendor the entire toolkit. Runtime uses `pip install funasr`.
VAD model weights are stored under `models/funasr_fsmn/`.

Official standalone usage:

```python
from funasr import AutoModel
model = AutoModel(model="fsmn-vad", disable_update=True)
print(model.generate(input="audio.wav")[0]["value"])
```

VAD model sources:
- https://huggingface.co/funasr/fsmn-vad
- ModelScope: damo/speech_fsmn_vad_zh-cn-16k-common-pytorch
""",
    )

    manifest["funasr_fsmn"] = {
        "vendor_notes": "vendor/FunASR.NOTES.md",
        "repo": "https://github.com/modelscope/FunASR",
        "model_dir": "models/funasr_fsmn",
        "model_rev": model_rev,
    }


def download_pyannote(manifest: dict[str, Any]) -> None:
    """Download pyannote weights from ModelScope into models/pyannote/."""
    dest_root = MODELS / "pyannote"
    dest_root.mkdir(parents=True, exist_ok=True)
    model_rev: dict[str, Any] = {}

    # Prefer ModelScope mirrors (no HF gated token).
    ms_candidates = [
        ("pyannote/segmentation-3.0", "segmentation-3.0"),
        ("pyannote/speaker-diarization-3.1", "speaker-diarization-3.1"),
        ("AI-ModelScope/pyannote-segmentation-3.0", "segmentation-3.0"),
    ]

    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_download
    except ImportError:
        ms_download = None
        print("[warn] modelscope not installed; pip install modelscope")

    for model_id, sub in ms_candidates:
        local_dir = dest_root / sub
        if sub == "segmentation-3.0":
            has_bin = local_dir.is_dir() and (
                any(local_dir.glob("pytorch_model*"))
                or any(local_dir.glob("*.safetensors"))
            )
        else:
            has_bin = local_dir.is_dir() and (
                any(local_dir.glob("pytorch_model*"))
                or any(local_dir.glob("*.safetensors"))
                or (local_dir / "config.yaml").is_file()
            )
        if has_bin:
            print(f"[skip] exists: {local_dir}")
            model_rev[model_id] = f"exists:{_rel(local_dir)}"
            continue
        if ms_download is None:
            model_rev[model_id] = "FAILED: modelscope missing"
            continue
        try:
            print(f"[modelscope] {model_id} -> {local_dir}")
            # avoid incomplete stub dirs
            if local_dir.exists() and not has_bin:
                shutil.rmtree(local_dir)
            path = ms_download(model_id, local_dir=str(local_dir))
            model_rev[model_id] = f"modelscope:{_rel(path)}"
        except Exception as e:
            print(f"[warn] ModelScope download failed for {model_id}: {e}")
            model_rev[model_id] = f"FAILED: {e}"

    # HF fallback only if nothing usable landed
    has_weights = any(
        (dest_root / name).is_dir() and any((dest_root / name).iterdir())
        for name in ("segmentation-3.0", "speaker-diarization-3.1", "voice-activity-detection")
    )
    if not has_weights:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        for repo_id, sub in (
            ("pyannote/segmentation-3.0", "segmentation-3.0"),
            ("pyannote/voice-activity-detection", "voice-activity-detection"),
        ):
            try:
                path = _snapshot_download(repo_id, dest_root / sub, token=token)
                model_rev[f"hf:{repo_id}"] = _rel(path)
            except Exception as e:
                model_rev[f"hf:{repo_id}"] = f"FAILED: {e}"

    _write_notes(
        VENDOR / "pyannote-audio.NOTES.md",
        """# pyannote.audio vendor note

Full repo: https://github.com/pyannote/pyannote-audio (large).

Runtime: `pip install pyannote.audio`.

Default weight source: **ModelScope** (no HF token required):
- https://modelscope.cn/models/pyannote/speaker-diarization-3.1
- Search ModelScope for `pyannote segmentation` / `segmentation-3.0`

Local cache: `models/pyannote/`

```bash
python -m vadbench.download --engine pyannote
```
""",
    )

    manifest["pyannote"] = {
        "vendor_notes": "vendor/pyannote-audio.NOTES.md",
        "repo": "https://github.com/pyannote/pyannote-audio",
        "model_dir": "models/pyannote",
        "model_rev": model_rev,
        "source": "modelscope",
    }


def download_firered(manifest: dict[str, Any]) -> None:
    rev = _git_clone(
        "https://github.com/FireRedTeam/FireRedVAD.git",
        VENDOR / "FireRedVAD",
    )
    dest = MODELS / "firered"
    dest.mkdir(parents=True, exist_ok=True)

    model_rev = None
    try:
        path = _snapshot_download("FireRedTeam/FireRedVAD", dest)
        model_rev = f"hf:FireRedTeam/FireRedVAD -> {_rel(path)}"
    except Exception as e:
        print(f"[warn] HF FireRedVAD download failed: {e}")
        try:
            from modelscope.hub.snapshot_download import (
                snapshot_download as ms_snapshot_download,
            )

            path = ms_snapshot_download(
                "xukaituo/FireRedVAD",
                local_dir=str(dest),
            )
            model_rev = f"modelscope:xukaituo/FireRedVAD -> {_rel(path)}"
        except Exception as e2:
            print(f"[error] FireRedVAD model download failed: {e2}")

    # Normalize layout to models/firered/VAD if nested
    for nested in (
        dest / "FireRedVAD" / "VAD",
        dest / "VAD",
    ):
        if (nested / "model.pth.tar").is_file():
            flat = dest / "VAD"
            if nested.resolve() != flat.resolve():
                if flat.exists():
                    shutil.rmtree(flat)
                shutil.copytree(nested, flat)
            break
    # Also copy ONNX reference from vendor if present
    onnx_src = VENDOR / "FireRedVAD" / "pretrained_models" / "onnx_models"
    onnx_dst = dest / "onnx_models"
    if onnx_src.is_dir() and not onnx_dst.exists():
        shutil.copytree(onnx_src, onnx_dst)

    manifest["firered"] = {
        "vendor": "vendor/FireRedVAD",
        "repo": "https://github.com/FireRedTeam/FireRedVAD",
        "commit": rev,
        "model_dir": "models/firered",
        "model_rev": model_rev,
    }

DOWNLOADERS = {
    "webrtc": download_webrtc,
    "silero": download_silero,
    "funasr_fsmn": download_funasr_fsmn,
    "pyannote": download_pyannote,
    "firered": download_firered,
}


def load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.is_file():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict[str, Any]) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    manifest["_updated_at"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {MANIFEST_PATH}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download VAD models and archive official source repos."
    )
    parser.add_argument(
        "--engine",
        action="append",
        dest="engines",
        help="Engine name or 'all'. Repeatable.",
    )
    args = parser.parse_args(argv)
    engines = args.engines or ["all"]
    selected: list[str] = []
    for e in engines:
        if e.lower() == "all":
            selected = list(ENGINES)
            break
        if e not in DOWNLOADERS:
            print(f"Unknown engine: {e}. Choose from: {', '.join(ENGINES)}")
            return 2
        selected.append(e)

    VENDOR.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    for name in selected:
        print(f"\n=== download {name} ===")
        DOWNLOADERS[name](manifest)

    save_manifest(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
