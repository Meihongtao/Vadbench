from __future__ import annotations

import shutil
from pathlib import Path

from vadbench.perf.ort_utils import MODELS, project_root


def _copy_if_needed(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    if not src.is_file():
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)
    return dst


def export_silero() -> Path:
    src = MODELS / "silero" / "silero_vad.onnx"
    dst = MODELS / "silero" / "onnx" / "silero_vad.onnx"
    return _copy_if_needed(src, dst)


def export_firered() -> Path:
    candidates = [
        MODELS / "firered" / "onnx_models" / "fireredvad_vad.onnx",
        MODELS / "firered" / "onnx_models" / "onnx_models" / "fireredvad_vad.onnx",
        project_root()
        / "vendor"
        / "FireRedVAD"
        / "pretrained_models"
        / "onnx_models"
        / "fireredvad_vad.onnx",
    ]
    src = next((p for p in candidates if p.is_file()), None)
    if src is None:
        raise FileNotFoundError(
            "FireRed ONNX not found. Run: python -m vadbench.download --engine firered"
        )
    dst = MODELS / "firered" / "onnx" / "fireredvad_vad.onnx"
    out = _copy_if_needed(src, dst)
    # Keep CMVN next to ONNX for convenience (feat extractor still uses VAD dir).
    for vad_dir in (
        MODELS / "firered" / "VAD",
        MODELS / "firered" / "FireRedVAD" / "VAD",
        project_root()
        / "vendor"
        / "FireRedVAD"
        / "pretrained_models"
        / "FireRedVAD"
        / "VAD",
    ):
        cmvn = vad_dir / "cmvn.ark"
        if cmvn.is_file():
            _copy_if_needed(cmvn, dst.parent / "cmvn.ark")
            break
    return out


def export_funasr() -> Path:
    dst = MODELS / "funasr_fsmn" / "onnx" / "model.onnx"
    if dst.is_file() and dst.stat().st_size > 1000:
        return dst
    import torch
    import yaml
    from funasr import AutoModel
    from funasr.models.fsmn_vad_streaming.export_meta import export_rebuild_model

    model_dir = MODELS / "funasr_fsmn"
    cfg = yaml.safe_load((model_dir / "config.yaml").read_text(encoding="utf-8"))
    am = AutoModel(
        model=str(model_dir),
        device="cpu",
        disable_update=True,
        disable_pbar=True,
    )
    model = export_rebuild_model(
        model=am.model, type="onnx", encoder=cfg.get("encoder", "FSMN")
    )
    dummy = model.export_dummy_inputs()
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(dst),
        opset_version=14,
        input_names=model.export_input_names(),
        output_names=model.export_output_names(),
        dynamic_axes=model.export_dynamic_axes(),
        dynamo=False,
    )
    return dst


def export_pyannote() -> Path:
    dst = MODELS / "pyannote" / "onnx" / "segmentation-3.0.onnx"
    if dst.is_file() and dst.stat().st_size > 1000:
        return dst
    import torch
    from pyannote.audio import Model

    seg_dir = MODELS / "pyannote" / "segmentation-3.0"
    model = Model.from_pretrained(str(seg_dir))
    model.eval()
    # duration=10s @ 16 kHz
    dummy = torch.randn(1, 1, 160000)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(dst),
        opset_version=14,
        input_names=["waveform"],
        output_names=["scores"],
        dynamic_axes={
            "waveform": {0: "batch", 2: "time"},
            "scores": {0: "batch", 1: "frames"},
        },
        dynamo=False,
    )
    return dst


EXPORTERS = {
    "silero": export_silero,
    "firered": export_firered,
    "funasr_fsmn": export_funasr,
    "pyannote": export_pyannote,
}


def export_all(engines: list[str] | None = None) -> dict[str, Path]:
    names = engines or list(EXPORTERS)
    out: dict[str, Path] = {}
    for name in names:
        if name not in EXPORTERS:
            if name == "webrtc":
                continue
            raise KeyError(f"No ONNX exporter for {name}")
        print(f"[export] {name} ...", flush=True)
        path = EXPORTERS[name]()
        print(f"[export] {name} -> {path} ({path.stat().st_size} bytes)", flush=True)
        out[name] = path
    return out


if __name__ == "__main__":
    export_all()
