from __future__ import annotations

from typing import Any

from vadbench.base import BaseVad

_REGISTRY: dict[str, type[BaseVad]] = {}

# Engine name -> backend module, for engines that could not be imported.
_ENGINE_MODULE: dict[str, str] = {
    "webrtc": "webrtc",
    "silero": "silero",
    "silero_onnx": "silero_onnx",
    "funasr_fsmn": "funasr_fsmn",
    "funasr_fsmn_onnx": "funasr_fsmn_onnx",
    "pyannote": "pyannote_vad",
    "pyannote_onnx": "pyannote_onnx",
    "firered": "firered",
    "firered_onnx": "firered_onnx",
}

# Extra pip package (the optional-dependency group) needed per engine, used to
# make "engine missing" errors actionable instead of just "not registered".
_ENGINE_EXTRA: dict[str, str] = {
    "webrtc": "webrtc",
    "silero": "silero",
    "silero_onnx": "silero",
    "funasr_fsmn": "funasr",
    "funasr_fsmn_onnx": "funasr",
    "pyannote": "pyannote",
    "pyannote_onnx": "pyannote",
    "firered": "firered",
    "firered_onnx": "firered",
}


def register(name: str):
    """Decorator to register a VAD backend class."""

    def deco(cls: type[BaseVad]):
        key = name.lower().strip()
        _REGISTRY[key] = cls
        return cls

    return deco


def available() -> list[str]:
    import vadbench.backends  # noqa: F401

    return sorted(_REGISTRY.keys())


def unavailable() -> dict[str, str]:
    """Engines whose backend module failed to import, with the reason."""
    import vadbench.backends as backends

    out: dict[str, str] = {}
    for engine, module in _ENGINE_MODULE.items():
        if engine in _REGISTRY:
            continue
        reason = backends.IMPORT_ERRORS.get(module)
        if reason:
            out[engine] = reason
    return out


def create(name: str, **kwargs: Any) -> BaseVad:
    """Create a VAD backend by name.

    Names: ``webrtc``, ``silero``, ``funasr_fsmn``, ``pyannote``, ``firered``
    (plus ``*_onnx`` variants).
    """
    import vadbench.backends  # noqa: F401

    key = name.lower().strip()
    if key not in _REGISTRY:
        missing = unavailable().get(key)
        if missing is not None:
            extra = _ENGINE_EXTRA.get(key)
            hint = f' pip install -e ".[{extra}]"' if extra else ""
            raise ImportError(
                f"VAD '{key}' is registered but its dependencies are missing: "
                f"{missing}.{hint}"
            )
        raise KeyError(
            f"Unknown VAD '{name}'. Available: {', '.join(available()) or '(none loaded)'}"
        )
    return _REGISTRY[key](**kwargs)
