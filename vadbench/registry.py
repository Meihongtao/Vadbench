from __future__ import annotations

import re
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

# Core dependencies, needed by every engine. If one of these is what's missing,
# pointing at an engine extra would be misleading.
_CORE_MODULES: frozenset[str] = frozenset(
    {"numpy", "soundfile", "huggingface_hub", "requests"}
)

_MISSING_MODULE_RE = re.compile(r"No module named '([^']+)'")


def _install_hint(engine: str, reason: str) -> str:
    """Turn an ImportError string into the pip command that actually fixes it."""
    match = _MISSING_MODULE_RE.search(reason)
    module = match.group(1).split(".")[0] if match else ""
    if module in _CORE_MODULES:
        return (
            " That is a core dependency, not an engine extra -- reinstall the "
            'package: pip install -e "."'
        )
    extra = _ENGINE_EXTRA.get(engine)
    if extra:
        return f' Try: pip install -e ".[{extra}]"'
    return ""


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
            raise ImportError(
                f"VAD '{key}' is registered but its dependencies are missing: "
                f"{missing}.{_install_hint(key, missing)}"
            )
        raise KeyError(
            f"Unknown VAD '{name}'. Available: {', '.join(available()) or '(none loaded)'}"
        )
    return _REGISTRY[key](**kwargs)
