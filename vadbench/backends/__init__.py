"""VAD backend adapters.

Importing a backend module registers its engines as a side effect. Engines have
very different dependency footprints — ``webrtc`` needs only a C extension while
``pyannote`` pulls in torch — so a minimal install (``pip install -e ".[webrtc]"``)
would otherwise make ``vadbench.available()`` explode just because one optional
dependency is missing.

Each module is therefore imported defensively. A backend that cannot be imported
is simply absent from the registry, and ``vadbench.registry`` reports the reason
if you try to use it. Only ``ImportError`` is swallowed: a syntax error or a typo
inside one of these modules is a real bug and must still surface.
"""

from __future__ import annotations

import importlib

BACKEND_MODULES: tuple[str, ...] = (
    "webrtc",
    "silero",
    "silero_onnx",
    "funasr_fsmn",
    "funasr_fsmn_onnx",
    "pyannote_vad",
    "pyannote_onnx",
    "firered",
    "firered_onnx",
)

# module name -> ImportError message, for engines that could not be loaded.
IMPORT_ERRORS: dict[str, str] = {}


def _load(name: str) -> None:
    try:
        importlib.import_module(f"vadbench.backends.{name}")
    except ImportError as e:
        IMPORT_ERRORS[name] = str(e)


for _name in BACKEND_MODULES:
    _load(_name)

del _name
