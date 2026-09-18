"""Simulate a minimal install (only webrtc) and check graceful degradation.

Run manually: ``python tests/check_minimal_install.py``
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BLOCKED = {
    "torch",
    "torchaudio",
    "onnxruntime",
    "funasr",
    "pyannote",
    "webrtcvad",
    "silero_vad",
    "fireredvad",
}

_real_import = builtins.__import__


def _fake_import(name, *args, **kwargs):
    if name.split(".")[0] in BLOCKED:
        raise ImportError(f"No module named {name.split('.')[0]!r}")
    return _real_import(name, *args, **kwargs)


def main() -> int:
    # Drop already-imported optional deps and backend modules so the block bites.
    for mod in list(sys.modules):
        root = mod.split(".")[0]
        if root in BLOCKED or mod.startswith("vadbench.backends"):
            del sys.modules[mod]
    builtins.__import__ = _fake_import

    import vadbench

    print("available  :", vadbench.available())
    if not vadbench.available():
        print("FAIL: nothing registered; minimal install should still expose engines")
        return 1

    print("unavailable:")
    for engine, reason in vadbench.unavailable().items():
        print(f"    {engine:<16} {reason[:60]}")

    # An engine whose deps are missing must fail loudly at create() time, and
    # the message must say which optional group to install.
    for engine in ("pyannote", "funasr_fsmn", "silero"):
        try:
            vadbench.create(engine)
        except ImportError as e:
            assert "pip install" in str(e), f"unhelpful message for {engine}: {e}"
            print(f"create({engine}) -> ImportError: {e}")
        else:
            if engine in vadbench.available():
                print(f"create({engine}) -> constructed (deps actually present)")
            else:
                print(f"FAIL: create({engine}) neither raised nor registered")
                return 1

    # A genuinely unknown name is still a KeyError, not an ImportError.
    try:
        vadbench.create("does_not_exist")
    except KeyError:
        print("create(does_not_exist) -> KeyError (correct)")
    else:
        print("FAIL: unknown engine did not raise KeyError")
        return 1

    print("\nok: minimal install degrades gracefully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
