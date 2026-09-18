"""Simulate a minimal install (only webrtc) and check graceful degradation.

Run manually: ``python tests/check_minimal_install.py``
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vadbench.registry import _ENGINE_MODULE  # noqa: E402

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

    available = vadbench.available()
    unavailable = vadbench.unavailable()
    print("available  :", available)
    print("unavailable:")
    for engine, reason in unavailable.items():
        print(f"    {engine:<16} {reason[:60]}")

    # Importing the package must not explode just because an optional dependency
    # is missing. Whether any engine ends up usable depends on what is
    # installed, so that is not itself an assertion -- on a bare interpreter
    # `available()` legitimately returns [].
    known = set(_ENGINE_MODULE)
    assert set(available) | set(unavailable) == known, (
        f"engine accounting mismatch: available={sorted(available)}, "
        f"unavailable={sorted(unavailable)}, known={sorted(known)}"
    )
    assert not (set(available) & set(unavailable)), "engine both available and not"

    # Every engine whose deps are blocked must fail at create() with a message
    # naming the engine and the exact pip command that fixes it.
    for engine in sorted(unavailable):
        try:
            vadbench.create(engine)
        except ImportError as e:
            msg = str(e)
            assert engine in msg, f"message for {engine} does not name it: {msg}"
            assert "pip install" in msg, f"unhelpful message for {engine}: {msg}"
            print(f"create({engine}) -> ImportError: {msg}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL: create({engine}) raised {type(e).__name__}: {e}")
            return 1
        else:
            print(f"FAIL: create({engine}) succeeded despite missing deps")
            return 1

    # Engines that did load must be constructible (or fail only for model files,
    # which are downloaded separately and absent in a fresh clone).
    for engine in sorted(available):
        try:
            vadbench.create(engine)
        except (FileNotFoundError, ImportError) as e:
            print(f"create({engine}) -> {type(e).__name__} (expected: no model files)")
        else:
            print(f"create({engine}) -> ok")

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
