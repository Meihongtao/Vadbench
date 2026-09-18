from __future__ import annotations

import argparse

from vadbench.eval.runner import run_eval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate VAD engines on prepared datasets.")
    parser.add_argument(
        "--engines",
        default="webrtc,silero,funasr_fsmn,firered,pyannote",
        help="Comma-separated engine names.",
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help=(
            "Dataset key(s): aishell4,alimeeting,ami,voxconverse "
            "or zh|en|all or comma list."
        ),
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="Deprecated alias of --dataset (zh|en|all).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Per-dataset file cap.",
    )
    parser.add_argument("--frame-ms", type=float, default=10.0)
    args = parser.parse_args(argv)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    dataset = args.lang if args.lang is not None else args.dataset
    run_eval(
        engines=engines,
        dataset=dataset,
        device=args.device,
        max_files=args.max_files,
        frame_ms=args.frame_ms,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
