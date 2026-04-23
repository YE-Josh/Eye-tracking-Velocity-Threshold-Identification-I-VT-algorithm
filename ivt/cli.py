"""Command-line interface for the I-VT classifier.

Run:
    python -m ivt.cli                     # fully interactive
    python -m ivt.cli --input data.csv    # interactive column picker
    python -m ivt.cli --input data.csv --x gaze_x --y gaze_y --time t \\
        --time-unit ms --threshold 1000 --output labelled.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .algorithm import IVTConfig, run_ivt, summarize_events
from .io import read_gaze_file, write_gaze_file


# -------------------------------------------------------------------
# Interactive prompts
# -------------------------------------------------------------------
def _prompt_path(prompt: str) -> Path:
    while True:
        raw = input(prompt).strip().strip('"').strip("'")
        if not raw:
            print("  Please enter a path.")
            continue
        p = Path(raw).expanduser()
        if not p.exists():
            print(f"  File not found: {p}")
            continue
        return p


def _prompt_choice(prompt: str, choices: list[str]) -> str:
    print(prompt)
    for i, c in enumerate(choices, start=1):
        print(f"  [{i}] {c}")
    while True:
        raw = input("Enter number or column name: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
        elif raw in choices:
            return raw
        print("  Invalid choice, try again.")


def _prompt_time_unit() -> str:
    print("Time unit of the time column:")
    print("  [1] ms  (milliseconds)  -- default")
    print("  [2] s   (seconds)")
    print("  [3] us  (microseconds)")
    raw = input("Enter choice [1-3] (default 1): ").strip() or "1"
    return {"1": "ms", "2": "s", "3": "us"}.get(raw, "ms")


def _prompt_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} (default {default}): ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  Invalid number, using default {default}.")
        return default


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Classify eye-tracking samples into fixations and saccades "
        "using the I-VT algorithm (Salvucci & Goldberg, 2000)."
    )
    p.add_argument("--input", "-i", type=str, help="Input CSV/TSV/XLSX file.")
    p.add_argument("--output", "-o", type=str, help="Output file path.")
    p.add_argument("--x", dest="x_col", type=str, help="Name of the x column (pixels).")
    p.add_argument("--y", dest="y_col", type=str, help="Name of the y column (pixels).")
    p.add_argument("--time", dest="time_col", type=str, help="Name of the time column.")
    p.add_argument(
        "--time-unit",
        choices=["ms", "s", "us"],
        default=None,
        help="Unit of the time column (default: ms).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Velocity threshold in pixels/second (default: 1000).",
    )
    p.add_argument(
        "--min-fixation-ms",
        type=float,
        default=None,
        help="Minimum fixation duration in ms (default: 60).",
    )
    p.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable merging of adjacent fixations.",
    )
    p.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Optional path to write an event-level summary (CSV/XLSX).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # ---- Input file ----
    if args.input:
        in_path = Path(args.input)
        if not in_path.exists():
            print(f"Input file not found: {in_path}", file=sys.stderr)
            return 1
    else:
        in_path = _prompt_path("Path to input file (.csv / .xlsx): ")

    print(f"\nReading {in_path} ...")
    df = read_gaze_file(in_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    cols = list(df.columns)

    # ---- Column selection ----
    x_col = args.x_col or _prompt_choice("\nSelect the X column (pixels):", cols)
    y_col = args.y_col or _prompt_choice("\nSelect the Y column (pixels):", cols)
    time_col = args.time_col or _prompt_choice("\nSelect the TIME column:", cols)

    time_unit = args.time_unit or _prompt_time_unit()

    # ---- Config ----
    threshold = (
        args.threshold
        if args.threshold is not None
        else _prompt_float("\nVelocity threshold in px/s", 1000.0)
    )
    min_fix = (
        args.min_fixation_ms
        if args.min_fixation_ms is not None
        else _prompt_float("Minimum fixation duration in ms", 60.0)
    )
    config = IVTConfig(
        velocity_threshold=threshold,
        min_fixation_duration_ms=min_fix,
        merge_adjacent_fixations=not args.no_merge,
    )

    # ---- Run ----
    print("\nRunning I-VT classification ...")
    result = run_ivt(
        df,
        x_col=x_col,
        y_col=y_col,
        time_col=time_col,
        time_unit=time_unit,
        config=config,
    )

    n_fix = int((result["eye_movement"] == "fixation").sum())
    n_sac = int((result["eye_movement"] == "saccade").sum())
    print(f"  Samples: {len(result)}  |  fixation={n_fix}  saccade={n_sac}")

    # ---- Output path ----
    if args.output:
        out_path = Path(args.output)
    else:
        default_out = in_path.with_name(in_path.stem + "_ivt" + in_path.suffix)
        raw = input(f"\nOutput path (default {default_out}): ").strip()
        out_path = Path(raw) if raw else default_out

    write_gaze_file(result, out_path)
    print(f"Wrote labelled data -> {out_path}")

    # ---- Event summary ----
    if args.summary is not None:
        summary = summarize_events(result, x_col=x_col, y_col=y_col)
        write_gaze_file(summary, args.summary)
        print(f"Wrote event summary -> {args.summary}")
    elif not args.output:
        raw = input("Also write an event-level summary? [y/N]: ").strip().lower()
        if raw == "y":
            summary = summarize_events(result, x_col=x_col, y_col=y_col)
            summary_path = out_path.with_name(out_path.stem + "_events.csv")
            write_gaze_file(summary, summary_path)
            print(f"Wrote event summary -> {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
