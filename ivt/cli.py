"""
Command-line interface for the I-VT project.

Two modes:
  1. Interactive   -- prompts the user to pick file, columns, settings.
  2. Non-interactive -- all options provided as CLI arguments.

Usage examples
--------------
Interactive:
    python -m ivt.cli

Non-interactive:
    python -m ivt.cli \\
        --input data/sample.csv \\
        --x-col gaze_x --y-col gaze_y --time-col timestamp \\
        --time-unit ms \\
        --velocity-threshold 1000 \\
        --min-fixation-duration 60 \\
        --output-dir results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .ivt_algorithm import (
    IVTConfig,
    run_ivt,
    DEFAULT_VELOCITY_THRESHOLD_PIX,
    DEFAULT_VELOCITY_THRESHOLD_DEG,
    DEFAULT_MIN_FIXATION_DURATION_MS,
)
from .io_utils import read_gaze_file, list_excel_sheets, write_ivt_outputs


# -----------------------------------------------------------------------------
# Interactive helpers
# -----------------------------------------------------------------------------
def _prompt(prompt: str, default=None) -> str:
    """Prompt user with an optional default value."""
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    if not val and default is not None:
        return default
    return val


def _prompt_float(prompt: str, default: float) -> float:
    while True:
        raw = _prompt(prompt, str(default))
        try:
            return float(raw)
        except ValueError:
            print(f"  '{raw}' is not a valid number. Try again.")


def _prompt_choice(prompt: str, choices: list, default_index: int = 0):
    print(prompt)
    for i, c in enumerate(choices):
        marker = " (default)" if i == default_index else ""
        print(f"  [{i}] {c}{marker}")
    while True:
        raw = input(
            f"Select index [0-{len(choices)-1}] (default {default_index}): "
        ).strip()
        if not raw:
            return choices[default_index]
        if raw.isdigit() and 0 <= int(raw) < len(choices):
            return choices[int(raw)]
        print("  Invalid selection. Try again.")


def run_interactive() -> None:
    print("=" * 60)
    print("  I-VT (Velocity Threshold Identification) -- Interactive")
    print("=" * 60)

    # --- 1. File -----------------------------------------------------------
    while True:
        path_str = _prompt("Path to CSV / Excel file")
        if not path_str:
            print("  A file path is required.")
            continue
        path = Path(path_str).expanduser()
        if path.exists():
            break
        print(f"  File not found: {path}")

    # --- 2. Sheet (Excel only) --------------------------------------------
    sheet_name = 0
    if path.suffix.lower() in {".xlsx", ".xls"}:
        sheets = list_excel_sheets(path)
        if len(sheets) > 1:
            sheet_name = _prompt_choice(
                "\nMultiple sheets found. Choose one:", sheets
            )
        else:
            sheet_name = sheets[0]

    df = read_gaze_file(path, sheet_name=sheet_name)
    print(f"\nLoaded file with {len(df)} rows and {len(df.columns)} columns.")
    print(f"First 5 rows:\n{df.head()}\n")

    # --- 3. Column selection ----------------------------------------------
    cols = list(df.columns)
    x_col = _prompt_choice("\nSelect X-coordinate column (pixels):", cols)
    y_col = _prompt_choice("\nSelect Y-coordinate column (pixels):", cols)
    time_col = _prompt_choice("\nSelect timestamp column:", cols)

    # --- 4. Time unit -----------------------------------------------------
    time_unit = _prompt_choice(
        "\nSelect time unit of the timestamp column:",
        ["ms", "s", "us", "ns"],
        default_index=0,
    )

    # --- 5. Velocity unit -------------------------------------------------
    velocity_unit = _prompt_choice(
        "\nSelect velocity threshold unit:",
        ["pixels/s", "deg/s"],
        default_index=0,
    )

    pixels_per_degree = None
    if velocity_unit == "deg/s":
        pixels_per_degree = _prompt_float(
            "Pixels per degree of visual angle", 35.0
        )
        default_threshold = DEFAULT_VELOCITY_THRESHOLD_DEG
    else:
        default_threshold = DEFAULT_VELOCITY_THRESHOLD_PIX

    velocity_threshold = _prompt_float(
        f"Velocity threshold ({velocity_unit})", default_threshold
    )
    min_fix_dur = _prompt_float(
        "Minimum fixation duration (ms)", DEFAULT_MIN_FIXATION_DURATION_MS
    )

    # --- 6. Output dir ----------------------------------------------------
    out_dir = _prompt("Output directory", str(path.parent / "ivt_output"))

    # --- 7. Run -----------------------------------------------------------
    config = IVTConfig(
        velocity_threshold=velocity_threshold,
        min_fixation_duration_ms=min_fix_dur,
        velocity_unit=velocity_unit,
        pixels_per_degree=pixels_per_degree,
    )
    print("\nRunning I-VT ...")
    result = run_ivt(
        df, x_col=x_col, y_col=y_col, time_col=time_col,
        config=config, time_unit=time_unit,
    )

    paths = write_ivt_outputs(result, out_dir, basename=path.stem)
    _report(result, paths)


def _report(result, paths: dict) -> None:
    n_samples = len(result.data)
    n_fix = len(result.fixations)
    n_sac = len(result.saccades)
    fix_pct = 100.0 * (result.data["eye_movement_type"] == "fixation").mean()

    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"Samples:   {n_samples}")
    print(f"Fixations: {n_fix}  ({fix_pct:.1f}% of samples)")
    print(f"Saccades:  {n_sac}")
    if n_fix > 0:
        mean_fix = result.fixations["duration_ms"].mean()
        median_fix = result.fixations["duration_ms"].median()
        print(f"Fixation duration -- mean: {mean_fix:.1f} ms, "
              f"median: {median_fix:.1f} ms")
    print("\nOutputs written to:")
    for k, v in paths.items():
        print(f"  {k:10s} -> {v}")


# -----------------------------------------------------------------------------
# Non-interactive (argparse) entry point
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ivt",
        description="Velocity Threshold Identification (I-VT) for gaze data.",
    )
    p.add_argument("--input", "-i", help="Path to CSV/Excel file.")
    p.add_argument("--sheet", default=0,
                   help="Sheet name or index for Excel files (default: 0).")
    p.add_argument("--x-col", help="X coordinate column name.")
    p.add_argument("--y-col", help="Y coordinate column name.")
    p.add_argument("--time-col", help="Timestamp column name.")
    p.add_argument("--time-unit", default="ms",
                   choices=["s", "ms", "us", "ns"],
                   help="Unit of the timestamp column (default: ms).")
    p.add_argument("--velocity-unit", default="pixels/s",
                   choices=["pixels/s", "deg/s"],
                   help="Unit for the velocity threshold.")
    p.add_argument("--pixels-per-degree", type=float, default=None,
                   help="Required if --velocity-unit deg/s.")
    p.add_argument("--velocity-threshold", type=float, default=None,
                   help="Velocity threshold; default depends on unit.")
    p.add_argument("--min-fixation-duration", type=float,
                   default=DEFAULT_MIN_FIXATION_DURATION_MS,
                   help=f"Minimum fixation duration in ms "
                        f"(default: {DEFAULT_MIN_FIXATION_DURATION_MS}).")
    p.add_argument("--output-dir", "-o", default=None,
                   help="Output directory (default: <input_dir>/ivt_output).")
    p.add_argument("--interactive", action="store_true",
                   help="Run in interactive prompt mode.")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.interactive or args.input is None:
        run_interactive()
        return 0

    required = [args.x_col, args.y_col, args.time_col]
    if not all(required):
        parser.error("--x-col, --y-col, and --time-col are required "
                     "in non-interactive mode.")

    path = Path(args.input).expanduser()
    df = read_gaze_file(path, sheet_name=args.sheet)

    if args.velocity_threshold is None:
        threshold = (DEFAULT_VELOCITY_THRESHOLD_DEG
                     if args.velocity_unit == "deg/s"
                     else DEFAULT_VELOCITY_THRESHOLD_PIX)
    else:
        threshold = args.velocity_threshold

    config = IVTConfig(
        velocity_threshold=threshold,
        min_fixation_duration_ms=args.min_fixation_duration,
        velocity_unit=args.velocity_unit,
        pixels_per_degree=args.pixels_per_degree,
    )
    result = run_ivt(
        df,
        x_col=args.x_col,
        y_col=args.y_col,
        time_col=args.time_col,
        config=config,
        time_unit=args.time_unit,
    )

    out_dir = args.output_dir or (path.parent / "ivt_output")
    paths = write_ivt_outputs(result, out_dir, basename=path.stem)
    _report(result, paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
