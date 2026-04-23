"""
Example: programmatic use of the I-VT algorithm.

This script:
  1. Generates a synthetic gaze trace (or loads one if you swap in a file).
  2. Runs I-VT with custom parameters.
  3. Writes CSV outputs and a summary figure.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ivt import IVTConfig, run_ivt, write_ivt_outputs
from ivt.visualization import plot_ivt_result


def make_demo_trace(random_state: int = 42) -> pd.DataFrame:
    """Four fixations separated by fast saccades, sampled at 100 Hz."""
    rng = np.random.default_rng(random_state)
    sampling_rate_hz = 100
    dt_ms = 1000.0 / sampling_rate_hz

    fixations = [(300, 400), (800, 350), (600, 600), (1000, 500)]
    samples_per_fix = 80   # 800 ms each
    saccade_samples = 4    # 40 ms saccades

    xs, ys, ts = [], [], []
    t = 0.0
    for i, (fx, fy) in enumerate(fixations):
        for _ in range(samples_per_fix):
            xs.append(fx + rng.normal(0, 2.0))
            ys.append(fy + rng.normal(0, 2.0))
            ts.append(t)
            t += dt_ms
        if i < len(fixations) - 1:
            nx, ny = fixations[i + 1]
            for s in range(saccade_samples):
                frac = (s + 1) / (saccade_samples + 1)
                xs.append(fx + frac * (nx - fx))
                ys.append(fy + frac * (ny - fy))
                ts.append(t)
                t += dt_ms
    return pd.DataFrame({"gaze_x": xs, "gaze_y": ys, "timestamp_ms": ts})


def main():
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    df = make_demo_trace()
    sample_csv = out_dir / "demo_trace.csv"
    df.to_csv(sample_csv, index=False)
    print(f"Synthetic trace saved to {sample_csv}")

    # Configure I-VT --------------------------------------------------------
    config = IVTConfig(
        velocity_threshold=1000.0,       # px/s
        min_fixation_duration_ms=60.0,   # ms
        velocity_unit="pixels/s",
    )

    # Run -------------------------------------------------------------------
    result = run_ivt(
        df,
        x_col="gaze_x",
        y_col="gaze_y",
        time_col="timestamp_ms",
        config=config,
        time_unit="ms",
    )

    # Report ---------------------------------------------------------------
    print(f"\nDetected {len(result.fixations)} fixations, "
          f"{len(result.saccades)} saccades.")
    print("\nFixation summary:")
    print(result.fixations.round(2))

    # Save outputs ---------------------------------------------------------
    paths = write_ivt_outputs(result, out_dir, basename="demo")
    for k, v in paths.items():
        print(f"  {k:10s} -> {v}")

    # Plot -----------------------------------------------------------------
    plot_ivt_result(result, save_path=out_dir / "demo_summary.png", show=False)


if __name__ == "__main__":
    main()
