"""Example: generate synthetic gaze data and run the I-VT pipeline.

Usage:
    python examples/example_usage.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running this script directly without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ivt import IVTConfig, run_ivt, summarize_events  # noqa: E402


def generate_synthetic_gaze(
    n_fixations: int = 8,
    fixation_duration_ms: float = 250.0,
    saccade_duration_ms: float = 40.0,
    sampling_rate_hz: float = 250.0,
    fixation_jitter_px: float = 2.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Produce a toy gaze trace alternating fixations and saccades."""
    rng = np.random.default_rng(seed)
    dt_ms = 1000.0 / sampling_rate_hz

    # Random fixation targets across a 1920x1080 "screen".
    targets = rng.uniform(low=[100, 100], high=[1820, 980], size=(n_fixations, 2))

    rows = []
    t = 0.0
    prev = targets[0].copy()
    for i, target in enumerate(targets):
        # --- Saccade to the target (skip on first) ---
        if i > 0:
            n_sac = max(2, int(saccade_duration_ms / dt_ms))
            for step in range(1, n_sac + 1):
                alpha = step / n_sac
                x = (1 - alpha) * prev[0] + alpha * target[0]
                y = (1 - alpha) * prev[1] + alpha * target[1]
                rows.append((t, x, y))
                t += dt_ms
        # --- Fixation at the target with small jitter ---
        n_fix = int(fixation_duration_ms / dt_ms)
        for _ in range(n_fix):
            x = target[0] + rng.normal(0, fixation_jitter_px)
            y = target[1] + rng.normal(0, fixation_jitter_px)
            rows.append((t, x, y))
            t += dt_ms
        prev = target

    return pd.DataFrame(rows, columns=["t_ms", "gaze_x", "gaze_y"])


def main():
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    # 1. Make synthetic data and save as CSV to simulate user input.
    df = generate_synthetic_gaze()
    input_path = out_dir / "synthetic_gaze.csv"
    df.to_csv(input_path, index=False)
    print(f"Wrote synthetic input: {input_path}  ({len(df)} samples)")

    # 2. Run I-VT.
    cfg = IVTConfig(velocity_threshold=1000.0, min_fixation_duration_ms=60.0)
    result = run_ivt(df, x_col="gaze_x", y_col="gaze_y", time_col="t_ms",
                     time_unit="ms", config=cfg)

    labelled_path = out_dir / "synthetic_gaze_ivt.csv"
    result.to_csv(labelled_path, index=False)
    print(f"Wrote labelled output:  {labelled_path}")

    # 3. Event-level summary.
    events = summarize_events(result, x_col="gaze_x", y_col="gaze_y")
    events_path = out_dir / "synthetic_gaze_events.csv"
    events.to_csv(events_path, index=False)
    print(f"Wrote event summary:    {events_path}")

    n_fix = (events.event_type == "fixation").sum()
    n_sac = (events.event_type == "saccade").sum()
    mean_fix = events.loc[events.event_type == "fixation", "duration_ms"].mean()
    print(f"\nDetected {n_fix} fixations and {n_sac} saccades.")
    print(f"Mean fixation duration: {mean_fix:.1f} ms")


if __name__ == "__main__":
    main()
