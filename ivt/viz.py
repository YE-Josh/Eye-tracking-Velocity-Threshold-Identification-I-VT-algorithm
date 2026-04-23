"""Optional visualisation helpers. Requires matplotlib."""

from __future__ import annotations

import numpy as np
import pandas as pd


def plot_classification(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    time_ms_col: str = "time_ms",
    velocity_col: str = "velocity_pxs",
    label_col: str = "eye_movement",
    threshold: float | None = None,
    save_path: str | None = None,
    show: bool = True,
):
    """Plot the gaze trace, velocity profile, and classification.

    Produces a 3-panel figure:
        1. x(t) and y(t) with fixation/saccade shading
        2. velocity over time with the threshold line
        3. 2D scatter of the gaze path coloured by event type
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install with `pip install matplotlib`."
        ) from e

    t = df[time_ms_col].to_numpy()
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    v = df[velocity_col].to_numpy()
    labels = df[label_col].to_numpy()
    is_fix = labels == "fixation"

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

    # --- Panel 1: x(t) and y(t) ---
    ax = axes[0]
    ax.plot(t, x, label="x (px)", linewidth=1)
    ax.plot(t, y, label="y (px)", linewidth=1)
    ax.set_ylabel("Position (px)")
    ax.set_title("Gaze position over time")
    ax.legend(loc="upper right")

    # --- Panel 2: velocity ---
    ax = axes[1]
    ax.plot(t, v, color="steelblue", linewidth=0.8)
    if threshold is not None:
        ax.axhline(threshold, color="crimson", linestyle="--", label=f"threshold = {threshold:g} px/s")
        ax.legend(loc="upper right")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Velocity (px/s)")
    ax.set_title("Point-to-point velocity")

    # --- Panel 3: 2D scatter ---
    ax = axes[2]
    ax.scatter(x[is_fix], y[is_fix], s=6, color="seagreen", label="fixation", alpha=0.7)
    ax.scatter(x[~is_fix], y[~is_fix], s=6, color="crimson", label="saccade", alpha=0.7)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Gaze scatter, classified")
    ax.invert_yaxis()  # screen coords
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right")

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig
