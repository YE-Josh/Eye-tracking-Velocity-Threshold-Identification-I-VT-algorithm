"""
Visualization helpers for I-VT results.

Provides publication-quality figures showing:
  - Gaze x/y traces over time with fixation/saccade shading
  - Scatter of fixations sized by duration
  - Velocity trace with threshold overlay
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_ivt_result(
    result,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    dpi: int = 300,
    figsize: tuple = (12, 9),
):
    """Produce a 3-panel summary figure of an IVT run.

    Panels
    ------
    (a) Gaze x,y vs. time, with fixation intervals shaded.
    (b) Velocity vs. time, with the threshold drawn.
    (c) 2-D scatter of the raw samples coloured by class, plus fixation
        centroids sized by duration.

    Parameters
    ----------
    result : IVTResult
    save_path : path-like, optional
        If given, save the figure (PNG by default; extension respected).
    show : bool
        Whether to call plt.show().
    dpi : int
        DPI for saved figure.
    figsize : tuple
        Figure size in inches.
    """
    data = result.data
    fixations = result.fixations
    config = result.config

    t = data["time_ms"].to_numpy()
    x = data.iloc[:, 0].to_numpy() if "x" not in data.columns else data["x"].to_numpy()
    # Use the first two numeric columns we originally set? Safer: infer from fixations.
    # To keep this robust, require the caller's x/y to be present as-is.
    # We'll instead read back from fixations centroids for the 2D panel.
    velocity = data["velocity"].to_numpy()
    labels = data["eye_movement_type"].to_numpy()

    # Identify x/y columns as the two numeric columns that aren't time/velocity/label
    reserved = {"time_ms", "velocity", "eye_movement_type"}
    numeric_cols = [
        c for c in data.columns
        if c not in reserved and np.issubdtype(data[c].dtype, np.number)
    ]
    if len(numeric_cols) < 2:
        raise ValueError("Could not identify x/y columns for plotting.")
    x_col, y_col = numeric_cols[0], numeric_cols[1]
    x = data[x_col].to_numpy()
    y = data[y_col].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # (a) x,y vs time --------------------------------------------------------
    ax = axes[0]
    ax.plot(t, x, label=x_col, linewidth=0.8)
    ax.plot(t, y, label=y_col, linewidth=0.8)
    for _, row in fixations.iterrows():
        ax.axvspan(row["start_time_ms"], row["end_time_ms"],
                   alpha=0.15, color="tab:green")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Gaze position (px)")
    ax.set_title("Gaze position over time (green = fixations)")
    ax.legend(loc="upper right", fontsize=9)

    # (b) velocity vs time ---------------------------------------------------
    ax = axes[1]
    ax.plot(t, velocity, linewidth=0.8, color="black")
    ax.axhline(config.velocity_threshold, color="red", linestyle="--",
               linewidth=1,
               label=f"Threshold = {config.velocity_threshold} {config.velocity_unit}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(f"Velocity ({config.velocity_unit})")
    ax.set_title("Gaze velocity")
    ax.legend(loc="upper right", fontsize=9)

    # (c) 2-D scatter --------------------------------------------------------
    ax = axes[2]
    fix_mask = labels == "fixation"
    ax.scatter(x[~fix_mask], y[~fix_mask], s=4, c="tab:red",
               label="Saccade samples", alpha=0.5)
    ax.scatter(x[fix_mask], y[fix_mask], s=4, c="tab:blue",
               label="Fixation samples", alpha=0.5)
    if len(fixations) > 0:
        sizes = np.clip(fixations["duration_ms"].to_numpy() / 5.0, 20, 400)
        ax.scatter(
            fixations["centroid_x"], fixations["centroid_y"],
            s=sizes, facecolors="none", edgecolors="tab:green",
            linewidths=1.5, label="Fixation centroids (size ∝ duration)",
        )
    ax.set_xlabel(f"{x_col} (px)")
    ax.set_ylabel(f"{y_col} (px)")
    ax.set_title("2-D gaze distribution")
    ax.legend(loc="upper right", fontsize=9)
    ax.invert_yaxis()  # Screen coordinate convention

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to {save_path.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
