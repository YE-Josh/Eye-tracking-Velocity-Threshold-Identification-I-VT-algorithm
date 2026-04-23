"""
Velocity Threshold Identification (I-VT) Algorithm
==================================================

Implementation of the I-VT fixation detection algorithm based on:
Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and saccades
in eye-tracking protocols. In Proceedings of the 2000 symposium on Eye
tracking research & applications (pp. 71-78).

The algorithm classifies gaze samples as either 'fixation' or 'saccade'
based on point-to-point velocity. Consecutive fixation samples shorter
than a minimum duration are discarded (reclassified as noise/saccade).

Author: Generated for Joash Ye's research toolkit
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# -----------------------------------------------------------------------------
# Default parameters (literature-based)
# -----------------------------------------------------------------------------
DEFAULT_VELOCITY_THRESHOLD_DEG = 30.0   # deg/s  (Olsen, 2012; Tobii I-VT default)
DEFAULT_VELOCITY_THRESHOLD_PIX = 1000.0  # pixels/s  (fallback when deg/s not applicable)
DEFAULT_MIN_FIXATION_DURATION_MS = 60.0  # ms  (Salvucci & Goldberg, 2000)


@dataclass
class IVTConfig:
    """Configuration for the I-VT algorithm.

    Attributes
    ----------
    velocity_threshold : float
        Velocity threshold separating fixations from saccades.
        Units match `velocity_unit` ('pixels/s' or 'deg/s').
    min_fixation_duration_ms : float
        Minimum duration (ms) for a sequence of fixation samples to be
        retained as a valid fixation. Shorter runs are reclassified.
    velocity_unit : str
        Either 'pixels/s' or 'deg/s'. If 'deg/s', `pixels_per_degree`
        must be provided.
    pixels_per_degree : Optional[float]
        Conversion factor. Required when `velocity_unit == 'deg/s'`.
    """

    velocity_threshold: float = DEFAULT_VELOCITY_THRESHOLD_PIX
    min_fixation_duration_ms: float = DEFAULT_MIN_FIXATION_DURATION_MS
    velocity_unit: str = "pixels/s"
    pixels_per_degree: Optional[float] = None

    def __post_init__(self):
        if self.velocity_unit not in {"pixels/s", "deg/s"}:
            raise ValueError("velocity_unit must be 'pixels/s' or 'deg/s'.")
        if self.velocity_unit == "deg/s" and not self.pixels_per_degree:
            raise ValueError(
                "pixels_per_degree must be provided when velocity_unit='deg/s'."
            )
        if self.velocity_threshold <= 0:
            raise ValueError("velocity_threshold must be > 0.")
        if self.min_fixation_duration_ms < 0:
            raise ValueError("min_fixation_duration_ms must be >= 0.")


@dataclass
class IVTResult:
    """Container for I-VT output.

    Attributes
    ----------
    data : pd.DataFrame
        Original samples augmented with 'velocity' and 'eye_movement_type'.
    fixations : pd.DataFrame
        Summary of detected fixations (start, end, duration, centroid).
    saccades : pd.DataFrame
        Summary of detected saccades (start, end, duration, amplitude).
    config : IVTConfig
        The configuration used for this run.
    """

    data: pd.DataFrame
    fixations: pd.DataFrame
    saccades: pd.DataFrame
    config: IVTConfig = field(repr=False)


# -----------------------------------------------------------------------------
# Core algorithm
# -----------------------------------------------------------------------------
def compute_velocity(
    x: np.ndarray, y: np.ndarray, t_ms: np.ndarray
) -> np.ndarray:
    """Compute point-to-point velocity (units/second).

    Uses centered differences internally, falling back to forward/backward
    differences at the endpoints. This is more robust to noise than a pure
    forward difference and is common practice in eye-tracking pipelines.

    Parameters
    ----------
    x, y : np.ndarray
        Gaze coordinates (same units, e.g., pixels).
    t_ms : np.ndarray
        Timestamps in milliseconds. Must be monotonically increasing.

    Returns
    -------
    velocity : np.ndarray
        Velocity in (coord-units)/second, same length as inputs.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t_ms = np.asarray(t_ms, dtype=float)

    n = len(x)
    if n < 2:
        return np.zeros(n)

    # Convert ms -> s for velocity in units/s
    t_s = t_ms / 1000.0

    dx = np.zeros(n)
    dy = np.zeros(n)
    dt = np.zeros(n)

    # Centered differences for interior points
    dx[1:-1] = x[2:] - x[:-2]
    dy[1:-1] = y[2:] - y[:-2]
    dt[1:-1] = t_s[2:] - t_s[:-2]

    # Endpoints: forward/backward differences
    dx[0] = x[1] - x[0]
    dy[0] = y[1] - y[0]
    dt[0] = t_s[1] - t_s[0]

    dx[-1] = x[-1] - x[-2]
    dy[-1] = y[-1] - y[-2]
    dt[-1] = t_s[-1] - t_s[-2]

    # Avoid division by zero
    dt = np.where(dt == 0, np.nan, dt)
    distance = np.sqrt(dx ** 2 + dy ** 2)
    velocity = distance / dt

    # NaN velocities (from dt==0) treated as 0
    velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
    return velocity


def classify_samples(velocity: np.ndarray, threshold: float) -> np.ndarray:
    """Label each sample as 'fixation' or 'saccade' based on velocity.

    Parameters
    ----------
    velocity : np.ndarray
        Per-sample velocity.
    threshold : float
        Velocity threshold. Samples below threshold are 'fixation'.

    Returns
    -------
    labels : np.ndarray of dtype object
        Array of 'fixation' / 'saccade' strings.
    """
    labels = np.where(velocity < threshold, "fixation", "saccade")
    return labels.astype(object)


def apply_minimum_fixation_duration(
    labels: np.ndarray, t_ms: np.ndarray, min_duration_ms: float
) -> np.ndarray:
    """Reclassify short fixation runs as saccades.

    Scans contiguous runs of 'fixation' labels; if the run's duration
    (last-sample-time minus first-sample-time) is below `min_duration_ms`,
    all samples in that run are relabeled as 'saccade'.

    Parameters
    ----------
    labels : np.ndarray
        Initial per-sample classifications.
    t_ms : np.ndarray
        Timestamps in milliseconds.
    min_duration_ms : float
        Minimum fixation duration threshold.

    Returns
    -------
    labels : np.ndarray
        Cleaned classifications.
    """
    labels = labels.copy()
    if min_duration_ms <= 0 or len(labels) == 0:
        return labels

    n = len(labels)
    i = 0
    while i < n:
        if labels[i] == "fixation":
            j = i
            while j < n and labels[j] == "fixation":
                j += 1
            # Run is [i, j)
            if j - i >= 1:
                duration = t_ms[j - 1] - t_ms[i]
                if duration < min_duration_ms:
                    labels[i:j] = "saccade"
            i = j
        else:
            i += 1
    return labels


def _summarize_events(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    time_col: str,
    label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build fixation and saccade summary tables from labelled samples."""
    fixations = []
    saccades = []

    labels = df[label_col].to_numpy()
    t = df[time_col].to_numpy()
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    n = len(df)
    i = 0
    fix_id = 0
    sac_id = 0
    while i < n:
        current = labels[i]
        j = i
        while j < n and labels[j] == current:
            j += 1
        # Event spans [i, j)
        start_t = t[i]
        end_t = t[j - 1]
        duration = end_t - start_t
        xs = x[i:j]
        ys = y[i:j]

        if current == "fixation":
            fix_id += 1
            fixations.append(
                {
                    "fixation_id": fix_id,
                    "start_index": i,
                    "end_index": j - 1,
                    "start_time_ms": start_t,
                    "end_time_ms": end_t,
                    "duration_ms": duration,
                    "n_samples": j - i,
                    "centroid_x": float(np.mean(xs)),
                    "centroid_y": float(np.mean(ys)),
                    "dispersion_x": float(np.std(xs)),
                    "dispersion_y": float(np.std(ys)),
                }
            )
        else:
            sac_id += 1
            amplitude = float(
                np.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2)
            )
            saccades.append(
                {
                    "saccade_id": sac_id,
                    "start_index": i,
                    "end_index": j - 1,
                    "start_time_ms": start_t,
                    "end_time_ms": end_t,
                    "duration_ms": duration,
                    "n_samples": j - i,
                    "start_x": float(xs[0]),
                    "start_y": float(ys[0]),
                    "end_x": float(xs[-1]),
                    "end_y": float(ys[-1]),
                    "amplitude_pixels": amplitude,
                }
            )
        i = j

    return (
        pd.DataFrame(fixations),
        pd.DataFrame(saccades),
    )


def run_ivt(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    time_col: str,
    config: Optional[IVTConfig] = None,
    time_unit: str = "ms",
) -> IVTResult:
    """Run the I-VT algorithm on a DataFrame of gaze samples.

    Parameters
    ----------
    df : pd.DataFrame
        Input samples.
    x_col, y_col : str
        Column names for gaze X and Y (in pixels).
    time_col : str
        Column name for timestamps.
    config : IVTConfig, optional
        Algorithm configuration. Defaults are used if None.
    time_unit : str
        Unit of the timestamp column. One of: 's', 'ms', 'us', 'ns'.
        Timestamps are converted to milliseconds internally.

    Returns
    -------
    IVTResult
        Augmented data plus fixation/saccade summaries.
    """
    if config is None:
        config = IVTConfig()

    # --- Validate columns ---------------------------------------------------
    for col in (x_col, y_col, time_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    # --- Copy and clean -----------------------------------------------------
    data = df.copy().reset_index(drop=True)

    # Drop rows with NaNs in the key columns (I-VT cannot handle gaps natively)
    n_before = len(data)
    data = data.dropna(subset=[x_col, y_col, time_col]).reset_index(drop=True)
    n_dropped = n_before - len(data)
    if n_dropped > 0:
        print(f"[I-VT] Dropped {n_dropped} samples with NaN in x/y/time.")

    if len(data) < 2:
        raise ValueError("Not enough valid samples to run I-VT (need >= 2).")

    # --- Convert time to ms -------------------------------------------------
    time_scale = {"s": 1000.0, "ms": 1.0, "us": 1e-3, "ns": 1e-6}
    if time_unit not in time_scale:
        raise ValueError(
            f"time_unit must be one of {list(time_scale.keys())}, got '{time_unit}'."
        )
    t_ms = data[time_col].to_numpy(dtype=float) * time_scale[time_unit]

    # Ensure monotonically increasing time
    if np.any(np.diff(t_ms) < 0):
        print("[I-VT] Warning: timestamps not monotonic. Sorting by time.")
        order = np.argsort(t_ms)
        data = data.iloc[order].reset_index(drop=True)
        t_ms = t_ms[order]

    data["time_ms"] = t_ms

    # --- Compute velocity ---------------------------------------------------
    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)
    velocity_pix_s = compute_velocity(x, y, t_ms)

    if config.velocity_unit == "deg/s":
        velocity = velocity_pix_s / config.pixels_per_degree
    else:
        velocity = velocity_pix_s

    data["velocity"] = velocity

    # --- Classify -----------------------------------------------------------
    labels = classify_samples(velocity, config.velocity_threshold)
    labels = apply_minimum_fixation_duration(
        labels, t_ms, config.min_fixation_duration_ms
    )
    data["eye_movement_type"] = labels

    # --- Summarize events ---------------------------------------------------
    fixations_df, saccades_df = _summarize_events(
        data, x_col, y_col, "time_ms", "eye_movement_type"
    )

    return IVTResult(
        data=data,
        fixations=fixations_df,
        saccades=saccades_df,
        config=config,
    )
