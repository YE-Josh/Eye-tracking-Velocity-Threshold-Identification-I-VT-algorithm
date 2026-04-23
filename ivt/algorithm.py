"""
Velocity Threshold Identification (I-VT) Algorithm
===================================================

Implementation of the I-VT fixation/saccade classifier based on
Salvucci & Goldberg (2000), "Identifying fixations and saccades in
eye-tracking protocols", ETRA 2000.

The algorithm:
    1. Compute point-to-point velocities between consecutive samples.
    2. Label each sample as FIXATION if velocity < threshold, else SACCADE.
    3. Collapse consecutive fixation samples into fixation groups.
    4. Optionally merge nearby fixations and discard short fixations.

Units
-----
Input  : x, y in pixels; time in milliseconds (ms)
Output : velocity in pixels/second; classification per sample
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
@dataclass
class IVTConfig:
    """Configuration for the I-VT classifier.

    Parameters
    ----------
    velocity_threshold : float
        Velocity threshold in pixels/second. Samples with velocity below
        this value are labelled as fixations; samples at or above are
        labelled as saccades. A common default in the literature is
        ~30 deg/s (Salvucci & Goldberg, 2000). In pixel space the
        appropriate threshold depends on your screen geometry and
        viewing distance; 1000 px/s is a reasonable starting point and
        should be tuned for your setup.
    min_fixation_duration_ms : float
        Fixations shorter than this are reclassified as saccades.
        Default 60 ms (physiologically implausible below this).
    merge_adjacent_fixations : bool
        If True, fixations separated by a very short saccade and a
        small spatial gap are merged into one fixation.
    max_merge_gap_ms : float
        Maximum temporal gap between two fixations allowed for merging.
    max_merge_distance_px : float
        Maximum spatial distance between two fixation centroids
        allowed for merging.
    """

    velocity_threshold: float = 1000.0
    min_fixation_duration_ms: float = 60.0
    merge_adjacent_fixations: bool = True
    max_merge_gap_ms: float = 75.0
    max_merge_distance_px: float = 30.0


EventType = Literal["fixation", "saccade"]


# -------------------------------------------------------------------
# Core velocity computation
# -------------------------------------------------------------------
def compute_velocity(x: np.ndarray, y: np.ndarray, t_ms: np.ndarray) -> np.ndarray:
    """Compute point-to-point velocity in pixels/second.

    Velocity at sample i is defined as the Euclidean distance between
    sample i and sample i-1 divided by the time difference. The first
    sample is given velocity 0 so the output aligns with the input
    length.

    Parameters
    ----------
    x, y : np.ndarray
        Gaze coordinates in pixels.
    t_ms : np.ndarray
        Timestamps in milliseconds.

    Returns
    -------
    np.ndarray
        Velocity per sample in pixels/second, same length as inputs.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t_ms = np.asarray(t_ms, dtype=float)

    if not (len(x) == len(y) == len(t_ms)):
        raise ValueError("x, y, and t must have the same length.")
    if len(x) < 2:
        return np.zeros_like(x)

    dx = np.diff(x)
    dy = np.diff(y)
    dt_s = np.diff(t_ms) / 1000.0  # ms -> s

    # Guard against zero or negative dt (duplicates / non-monotonic time).
    dt_s = np.where(dt_s <= 0, np.nan, dt_s)

    dist = np.sqrt(dx * dx + dy * dy)
    v = dist / dt_s

    # Prepend a 0 so output length == input length.
    return np.concatenate(([0.0], v))


# -------------------------------------------------------------------
# Classification
# -------------------------------------------------------------------
def classify_samples(velocity: np.ndarray, threshold: float) -> np.ndarray:
    """Label each sample as 'fixation' or 'saccade'.

    NaN velocities are treated as saccades (unknown / gap).
    """
    labels = np.where(velocity < threshold, "fixation", "saccade")
    labels[np.isnan(velocity)] = "saccade"
    return labels.astype(object)


def _find_runs(labels: np.ndarray, target: str) -> list[tuple[int, int]]:
    """Return [start, end) index pairs for consecutive runs of `target`."""
    runs = []
    in_run = False
    start = 0
    for i, lab in enumerate(labels):
        if lab == target and not in_run:
            in_run = True
            start = i
        elif lab != target and in_run:
            in_run = False
            runs.append((start, i))
    if in_run:
        runs.append((start, len(labels)))
    return runs


def _postprocess(
    labels: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t_ms: np.ndarray,
    config: IVTConfig,
) -> np.ndarray:
    """Apply min-duration filtering and optional fixation merging."""
    labels = labels.copy()

    # ---- 1. Merge adjacent fixations separated by small gaps -------
    if config.merge_adjacent_fixations:
        changed = True
        while changed:
            changed = False
            runs = _find_runs(labels, "fixation")
            for a, b in zip(runs[:-1], runs[1:]):
                gap_start, gap_end = a[1], b[0]  # saccade segment indices
                if gap_start >= gap_end:
                    continue
                gap_ms = t_ms[gap_end - 1] - t_ms[gap_start]
                cx1, cy1 = x[a[0]:a[1]].mean(), y[a[0]:a[1]].mean()
                cx2, cy2 = x[b[0]:b[1]].mean(), y[b[0]:b[1]].mean()
                dist = np.hypot(cx2 - cx1, cy2 - cy1)
                if (
                    gap_ms <= config.max_merge_gap_ms
                    and dist <= config.max_merge_distance_px
                ):
                    labels[gap_start:gap_end] = "fixation"
                    changed = True
                    break  # restart scan after a merge

    # ---- 2. Discard fixations shorter than the minimum duration ----
    for start, end in _find_runs(labels, "fixation"):
        if end - start < 1:
            continue
        duration_ms = t_ms[end - 1] - t_ms[start]
        if duration_ms < config.min_fixation_duration_ms:
            labels[start:end] = "saccade"

    return labels


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def run_ivt(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    time_col: str,
    time_unit: Literal["ms", "s", "us"] = "ms",
    config: IVTConfig | None = None,
) -> pd.DataFrame:
    """Run the I-VT algorithm on an eye-tracking DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing at least the x, y and time columns.
    x_col, y_col, time_col : str
        Column names for the gaze x (px), gaze y (px), and timestamp.
    time_unit : {'ms', 's', 'us'}
        Unit of the timestamp column. Internally all timestamps are
        converted to milliseconds.
    config : IVTConfig, optional
        Algorithm configuration. Defaults used if not provided.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with three added columns:
            - `time_ms`        : timestamp in ms (converted if needed)
            - `velocity_pxs`   : point-to-point velocity (px/s)
            - `eye_movement`   : 'fixation' or 'saccade' per sample
    """
    if config is None:
        config = IVTConfig()

    for col in (x_col, y_col, time_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    out = df.copy()

    # --- Time conversion to ms ---
    t_raw = out[time_col].to_numpy(dtype=float)
    if time_unit == "ms":
        t_ms = t_raw
    elif time_unit == "s":
        t_ms = t_raw * 1000.0
    elif time_unit == "us":
        t_ms = t_raw / 1000.0
    else:
        raise ValueError(f"Unsupported time_unit '{time_unit}'.")

    x = out[x_col].to_numpy(dtype=float)
    y = out[y_col].to_numpy(dtype=float)

    # --- Velocity + classification ---
    v = compute_velocity(x, y, t_ms)
    labels = classify_samples(v, config.velocity_threshold)
    labels = _postprocess(labels, x, y, t_ms, config)

    out["time_ms"] = t_ms
    out["velocity_pxs"] = v
    out["eye_movement"] = labels
    return out


def summarize_events(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    time_ms_col: str = "time_ms",
    label_col: str = "eye_movement",
) -> pd.DataFrame:
    """Collapse per-sample labels into an event-level summary.

    Returns one row per fixation/saccade with start/end time,
    duration, sample count, and (for fixations) centroid coordinates.
    """
    labels = df[label_col].to_numpy()
    t = df[time_ms_col].to_numpy(dtype=float)
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    events = []
    event_id = 0
    current = labels[0]
    start_idx = 0

    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != current:
            end_idx = i
            seg_x = x[start_idx:end_idx]
            seg_y = y[start_idx:end_idx]
            events.append(
                {
                    "event_id": event_id,
                    "event_type": current,
                    "start_ms": t[start_idx],
                    "end_ms": t[end_idx - 1],
                    "duration_ms": t[end_idx - 1] - t[start_idx],
                    "n_samples": end_idx - start_idx,
                    "centroid_x": float(np.nanmean(seg_x)),
                    "centroid_y": float(np.nanmean(seg_y)),
                }
            )
            event_id += 1
            if i < len(labels):
                current = labels[i]
                start_idx = i

    return pd.DataFrame(events)
