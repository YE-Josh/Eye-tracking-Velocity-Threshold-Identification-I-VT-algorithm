"""Tests for the I-VT algorithm."""

import numpy as np
import pandas as pd
import pytest

from ivt import IVTConfig, compute_velocity, run_ivt, summarize_events


# -------------------------------------------------------------------
# compute_velocity
# -------------------------------------------------------------------
def test_velocity_constant_motion():
    # 10 px every 10 ms => 1000 px/s
    n = 10
    t = np.arange(n) * 10.0  # ms
    x = np.arange(n) * 10.0
    y = np.zeros(n)
    v = compute_velocity(x, y, t)
    assert v[0] == 0.0
    assert np.allclose(v[1:], 1000.0)


def test_velocity_stationary():
    t = np.arange(10) * 4.0
    x = np.full(10, 500.0)
    y = np.full(10, 500.0)
    v = compute_velocity(x, y, t)
    assert np.allclose(v, 0.0)


def test_velocity_length_mismatch():
    with pytest.raises(ValueError):
        compute_velocity(np.array([0, 1]), np.array([0, 1, 2]), np.array([0, 4, 8]))


# -------------------------------------------------------------------
# run_ivt
# -------------------------------------------------------------------
def _make_df():
    # 20 stationary samples (fixation), 5 fast samples (saccade),
    # 20 stationary samples (fixation).
    t_step = 4.0  # ms (250 Hz)
    rows = []
    t = 0.0
    # fixation 1 at (100, 100)
    for _ in range(20):
        rows.append((t, 100.0, 100.0))
        t += t_step
    # saccade: jump 40 px per sample => 10,000 px/s
    for i in range(1, 6):
        rows.append((t, 100.0 + 40 * i, 100.0))
        t += t_step
    # fixation 2 at (300, 100)
    for _ in range(20):
        rows.append((t, 300.0, 100.0))
        t += t_step
    return pd.DataFrame(rows, columns=["t", "x", "y"])


def test_run_ivt_labels_and_columns():
    df = _make_df()
    out = run_ivt(df, x_col="x", y_col="y", time_col="t", time_unit="ms")
    assert {"time_ms", "velocity_pxs", "eye_movement"}.issubset(out.columns)
    assert set(out["eye_movement"].unique()).issubset({"fixation", "saccade"})
    # Majority of samples should be fixations.
    assert (out["eye_movement"] == "fixation").sum() > (out["eye_movement"] == "saccade").sum()


def test_time_unit_conversion():
    df = _make_df().rename(columns={"t": "t_s"})
    df["t_s"] = df["t_s"] / 1000.0  # seconds
    out = run_ivt(df, x_col="x", y_col="y", time_col="t_s", time_unit="s")
    # Velocities should be identical to the ms version.
    df_ms = _make_df()
    out_ms = run_ivt(df_ms, x_col="x", y_col="y", time_col="t", time_unit="ms")
    np.testing.assert_allclose(out["velocity_pxs"].values, out_ms["velocity_pxs"].values)


def test_missing_column_raises():
    df = _make_df()
    with pytest.raises(KeyError):
        run_ivt(df, x_col="nope", y_col="y", time_col="t")


def test_min_fixation_filter():
    # Short fixation between two saccades should be discarded.
    df = _make_df()
    cfg = IVTConfig(velocity_threshold=1000.0, min_fixation_duration_ms=1000.0,
                    merge_adjacent_fixations=False)
    out = run_ivt(df, x_col="x", y_col="y", time_col="t", config=cfg)
    # With a 1-second minimum, neither fixation is long enough (each ~80 ms).
    assert (out["eye_movement"] == "fixation").sum() == 0


def test_summarize_events_structure():
    df = _make_df()
    out = run_ivt(df, x_col="x", y_col="y", time_col="t")
    events = summarize_events(out, x_col="x", y_col="y")
    assert {"event_type", "start_ms", "end_ms", "duration_ms",
            "n_samples", "centroid_x", "centroid_y"}.issubset(events.columns)
    # At least one fixation expected.
    assert (events["event_type"] == "fixation").any()
