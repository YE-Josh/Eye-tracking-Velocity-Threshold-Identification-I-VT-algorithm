"""
Unit tests for the I-VT algorithm.

Run with:
    pytest tests/
"""

import numpy as np
import pandas as pd
import pytest

from ivt import (
    IVTConfig,
    compute_velocity,
    classify_samples,
    apply_minimum_fixation_duration,
    run_ivt,
)


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------
def make_synthetic_trace(
    n_fixations: int = 3,
    samples_per_fixation: int = 60,   # 60 samples @ 100 Hz = 600 ms
    sampling_rate_hz: int = 100,
    saccade_length_samples: int = 5,
    fixation_positions: list = None,
    noise_std: float = 1.0,
    random_state: int = 0,
) -> pd.DataFrame:
    """Generate a synthetic gaze trace consisting of fixations separated
    by fast saccades. Returns a DataFrame with columns x, y, t_ms."""
    rng = np.random.default_rng(random_state)
    dt_ms = 1000.0 / sampling_rate_hz

    if fixation_positions is None:
        fixation_positions = [(100 + 200 * i, 200 + 50 * i)
                              for i in range(n_fixations)]
    assert len(fixation_positions) == n_fixations

    xs, ys, ts = [], [], []
    t = 0.0
    for i, (fx, fy) in enumerate(fixation_positions):
        # Fixation
        for _ in range(samples_per_fixation):
            xs.append(fx + rng.normal(0, noise_std))
            ys.append(fy + rng.normal(0, noise_std))
            ts.append(t)
            t += dt_ms
        # Saccade to next fixation (linear interp)
        if i < n_fixations - 1:
            nx, ny = fixation_positions[i + 1]
            for s in range(saccade_length_samples):
                frac = (s + 1) / (saccade_length_samples + 1)
                xs.append(fx + frac * (nx - fx))
                ys.append(fy + frac * (ny - fy))
                ts.append(t)
                t += dt_ms

    return pd.DataFrame({"x": xs, "y": ys, "t_ms": ts})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestCoreFunctions:
    def test_compute_velocity_linear_motion(self):
        # Move 10 px every 10 ms -> 1000 px/s
        x = np.arange(0, 100, 10, dtype=float)
        y = np.zeros_like(x)
        t = np.arange(0, 100, 10, dtype=float)
        v = compute_velocity(x, y, t)
        # Interior points should be ~1000 px/s
        assert np.allclose(v[1:-1], 1000.0, rtol=1e-6)

    def test_compute_velocity_stationary(self):
        x = np.full(10, 100.0)
        y = np.full(10, 100.0)
        t = np.arange(0, 100, 10, dtype=float)
        v = compute_velocity(x, y, t)
        assert np.allclose(v, 0.0)

    def test_compute_velocity_short_input(self):
        assert len(compute_velocity(np.array([]), np.array([]), np.array([]))) == 0
        assert len(compute_velocity(np.array([1.0]), np.array([1.0]),
                                    np.array([0.0]))) == 1

    def test_classify_samples(self):
        v = np.array([10.0, 500.0, 20.0, 2000.0])
        labels = classify_samples(v, threshold=100.0)
        assert labels.tolist() == ["fixation", "saccade", "fixation", "saccade"]

    def test_apply_minimum_fixation_duration_removes_short(self):
        labels = np.array(["fixation", "fixation", "saccade",
                           "fixation", "fixation", "fixation"], dtype=object)
        # Samples at 0,10,20,30,40,50 ms
        t = np.array([0, 10, 20, 30, 40, 50], dtype=float)
        out = apply_minimum_fixation_duration(labels, t, min_duration_ms=20.0)
        # First run: 0->10 ms = 10 ms duration -> below threshold -> saccade
        # Second run: 30->50 ms = 20 ms duration -> kept
        assert out[0] == "saccade"
        assert out[1] == "saccade"
        assert out[3] == "fixation"

    def test_apply_minimum_fixation_duration_zero_threshold(self):
        labels = np.array(["fixation", "saccade"], dtype=object)
        t = np.array([0, 10], dtype=float)
        out = apply_minimum_fixation_duration(labels, t, min_duration_ms=0)
        assert out.tolist() == ["fixation", "saccade"]


class TestConfig:
    def test_default_config_valid(self):
        c = IVTConfig()
        assert c.velocity_threshold > 0
        assert c.velocity_unit == "pixels/s"

    def test_deg_unit_requires_ppd(self):
        with pytest.raises(ValueError):
            IVTConfig(velocity_unit="deg/s")

    def test_deg_unit_with_ppd(self):
        c = IVTConfig(velocity_unit="deg/s", pixels_per_degree=35.0)
        assert c.pixels_per_degree == 35.0

    def test_invalid_unit(self):
        with pytest.raises(ValueError):
            IVTConfig(velocity_unit="rad/s")

    def test_negative_threshold(self):
        with pytest.raises(ValueError):
            IVTConfig(velocity_threshold=-1)


class TestRunIVT:
    def test_detects_expected_fixation_count(self):
        df = make_synthetic_trace(n_fixations=3, samples_per_fixation=60)
        config = IVTConfig(
            velocity_threshold=1000.0,
            min_fixation_duration_ms=60.0,
        )
        result = run_ivt(df, "x", "y", "t_ms", config=config, time_unit="ms")
        # We built 3 fixations; algorithm should find 3
        assert len(result.fixations) == 3
        assert len(result.saccades) >= 2  # at least the inter-fixation saccades

    def test_missing_column_raises(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "t": [0, 10]})
        with pytest.raises(KeyError):
            run_ivt(df, "x", "y", "time_missing")

    def test_time_unit_conversion(self):
        # Time in seconds should produce same result as in ms (after conversion)
        df = make_synthetic_trace(n_fixations=2, samples_per_fixation=60)
        df_s = df.copy()
        df_s["t_ms"] = df_s["t_ms"] / 1000.0  # now in seconds

        config = IVTConfig(velocity_threshold=1000.0,
                           min_fixation_duration_ms=60.0)
        r_ms = run_ivt(df, "x", "y", "t_ms", config=config, time_unit="ms")
        r_s = run_ivt(df_s, "x", "y", "t_ms", config=config, time_unit="s")
        assert len(r_ms.fixations) == len(r_s.fixations)
        assert len(r_ms.saccades) == len(r_s.saccades)

    def test_handles_nans(self):
        df = make_synthetic_trace(n_fixations=2, samples_per_fixation=60)
        df.loc[5:10, "x"] = np.nan
        config = IVTConfig(velocity_threshold=1000.0,
                           min_fixation_duration_ms=60.0)
        result = run_ivt(df, "x", "y", "t_ms", config=config, time_unit="ms")
        assert len(result.data) == len(df) - 6  # 6 samples dropped

    def test_fixation_summary_columns(self):
        df = make_synthetic_trace(n_fixations=2, samples_per_fixation=60)
        config = IVTConfig(velocity_threshold=1000.0,
                           min_fixation_duration_ms=60.0)
        result = run_ivt(df, "x", "y", "t_ms", config=config)
        expected_cols = {
            "fixation_id", "start_time_ms", "end_time_ms",
            "duration_ms", "centroid_x", "centroid_y",
        }
        assert expected_cols.issubset(result.fixations.columns)
