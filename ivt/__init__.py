"""I-VT (Velocity Threshold Identification) eye-movement classifier."""

from .algorithm import (
    IVTConfig,
    classify_samples,
    compute_velocity,
    run_ivt,
    summarize_events,
)
from .io import read_gaze_file, write_gaze_file

__all__ = [
    "IVTConfig",
    "classify_samples",
    "compute_velocity",
    "run_ivt",
    "summarize_events",
    "read_gaze_file",
    "write_gaze_file",
]

__version__ = "0.1.0"
