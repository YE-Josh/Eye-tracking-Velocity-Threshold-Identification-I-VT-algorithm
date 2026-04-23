"""
I-VT: Velocity Threshold Identification for eye-tracking data.
"""

from .ivt_algorithm import (
    IVTConfig,
    IVTResult,
    compute_velocity,
    classify_samples,
    apply_minimum_fixation_duration,
    run_ivt,
    DEFAULT_VELOCITY_THRESHOLD_DEG,
    DEFAULT_VELOCITY_THRESHOLD_PIX,
    DEFAULT_MIN_FIXATION_DURATION_MS,
)
from .io_utils import read_gaze_file, list_excel_sheets, write_ivt_outputs

__all__ = [
    "IVTConfig",
    "IVTResult",
    "compute_velocity",
    "classify_samples",
    "apply_minimum_fixation_duration",
    "run_ivt",
    "read_gaze_file",
    "list_excel_sheets",
    "write_ivt_outputs",
    "DEFAULT_VELOCITY_THRESHOLD_DEG",
    "DEFAULT_VELOCITY_THRESHOLD_PIX",
    "DEFAULT_MIN_FIXATION_DURATION_MS",
]

__version__ = "0.1.0"
