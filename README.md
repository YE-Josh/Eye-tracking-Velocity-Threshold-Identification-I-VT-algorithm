# Velocity Threshold Identification (I-VT) Algorithm

A Python implementation of the **I-VT algorithm** (Salvucci & Goldberg, 2000) for classifying 2-D eye-movement data into **fixations** and **saccades**.

The tool reads gaze samples from a CSV or Excel file, lets you map your X / Y / time columns interactively, converts timestamps to milliseconds, computes point-to-point velocity, applies a velocity threshold and a minimum-fixation-duration filter, and writes classified samples together with per-event summary tables back to disk.

---

## Features

- **Flexible input** — CSV, TSV, `.xlsx`, or `.xls`; automatic Excel sheet selection
- **Interactive column picker** — choose X, Y, and timestamp columns from a menu (no need to rename your data)
- **Unit handling** — timestamps in s, ms, µs, or ns; velocity threshold in pixels/s **or** degrees/s (with configurable pixels-per-degree)
- **Configurable parameters** with sensible research defaults
  - Velocity threshold: `1000 pixels/s` (or `30 °/s` — Tobii I-VT default, Olsen 2012)
  - Minimum fixation duration: `60 ms` (Salvucci & Goldberg 2000)
- **Three output tables** per run
  - `*_samples.csv` — every input sample plus `velocity` and `eye_movement_type`
  - `*_fixations.csv` — ID, start/end time, duration, centroid (x, y), dispersion
  - `*_saccades.csv` — ID, start/end time, duration, amplitude, start/end position
- **Publication-style summary figure** (x/y trace, velocity trace with threshold, 2-D scatter with fixation centroids)
- **Programmatic API** for pipeline integration and custom analyses
- **Unit-tested** with 16 passing tests covering velocity computation, classification, and end-to-end behaviour

---

## Installation

```bash
git clone https://github.com/<your-username>/ivt-algorithm.git
cd ivt-algorithm
pip install -r requirements.txt
pip install -e .          # optional: install as an editable package
```

Requires Python >= 3.9. Core dependencies: `numpy`, `pandas`, `matplotlib`, `openpyxl`.

---

## Quick start

### 1. Interactive mode (recommended for first-time users)

```bash
python -m ivt.cli
```

You'll be guided step-by-step:

```
============================================================
  I-VT (Velocity Threshold Identification) -- Interactive
============================================================
Path to CSV / Excel file: my_gaze_data.xlsx

Loaded file with 5400 rows and 6 columns.
First 5 rows: ...

Select X-coordinate column (pixels):
  [0] participant_id
  [1] trial
  [2] gaze_x        <-- choose index 2
  [3] gaze_y
  [4] timestamp
  [5] pupil_diameter
Select index [0-5]: 2

...

Velocity threshold (pixels/s) [1000.0]:
Minimum fixation duration (ms) [60.0]:
Output directory [./ivt_output]:

Running I-VT ...
```

### 2. Non-interactive mode (for scripting / batch processing)

```bash
python -m ivt.cli \
  --input data/participant01.csv \
  --x-col gaze_x --y-col gaze_y --time-col timestamp \
  --time-unit ms \
  --velocity-threshold 1000 \
  --min-fixation-duration 60 \
  --output-dir results/participant01
```

### 3. Python API

```python
import pandas as pd
from ivt import IVTConfig, run_ivt, write_ivt_outputs
from ivt.visualization import plot_ivt_result

df = pd.read_csv("gaze_data.csv")

config = IVTConfig(
    velocity_threshold=1000.0,        # pixels/s
    min_fixation_duration_ms=60.0,
    velocity_unit="pixels/s",
)

result = run_ivt(
    df,
    x_col="gaze_x",
    y_col="gaze_y",
    time_col="timestamp_ms",
    config=config,
    time_unit="ms",
)

print(result.fixations.head())
print(result.saccades.head())

write_ivt_outputs(result, "output/", basename="participant01")
plot_ivt_result(result, save_path="output/participant01_summary.png")
```

---

## Algorithm

For each gaze sample *i*:

1. **Velocity**: centered-difference point-to-point velocity
   `v_i = sqrt((x_{i+1} - x_{i-1})^2 + (y_{i+1} - y_{i-1})^2) / (t_{i+1} - t_{i-1})`
   (forward/backward difference at the endpoints)

2. **Initial classification** — `v_i < threshold` -> `fixation`, else `saccade`

3. **Minimum-duration filter** — any contiguous run of `fixation` samples whose total duration is less than `min_fixation_duration_ms` is relabelled as `saccade`

4. **Event summarisation** — contiguous same-label runs are collapsed into fixation / saccade events with their temporal, spatial, and dispersion properties

---

## Parameter selection guide

| Context | Recommended threshold | Reference |
|---|---|---|
| Tobii Pro desktop trackers | 30 °/s | Olsen (2012) |
| EyeLink / high-freq (>=500 Hz), lab | 30-50 °/s | Salvucci & Goldberg (2000) |
| Head-mounted / mobile eye tracking (noisier) | 100 °/s | Munn et al. (2008) |
| Screen-coordinate data, no deg conversion | 800-1200 pixels/s | Depends on viewing geometry |

`min_fixation_duration_ms` is typically in the **60-100 ms** range; 60 ms is the classical default.

If you are working in degrees of visual angle, supply `--pixels-per-degree` (computed from your screen width, resolution, and viewing distance).

---

## Output file schema

**`*_samples.csv`** — one row per input sample

| column | description |
|---|---|
| (original columns) | preserved as-is |
| `time_ms` | timestamp normalised to ms |
| `velocity` | per-sample velocity in `velocity_unit` |
| `eye_movement_type` | `fixation` or `saccade` |

**`*_fixations.csv`** — one row per fixation
`fixation_id, start_index, end_index, start_time_ms, end_time_ms, duration_ms, n_samples, centroid_x, centroid_y, dispersion_x, dispersion_y`

**`*_saccades.csv`** — one row per saccade
`saccade_id, start_index, end_index, start_time_ms, end_time_ms, duration_ms, n_samples, start_x, start_y, end_x, end_y, amplitude_pixels`

---

## Project structure

```
ivt-algorithm/
├── ivt/
│   ├── __init__.py
│   ├── ivt_algorithm.py    # core algorithm
│   ├── io_utils.py         # file reading / writing
│   ├── cli.py              # interactive + non-interactive CLI
│   └── visualization.py    # plotting helpers
├── examples/
│   └── run_example.py      # end-to-end synthetic demo
├── tests/
│   └── test_ivt.py         # 16 unit tests
├── requirements.txt
├── setup.py
├── LICENSE                 # MIT
└── README.md
```

---

## Running the tests

```bash
pytest tests/ -v
```

---

## Running the bundled example

```bash
PYTHONPATH=. python examples/run_example.py
```

This generates a synthetic gaze trace with 4 fixations separated by fast saccades and writes `examples/output/demo_summary.png` together with the three CSV outputs.

---

## References

- Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and saccades in eye-tracking protocols. In *Proceedings of the 2000 Symposium on Eye Tracking Research & Applications* (pp. 71–78). ACM.
- Olsen, A. (2012). *The Tobii I-VT Fixation Filter: Algorithm description*. Tobii Technology.
- Holmqvist, K., Nyström, M., Andersson, R., Dewhurst, R., Jarodzka, H., & van de Weijer, J. (2011). *Eye tracking: A comprehensive guide to methods and measures*. Oxford University Press.

---

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

If you use this implementation in your research, please cite:

```
@software{ivt_algorithm_2026,
  author  = {Ye, Joash},
  title   = {Velocity Threshold Identification (I-VT) Algorithm},
  year    = {2026},
  url     = {https://github.com/<your-username>/ivt-algorithm}
}
```
