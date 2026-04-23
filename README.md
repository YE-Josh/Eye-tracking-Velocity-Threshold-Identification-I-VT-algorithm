# I-VT: Velocity Threshold Identification for Eye-Movement Data

A small, well-tested Python implementation of the **I-VT (Velocity Threshold Identification)** algorithm described by Salvucci & Goldberg (2000) for classifying 2D gaze samples into **fixations** and **saccades**.

The tool reads eye-tracking data from a **CSV or Excel** file, asks the user which columns contain the `x`, `y` (pixels) and `time` values, converts time to milliseconds, computes point-to-point velocities, and writes a new file in which every sample is annotated with an `eye_movement` label (`fixation` / `saccade`).

---

## Features

- Works with `.csv`, `.tsv`, `.xlsx`, `.xls` input files
- Interactive column selection (numbered list) or fully scripted via CLI flags
- Automatic time-unit conversion (`ms`, `s`, `µs`)
- Classic I-VT classification with two standard refinements:
  - merging of adjacent fixations separated by a tiny saccade
  - discarding physiologically implausible short fixations (default < 60 ms)
- Per-sample output **and** event-level summary (one row per fixation/saccade)
- Optional plotting of the velocity profile and 2D gaze scatter
- Clean, importable Python API plus a `python -m ivt.cli` command-line entry point
- Pytest suite covering the core algorithm

---

## Installation

```bash
git clone https://github.com/<your-username>/ivt.git
cd ivt
pip install -r requirements.txt
# or, as an installable package (gives you the `ivt` command):
pip install -e .
```

Requires Python ≥ 3.9.

---

## Quick start

### Interactive CLI

```bash
python -m ivt.cli
```

You will be prompted for:

1. The path to the input CSV/Excel file.
2. Which columns correspond to `x`, `y`, and `time`.
3. The time unit (`ms` / `s` / `µs`).
4. The velocity threshold (pixels per second) and minimum fixation duration.
5. The output path.

### Scripted CLI

```bash
python -m ivt.cli \
    --input data/raw_gaze.csv \
    --x gaze_x --y gaze_y --time t \
    --time-unit ms \
    --threshold 1000 \
    --min-fixation-ms 60 \
    --output data/raw_gaze_ivt.csv \
    --summary data/raw_gaze_events.csv
```

### Python API

```python
import pandas as pd
from ivt import IVTConfig, run_ivt, summarize_events

df = pd.read_csv("data/raw_gaze.csv")

config = IVTConfig(
    velocity_threshold=1000.0,       # px/s
    min_fixation_duration_ms=60.0,
    merge_adjacent_fixations=True,
)

labelled = run_ivt(
    df,
    x_col="gaze_x",
    y_col="gaze_y",
    time_col="t",
    time_unit="ms",
    config=config,
)

labelled.to_csv("data/raw_gaze_ivt.csv", index=False)

events = summarize_events(labelled, x_col="gaze_x", y_col="gaze_y")
events.to_csv("data/raw_gaze_events.csv", index=False)
```

---

## Input and output format

### Input

Any tabular file with at least three columns:

| column | meaning                               | unit             |
| ------ | ------------------------------------- | ---------------- |
| x      | horizontal gaze coordinate            | pixels           |
| y      | vertical gaze coordinate              | pixels           |
| time   | timestamp                             | ms, s, or µs     |

Column names are arbitrary — you pick them interactively or pass them as flags.

### Per-sample output

The input file is returned with three extra columns appended:

| new column     | meaning                                         |
| -------------- | ----------------------------------------------- |
| `time_ms`      | timestamp converted to milliseconds             |
| `velocity_pxs` | point-to-point velocity in pixels / second      |
| `eye_movement` | `fixation` or `saccade`                         |

### Event-level summary (optional)

One row per fixation / saccade:

| column        | meaning                                           |
| ------------- | ------------------------------------------------- |
| `event_id`    | running index of the event                        |
| `event_type`  | `fixation` or `saccade`                           |
| `start_ms`    | event start time (ms)                             |
| `end_ms`      | event end time (ms)                               |
| `duration_ms` | event duration (ms)                               |
| `n_samples`   | number of samples in the event                    |
| `centroid_x`  | mean x-coordinate during the event (pixels)      |
| `centroid_y`  | mean y-coordinate during the event (pixels)      |

---

## Algorithm summary

For each sample *i*:

1. Compute the 2D point-to-point velocity

   $v_i = \dfrac{\sqrt{(x_i - x_{i-1})^2 + (y_i - y_{i-1})^2}}{t_i - t_{i-1}}$

   (distance in pixels, time converted to seconds → velocity in px/s).

2. Label the sample **fixation** if $v_i < \tau$, otherwise **saccade**.
3. Collapse consecutive same-label samples into events.
4. *(Optional)* Merge two fixations if the gap between them is shorter than `max_merge_gap_ms` and their centroids are within `max_merge_distance_px`.
5. *(Optional)* Reclassify any fixation shorter than `min_fixation_duration_ms` as a saccade.

### Choosing the threshold

The threshold $\tau$ is expressed here in **pixels per second**, not degrees per second, because pixel-space is the only thing knowable from raw (x, y) coordinates without screen geometry. A common rule of thumb in the literature is ~30°/s; the equivalent in pixels depends on your monitor resolution and viewing distance. `1000 px/s` is a reasonable starting point for typical desktop setups and should be tuned for your recordings. You can convert ° / s to px / s if you know your screen's pixels-per-degree:

```
px_per_deg = screen_px_per_cm * viewing_distance_cm * tan(1°)
threshold_px_s = threshold_deg_s * px_per_deg
```

---

## Project layout

```
ivt_project/
├── ivt/
│   ├── __init__.py      # public API
│   ├── algorithm.py     # core I-VT implementation
│   ├── cli.py           # command-line interface
│   ├── io.py            # CSV / Excel read + write helpers
│   └── viz.py           # optional matplotlib plots
├── examples/
│   └── example_usage.py # synthetic data demo
├── tests/
│   └── test_algorithm.py
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Running the example

```bash
python examples/example_usage.py
```

This generates a synthetic gaze trace, classifies it, and writes three files to `examples/output/`:

- `synthetic_gaze.csv` — raw input
- `synthetic_gaze_ivt.csv` — per-sample labels
- `synthetic_gaze_events.csv` — event-level summary

---

## Running the tests

```bash
pytest -q
```

---

## Reference

Salvucci, D. D., & Goldberg, J. H. (2000). *Identifying fixations and saccades in eye-tracking protocols.* In Proceedings of the 2000 Symposium on Eye Tracking Research & Applications (ETRA '00), pp. 71–78. ACM. https://doi.org/10.1145/355017.355028

---

## License

MIT — see [`LICENSE`](LICENSE).
