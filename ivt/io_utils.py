"""
File I/O utilities for the I-VT project.

Handles reading CSV / Excel files and writing outputs (augmented data,
fixation tables, saccade tables) back to disk.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import pandas as pd


SUPPORTED_INPUT_EXT = {".csv", ".tsv", ".xlsx", ".xls"}


def read_gaze_file(
    path: Union[str, Path],
    sheet_name: Union[str, int, None] = 0,
) -> pd.DataFrame:
    """Read a gaze data file (CSV, TSV, or Excel) into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the input file.
    sheet_name : str, int, or None
        Sheet to read for Excel files. Ignored for CSV/TSV.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_INPUT_EXT:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {sorted(SUPPORTED_INPUT_EXT)}"
        )

    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    # Excel
    return pd.read_excel(path, sheet_name=sheet_name)


def list_excel_sheets(path: Union[str, Path]) -> list[str]:
    """Return sheet names of an Excel workbook."""
    path = Path(path)
    xl = pd.ExcelFile(path)
    return xl.sheet_names


def write_ivt_outputs(
    result,  # IVTResult; not type-hinted to avoid circular import
    output_dir: Union[str, Path],
    basename: str = "ivt_result",
) -> dict:
    """Write the IVT result tables to disk.

    Creates three files in `output_dir`:
      - {basename}_samples.csv      (per-sample data + labels)
      - {basename}_fixations.csv    (fixation summary)
      - {basename}_saccades.csv     (saccade summary)

    Returns
    -------
    dict
        Mapping of output type -> absolute path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "samples": output_dir / f"{basename}_samples.csv",
        "fixations": output_dir / f"{basename}_fixations.csv",
        "saccades": output_dir / f"{basename}_saccades.csv",
    }
    result.data.to_csv(paths["samples"], index=False)
    result.fixations.to_csv(paths["fixations"], index=False)
    result.saccades.to_csv(paths["saccades"], index=False)

    return {k: str(v.resolve()) for k, v in paths.items()}
