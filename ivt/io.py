"""File I/O helpers for the I-VT project.

Reads gaze data from CSV or Excel files and writes annotated output
back in the same format.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SUPPORTED_INPUT = {".csv", ".tsv", ".xlsx", ".xls"}


def read_gaze_file(path: str | Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Read a CSV/TSV/Excel file into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the input file.
    sheet_name : str | int | None
        For Excel files, the sheet to load (default: first sheet).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError(
        f"Unsupported file extension '{ext}'. Supported: {sorted(SUPPORTED_INPUT)}"
    )


def write_gaze_file(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to CSV or Excel based on file extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext == ".tsv":
        df.to_csv(path, index=False, sep="\t")
    elif ext in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"Unsupported output extension '{ext}'.")
    return path
