"""Parquet file loading utilities."""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from gharchive.transform import optimize_types


def load_period(output_dir: Path, start: date, end: date) -> pd.DataFrame:
    """Load and concatenate parquet files for a date range."""
    output_dir = Path(output_dir)
    frames = []
    current = start
    while current <= end:
        path = output_dir / f"{current.strftime('%Y%m%d')}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
        current += timedelta(days=1)
    if not frames:
        raise FileNotFoundError(f"No parquet files found in {output_dir} for {start}–{end}")
    df = pd.concat(frames, ignore_index=True)
    return optimize_types(df)
