"""Type optimization and aggregation transforms."""

import pandas as pd


def optimize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns and convert type to category."""
    df = df.copy()
    for col in ("actor_id", "repo_id", "cnt"):
        df[col] = df[col].astype("Int32")
    df["type"] = df["type"].astype("category")
    return df
