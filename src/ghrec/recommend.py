"""Popularity-based recommendation scoring."""

import pandas as pd

from gharchive.loader import load_period


def popularity_scores(
    df: pd.DataFrame, weights: dict[str, float]
) -> pd.Series:
    """Compute weighted popularity score per repo.

    Returns a Series indexed by repo_id, sorted descending.
    """
    pivot = (
        df.groupby(["repo_id", "type"], observed=True)["cnt"]
        .sum()
        .reset_index()
        .pivot_table(index="repo_id", columns="type", values="cnt", fill_value=0)
    )
    score = pd.Series(0.0, index=pivot.index, name="score")
    for event_type, weight in weights.items():
        if event_type in pivot.columns:
            score += pivot[event_type] * weight
    return score.sort_values(ascending=False)


def top_n_repos(scores: pd.Series, n: int) -> pd.Series:
    """Return the top N repos by score."""
    return scores.head(n)
