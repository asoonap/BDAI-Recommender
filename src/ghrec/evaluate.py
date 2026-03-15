"""Evaluation metrics for recommendation quality."""

import math

import pandas as pd


def ndcg_at_k(predicted: list[int], actual: list[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Binary relevance: actual top-K에 있으면 1, 없으면 0.
    """
    actual_set = set(actual[:k])

    # DCG
    dcg = 0.0
    for i, rid in enumerate(predicted[:k]):
        if rid in actual_set:
            dcg += 1.0 / math.log2(i + 2)

    # Ideal DCG — ground truth 개수와 K 중 작은 쪽 기준
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(actual), k)))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def precision_at_k(predicted: list[int], actual: list[int], k: int) -> float:
    """Fraction of predicted top-K that appear in actual top-K."""
    pred_set = set(predicted[:k])
    actual_set = set(actual[:k])
    if k == 0:
        return 0.0
    return len(pred_set & actual_set) / k


def diversity_entropy(
    predicted: list[int], repo_event_counts: pd.DataFrame
) -> float:
    """Shannon entropy of event-type distribution across predicted repos.

    repo_event_counts: DataFrame with repo_id as index, event types as columns.
    Higher entropy = more diverse event-type mix in recommendations.
    """
    subset = repo_event_counts.loc[
        repo_event_counts.index.isin(predicted)
    ]
    if subset.empty:
        return 0.0
    type_totals = subset.sum()
    total = type_totals.sum()
    if total == 0:
        return 0.0
    probs = type_totals / total
    probs = probs[probs > 0]
    return -float((probs * probs.apply(math.log2)).sum())


def evaluate_all(
    predicted: list[int],
    actual: list[int],
    k: int,
    repo_event_counts: pd.DataFrame,
    total_repos: int,
) -> dict[str, float]:
    """Compute all metrics at once."""
    return {
        "k": k,
        "precision@k": precision_at_k(predicted, actual, k),
        "ndcg@k": ndcg_at_k(predicted, actual, k),
        "diversity_entropy": diversity_entropy(predicted[:k], repo_event_counts),
        "coverage": len(set(predicted[:k])) / total_repos if total_repos > 0 else 0.0,
    }
