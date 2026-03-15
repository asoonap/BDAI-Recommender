"""Daily aggregation extraction from GitHub Archive via BigQuery."""

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

QUERY_TEMPLATE = """
SELECT
    actor.id AS actor_id,
    repo.id AS repo_id,
    type,
    COUNT(*) AS cnt
FROM `githubarchive.day.{date_str}`
GROUP BY 1, 2, 3
"""

COST_PER_TB = 6.25  # BigQuery on-demand pricing (USD)


def dry_run(client: bigquery.Client, date_str: str) -> dict[str, float]:
    """Run a dry-run query and return estimated bytes/cost."""
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    query = QUERY_TEMPLATE.format(date_str=date_str)
    job = client.query(query, job_config=job_config)
    bytes_processed = job.total_bytes_processed
    cost = bytes_processed / (1024**4) * COST_PER_TB
    return {"bytes_processed": bytes_processed, "estimated_cost_usd": cost}


def extract_single_day(client: bigquery.Client, date_str: str) -> pd.DataFrame:
    """Execute aggregation query for a single day, return DataFrame."""
    query = QUERY_TEMPLATE.format(date_str=date_str)
    return client.query(query).to_dataframe()


def extract_date_range(
    client: bigquery.Client,
    start_date: date,
    end_date: date,
    output_dir: Path,
    logger: logging.Logger,
) -> list[str]:
    """Extract daily aggregations for a date range, saving parquet files.

    Skips dates whose parquet file already exists.
    Returns list of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y%m%d")
        out_path = output_dir / f"{date_str}.parquet"

        if out_path.exists():
            logger.info(f"SKIP {date_str} (already exists)")
            saved.append(str(out_path))
            current += timedelta(days=1)
            continue

        logger.info(f"Extracting {date_str} ...")
        df = extract_single_day(client, date_str)
        df.to_parquet(out_path, index=False)
        logger.info(f"  → {len(df):,} rows, {out_path.stat().st_size / 1024:.0f} KB")
        saved.append(str(out_path))
        current += timedelta(days=1)

    return saved
