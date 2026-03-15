"""BigQuery client initialization and logging."""

import logging

from google.cloud import bigquery


def create_client(key_path: str) -> bigquery.Client:
    """Create a BigQuery client from a service account key file."""
    return bigquery.Client.from_service_account_json(key_path)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
