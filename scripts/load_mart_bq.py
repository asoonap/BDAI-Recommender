"""
Load daily_agg parquet files into BigQuery mart.fact_user_repo_activity
"""
import os
import glob
import datetime
import pandas as pd
from google.cloud import bigquery

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/kakao/Documents/gcp-key.json"

PROJECT = "bda-coai"
DATASET = "mart"
TABLE = "fact_user_repo_activity"
FULL_TABLE = f"{PROJECT}.{DATASET}.{TABLE}"
PARQUET_DIR = "/Users/kakao/bda-2/data/daily_agg"

client = bigquery.Client(project=PROJECT)

# --- 1. Check existing datasets to pick location ---
print("Existing datasets:")
existing = list(client.list_datasets())
if existing:
    for ds in existing:
        info = client.get_dataset(ds.reference)
        print(f"  {ds.dataset_id} -> location: {info.location}")
    # Use same location as first existing dataset
    first_loc = client.get_dataset(existing[0].reference).location
else:
    first_loc = "US"
print(f"\nWill use location: {first_loc}")

# --- 2. Create dataset if not exists ---
dataset_ref = bigquery.Dataset(f"{PROJECT}.{DATASET}")
dataset_ref.location = first_loc
dataset = client.create_dataset(dataset_ref, exists_ok=True)
print(f"Dataset '{DATASET}' ready (location={first_loc})")

# --- 3. Read all parquets, add activity_date, rename columns ---
parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
print(f"\nFound {len(parquet_files)} parquet files")

frames = []
for f in parquet_files:
    basename = os.path.basename(f).replace(".parquet", "")
    date = datetime.date(int(basename[:4]), int(basename[4:6]), int(basename[6:8]))
    df = pd.read_parquet(f)
    df = df.rename(columns={
        "actor_id": "user_id",
        "repo_id": "repo_id",
        "type": "action",
        "cnt": "event_count",
    })
    df["activity_date"] = date
    frames.append(df)

combined = pd.concat(frames, ignore_index=True)
# Drop rows with null keys (some repo_id are NA)
null_before = combined.isnull().any(axis=1).sum()
combined = combined.dropna(subset=["user_id", "repo_id"]).reset_index(drop=True)
print(f"Dropped {null_before} rows with null user_id/repo_id")
# Cast to proper types for BQ
combined["user_id"] = combined["user_id"].astype("int64")
combined["repo_id"] = combined["repo_id"].astype("int64")
combined["action"] = combined["action"].astype("str")
combined["event_count"] = combined["event_count"].astype("int64")
combined["activity_date"] = pd.to_datetime(combined["activity_date"]).dt.date

print(f"Combined DataFrame: {len(combined):,} rows, columns={list(combined.columns)}")
print(combined.dtypes)

# --- 4. Define schema and load ---
schema = [
    bigquery.SchemaField("user_id", "INTEGER"),
    bigquery.SchemaField("repo_id", "INTEGER"),
    bigquery.SchemaField("action", "STRING"),
    bigquery.SchemaField("event_count", "INTEGER"),
    bigquery.SchemaField("activity_date", "DATE"),
]

job_config = bigquery.LoadJobConfig(
    schema=schema,
    write_disposition="WRITE_TRUNCATE",
)

print(f"\nUploading to {FULL_TABLE} ...")
job = client.load_table_from_dataframe(combined, FULL_TABLE, job_config=job_config)
job.result()  # wait
table = client.get_table(FULL_TABLE)
print(f"Loaded {table.num_rows:,} rows into {FULL_TABLE}")

# --- 5. Validation queries ---
print("\n" + "=" * 60)
print("VALIDATION QUERIES")
print("=" * 60)

# 5a. Total row count
q = f"SELECT COUNT(*) AS total_rows FROM `{FULL_TABLE}`"
print(f"\n-- Total row count --")
for row in client.query(q).result():
    print(f"  {row.total_rows:,}")

# 5b. Date range
q = f"SELECT MIN(activity_date) AS min_date, MAX(activity_date) AS max_date, COUNT(DISTINCT activity_date) AS n_days FROM `{FULL_TABLE}`"
print(f"\n-- Date range --")
for row in client.query(q).result():
    print(f"  {row.min_date} ~ {row.max_date}  ({row.n_days} days)")

# 5c. Sample rows
q = f"SELECT * FROM `{FULL_TABLE}` LIMIT 5"
print(f"\n-- Sample rows --")
for row in client.query(q).result():
    print(f"  user_id={row.user_id}, repo_id={row.repo_id}, action={row.action}, event_count={row.event_count}, date={row.activity_date}")

# 5d. DAU by day (first 5 days)
q = f"""
SELECT activity_date, COUNT(DISTINCT user_id) AS dau
FROM `{FULL_TABLE}`
GROUP BY activity_date
ORDER BY activity_date
LIMIT 5
"""
print(f"\n-- DAU (first 5 days) --")
for row in client.query(q).result():
    print(f"  {row.activity_date}  DAU={row.dau:,}")

print("\nDone!")
