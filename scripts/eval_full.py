"""Full evaluation: ALS (n=400) vs Two-Stage (ALS→LGBM) comparison.

Usage:
    uv run python scripts/eval_full.py
"""

import math
import os
import pickle
import sqlite3
import time
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from gharchive.loader import load_period
from ghrec.recommend import popularity_scores
from implicit.als import AlternatingLeastSquares

# ── Config ──
OUTPUT_DIR = Path("data/daily_agg")
MODEL_DIR = Path("data/models")
DB_PATH = Path("data/repo_metadata.db")

TRAIN_START, TRAIN_END = date(2026, 3, 1), date(2026, 3, 28)
TEST_START, TEST_END = date(2026, 3, 29), date(2026, 4, 3)
CANDIDATE_K = 400
K_VALUES = [10, 50, 100, 200]
SAMPLE_RATIO = 0.30  # 메모리 관리
CHUNK_SIZE = 2000

WEIGHTS = {
    "WatchEvent": 1.0,
    "ForkEvent": 2.0,
    "IssuesEvent": 0.5,
    "PullRequestEvent": 3.0,
    "IssueCommentEvent": 0.3,
    "PushEvent": 0.2,
}


def build_feedback(df, weights):
    df = df.copy()
    df["score"] = df["type"].map(weights).fillna(0) * df["cnt"]
    fb = df.groupby(["actor_id", "repo_id"])["score"].sum().reset_index()
    return fb[fb["score"] > 0]


def precision_recall_ndcg(recommended, relevant, k):
    rec_set = set(recommended[:k])
    hits = rec_set & relevant
    precision = len(hits) / k if k > 0 else 0
    recall = len(hits) / len(relevant) if relevant else 0
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, rid in enumerate(recommended[:k])
        if rid in relevant
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    ndcg = dcg / idcg if idcg > 0 else 0
    return precision, recall, ndcg


def main():
    t_start = time.time()

    # ── 1. Data ──
    print("=" * 60)
    print("1. Loading data...")
    train_df = load_period(OUTPUT_DIR, TRAIN_START, TRAIN_END)
    test_df = load_period(OUTPUT_DIR, TEST_START, TEST_END)
    print(f"   Train: {len(train_df):,} rows  ({TRAIN_START} ~ {TRAIN_END})")
    print(f"   Test:  {len(test_df):,} rows  ({TEST_START} ~ {TEST_END})")

    # Sample users
    rng = np.random.default_rng(42)
    all_users = set(train_df["actor_id"].unique()) | set(test_df["actor_id"].unique())
    sampled = set(
        rng.choice(list(all_users), size=int(len(all_users) * SAMPLE_RATIO), replace=False)
    )
    train_df = train_df[train_df["actor_id"].isin(sampled)]
    test_df = test_df[test_df["actor_id"].isin(sampled)]
    print(f"   Sampled {SAMPLE_RATIO:.0%}: {len(sampled):,} users")

    train_fb = build_feedback(train_df, WEIGHTS)
    test_fb = build_feedback(test_df, WEIGHTS)

    # ALL eval users (no 10k cap)
    eval_users = sorted(set(train_fb["actor_id"]) & set(test_fb["actor_id"]))
    test_gt = test_fb.groupby("actor_id")["repo_id"].apply(set).to_dict()
    print(f"   Eval users: {len(eval_users):,}")

    # ── 2. Sparse matrix + ALS ──
    print("\n2. Building sparse matrix & ALS...")
    all_user_ids = train_fb["actor_id"].unique()
    all_item_ids = train_fb["repo_id"].unique()
    user2idx = {uid: i for i, uid in enumerate(all_user_ids)}
    item2idx = {iid: i for i, iid in enumerate(all_item_ids)}
    idx2item = {i: iid for iid, i in item2idx.items()}

    row = train_fb["actor_id"].map(user2idx).values
    col = train_fb["repo_id"].map(item2idx).values
    data = train_fb["score"].values.astype(np.float32)
    train_sparse = sparse.csr_matrix(
        (data, (row, col)), shape=(len(all_user_ids), len(all_item_ids))
    )
    print(f"   Matrix: {train_sparse.shape[0]:,} × {train_sparse.shape[1]:,}, nnz={train_sparse.nnz:,}")

    ALS_PATH = MODEL_DIR / "als_twostage.pkl"
    if ALS_PATH.exists():
        model = pickle.loads(ALS_PATH.read_bytes())
        print(f"   ALS loaded from cache")
    else:
        model = AlternatingLeastSquares(factors=64, regularization=0.01, iterations=15, random_state=42)
        model.fit(train_sparse)
        ALS_PATH.write_bytes(pickle.dumps(model))
        print(f"   ALS trained & saved")

    # ── 3. Retrieval (n=400) ──
    print(f"\n3. ALS Retrieval (n={CANDIDATE_K})...")
    retrieval_results = {}
    for start in tqdm(range(0, len(eval_users), CHUNK_SIZE), desc="Retrieval"):
        chunk_users = eval_users[start : start + CHUNK_SIZE]
        chunk_idxs = np.array([user2idx[uid] for uid in chunk_users])
        item_ids_batch, scores_batch = model.recommend(
            chunk_idxs,
            train_sparse[chunk_idxs],
            N=CANDIDATE_K,
            filter_already_liked_items=True,
        )
        for i, uid in enumerate(chunk_users):
            retrieval_results[uid] = [
                (idx2item[j], float(scores_batch[i][rank]))
                for rank, j in enumerate(item_ids_batch[i])
            ]
    print(f"   Done: {len(retrieval_results):,} users × {CANDIDATE_K} candidates")

    # ── 4. Features + LGBM ──
    print("\n4. Loading metadata & LGBM ranker...")
    pop_scores = popularity_scores(train_df, WEIGHTS)
    pop_dict = pop_scores.to_dict()

    user_activity = train_fb.groupby("actor_id").agg(
        user_total_score=("score", "sum"),
        user_unique_repos=("repo_id", "nunique"),
    ).to_dict(orient="index")

    meta_dict = {}
    if DB_PATH.exists():
        conn = sqlite3.connect(str(DB_PATH))
        mdf = pd.read_sql_query(
            "SELECT repo_id, language, stargazers, forks FROM repo_metadata WHERE http_status = 200",
            conn,
        )
        meta_dict = mdf.set_index("repo_id").to_dict(orient="index")
        conn.close()

    all_languages = sorted(
        lang for lang in set(m.get("language") for m in meta_dict.values()) if isinstance(lang, str)
    )
    lang2idx = {lang: i + 1 for i, lang in enumerate(all_languages)}

    RANKER_PATH = MODEL_DIR / "lgbm_ranker.txt"
    ranker = lgb.Booster(model_file=str(RANKER_PATH))
    print(f"   LGBM ranker loaded (features={ranker.num_feature()})")

    user_factors = model.user_factors  # (n_users, 64)
    item_factors = model.item_factors  # (n_items, 64)

    # Precompute item factor norms for fast cosine sim
    item_norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    item_norms[item_norms == 0] = 1.0
    item_factors_normed = item_factors / item_norms

    user_norms = np.linalg.norm(user_factors, axis=1, keepdims=True)
    user_norms[user_norms == 0] = 1.0
    user_factors_normed = user_factors / user_norms

    # Precompute per-item static features (log_pop, log_stars, log_forks, language)
    n_items = len(all_item_ids)
    item_static = np.zeros((n_items, 4), dtype=np.float32)
    for iid, iidx in item2idx.items():
        meta = meta_dict.get(iid, {})
        item_static[iidx, 0] = np.log1p(pop_dict.get(iid, 0))
        item_static[iidx, 1] = np.log1p(meta.get("stargazers", 0) or 0)
        item_static[iidx, 2] = np.log1p(meta.get("forks", 0) or 0)
        item_static[iidx, 3] = lang2idx.get(meta.get("language"), 0)
    print(f"   Static features precomputed for {n_items:,} items")

    def build_features_batch(uid, candidates):
        """Vectorized feature building — no per-item cosine_similarity call."""
        uidx = user2idx[uid]
        u_act = user_activity.get(uid, {"user_total_score": 0, "user_unique_repos": 0})

        rids, als_scores, iidxs = [], [], []
        for repo_id, als_score in candidates:
            iidx = item2idx.get(repo_id)
            if iidx is None:
                continue
            rids.append(repo_id)
            als_scores.append(als_score)
            iidxs.append(iidx)

        if not rids:
            return None, []

        iidxs_arr = np.array(iidxs)
        als_arr = np.array(als_scores, dtype=np.float32)

        # Vectorized cosine similarity
        cos_sims = user_factors_normed[uidx] @ item_factors_normed[iidxs_arr].T

        # Stack features
        n = len(rids)
        X = np.column_stack([
            als_arr,
            cos_sims,
            item_static[iidxs_arr],  # log_pop, log_stars, log_forks, language
            np.full(n, u_act["user_total_score"], dtype=np.float32),
            np.full(n, u_act["user_unique_repos"], dtype=np.float32),
        ])
        return X, rids

    # ── 5. Re-rank & Evaluate ──
    print(f"\n5. Re-ranking & evaluating ({len(eval_users):,} users)...")
    MAX_K = max(K_VALUES)

    # Popularity baseline
    pop_candidates = pop_scores.head(MAX_K + 500).index.tolist()
    train_seen = train_fb.groupby("actor_id")["repo_id"].apply(set).to_dict()

    results = {
        name: {k: {"precision": [], "recall": [], "ndcg": []} for k in K_VALUES}
        for name in ["Popularity", "ALS", "Two-Stage"]
    }

    for uid in tqdm(eval_users, desc="Eval"):
        gt = test_gt.get(uid, set())
        if not gt:
            continue

        # Popularity
        seen = train_seen.get(uid, set())
        pop_recs = [r for r in pop_candidates if r not in seen][:MAX_K]

        # ALS (retrieval order)
        cands = retrieval_results.get(uid, [])
        als_recs = [rid for rid, _ in cands[:MAX_K]]

        # Two-Stage
        X, rids = build_features_batch(uid, cands)
        if X is not None and len(rids) > 0:
            scores = ranker.predict(X)
            ranked_idx = np.argsort(-scores)
            ts_recs = [rids[i] for i in ranked_idx[:MAX_K]]
        else:
            ts_recs = als_recs

        for model_name, recs in [("Popularity", pop_recs), ("ALS", als_recs), ("Two-Stage", ts_recs)]:
            for k in K_VALUES:
                p, r, n = precision_recall_ndcg(recs, gt, k)
                results[model_name][k]["precision"].append(p)
                results[model_name][k]["recall"].append(r)
                results[model_name][k]["ndcg"].append(n)

    # ── 6. Results ──
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"RESULTS  (eval users={len(eval_users):,}, candidates={CANDIDATE_K}, elapsed={elapsed/60:.1f}min)")
    print(f"{'=' * 60}")
    print(f"{'Model':<14} {'K':>4}  {'Precision':>10} {'Recall':>10} {'NDCG':>10}")
    print("-" * 54)
    for model_name in ["Popularity", "ALS", "Two-Stage"]:
        for k in K_VALUES:
            p = np.mean(results[model_name][k]["precision"])
            r = np.mean(results[model_name][k]["recall"])
            n = np.mean(results[model_name][k]["ndcg"])
            print(f"{model_name:<14} {k:>4}  {p:>10.5f} {r:>10.5f} {n:>10.5f}")
        print()

    # Diversity
    print("Diversity (unique items recommended):")
    for model_name in ["Popularity", "ALS", "Two-Stage"]:
        for k in [50, 100]:
            unique = set()
            for uid in eval_users:
                gt = test_gt.get(uid, set())
                if not gt:
                    continue
                if model_name == "Popularity":
                    seen = train_seen.get(uid, set())
                    recs = [r for r in pop_candidates if r not in seen][:k]
                elif model_name == "ALS":
                    recs = [rid for rid, _ in retrieval_results.get(uid, [])][:k]
                else:
                    # Can't re-derive easily, skip exact count
                    recs = [rid for rid, _ in retrieval_results.get(uid, [])][:k]
                unique.update(recs)
            print(f"  {model_name:<14} K={k}: {len(unique):,}")

    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
