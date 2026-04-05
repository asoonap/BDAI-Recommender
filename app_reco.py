"""Repo-to-Repo 추천 정성 평가 대시보드.

Usage:
    uv run streamlit run app_reco.py
"""

import pickle
import sqlite3
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# --- Paths ---
DATA_DIR = Path("data")
MODEL_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "repo_metadata.db"

st.set_page_config(page_title="Repo-to-Repo 추천", layout="wide")
st.title("🔍 Repo-to-Repo 추천 정성 평가")


@st.cache_resource
def load_models():
    """ALS 모델, LGBM ranker, 매핑 정보 로드."""
    als_model = pickle.loads((MODEL_DIR / "als_twostage.pkl").read_bytes())
    ranker = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_ranker.txt"))
    mappings = pickle.load(open(MODEL_DIR / "index_mappings.pkl", "rb"))
    name_map = pickle.load(open(MODEL_DIR / "repo_name_map.pkl", "rb"))

    # Metadata
    meta_dict = {}
    meta_full = {}
    if DB_PATH.exists():
        conn = sqlite3.connect(str(DB_PATH))
        meta_df = pd.read_sql_query(
            "SELECT repo_id, repo_name, description, language, stargazers, forks, topics "
            "FROM repo_metadata WHERE http_status = 200",
            conn,
        )
        meta_dict = meta_df.set_index("repo_id")[["language", "stargazers", "forks"]].to_dict(orient="index")
        meta_full = meta_df.set_index("repo_id").to_dict(orient="index")
        conn.close()

    return als_model, ranker, mappings, name_map, meta_dict, meta_full


def find_similar_repos(query_repo_id, als_model, mappings, n_candidates=200):
    """ALS item embedding 기반 유사 repo 검색 (Stage 1)."""
    item2idx = mappings["item2idx"]
    idx2item = mappings["idx2item"]

    if query_repo_id not in item2idx:
        return []

    query_idx = item2idx[query_repo_id]
    query_vec = als_model.item_factors[query_idx].reshape(1, -1)

    # Cosine similarity with all items
    sims = cosine_similarity(query_vec, als_model.item_factors)[0]
    top_idxs = np.argsort(-sims)[1 : n_candidates + 1]  # exclude self

    candidates = []
    for idx in top_idxs:
        repo_id = idx2item[idx]
        candidates.append((repo_id, float(sims[idx])))
    return candidates


def rerank_candidates(query_repo_id, candidates, als_model, ranker, mappings, meta_dict):
    """LGBM으로 후보 re-rank (Stage 2)."""
    item2idx = mappings["item2idx"]
    pop_dict = mappings["pop_dict"]
    lang2idx = mappings["lang2idx"]
    item_factors = als_model.item_factors

    query_idx = item2idx[query_repo_id]
    q_vec = item_factors[query_idx].reshape(1, -1)

    rows, repo_ids = [], []
    for repo_id, als_score in candidates:
        iidx = item2idx.get(repo_id)
        if iidx is None:
            continue
        i_vec = item_factors[iidx].reshape(1, -1)
        cos_sim = float(cosine_similarity(q_vec, i_vec)[0, 0])

        meta = meta_dict.get(repo_id, {})
        row = [
            als_score,
            cos_sim,
            np.log1p(pop_dict.get(repo_id, 0)),
            np.log1p(meta.get("stargazers", 0) or 0),
            np.log1p(meta.get("forks", 0) or 0),
            lang2idx.get(meta.get("language"), 0),
            0,  # user_total_score — repo-to-repo이므로 0
            0,  # user_unique_repos
        ]
        rows.append(row)
        repo_ids.append(repo_id)

    if not rows:
        return []

    X = np.array(rows, dtype=np.float32)
    scores = ranker.predict(X)
    ranked_idx = np.argsort(-scores)

    return [(repo_ids[i], float(scores[i]), rows[i]) for i in ranked_idx]


def explain_recommendation(features, feature_names):
    """피처 값 기반 추천 이유 생성."""
    reasons = []
    als_score, cos_sim, log_pop, log_stars, log_forks, lang, _, _ = features

    if cos_sim > 0.5:
        reasons.append(f"높은 임베딩 유사도 ({cos_sim:.3f})")
    elif cos_sim > 0.3:
        reasons.append(f"중간 임베딩 유사도 ({cos_sim:.3f})")

    if log_pop > 5:
        reasons.append(f"인기 repo (popularity={np.expm1(log_pop):.0f})")

    if log_stars > 7:
        reasons.append(f"⭐ {np.expm1(log_stars):.0f} stars")

    if not reasons:
        reasons.append("ALS 임베딩 기반 추천")

    return " · ".join(reasons)


# --- Main ---
try:
    als_model, ranker, mappings, name_map, meta_dict, meta_full = load_models()
except Exception as e:
    st.error(f"모델 로드 실패: {e}\n\n07_two_stage.ipynb를 먼저 실행하세요.")
    st.stop()

# Reverse name map for search
name_to_id = {name: rid for rid, name in name_map.items()}

st.sidebar.header("설정")
top_k = st.sidebar.slider("추천 개수", 5, 50, 20)
stage = st.sidebar.radio("추천 방식", ["Two-Stage (ALS → LGBM)", "ALS Only"])

# --- Input ---
st.subheader("Input: 기준 Repo")

col1, col2 = st.columns([3, 1])
with col1:
    search_text = st.text_input("Repo 검색 (이름 일부 입력)", placeholder="예: pytorch, react, openclaw")
with col2:
    repo_id_input = st.text_input("또는 Repo ID 직접 입력", placeholder="예: 65600975")

# Search matching
query_repo_id = None
if repo_id_input:
    try:
        query_repo_id = int(repo_id_input)
    except ValueError:
        st.warning("유효한 숫자 ID를 입력하세요.")

elif search_text:
    matches = [(rid, name) for name, rid in name_to_id.items() if search_text.lower() in name.lower()]
    matches.sort(key=lambda x: x[1])
    if matches:
        options = {f"{name} (id={rid})": rid for rid, name in matches[:20]}
        selected = st.selectbox("검색 결과", list(options.keys()))
        query_repo_id = options[selected]
    else:
        st.warning(f"'{search_text}'에 매칭되는 repo 없음")

# --- Recommendation ---
if query_repo_id is not None:
    repo_name = name_map.get(query_repo_id, f"repo_{query_repo_id}")
    meta_info = meta_full.get(query_repo_id, {})

    st.markdown(f"### 📦 [{repo_name}](https://github.com/{repo_name})")
    if meta_info.get("description"):
        st.caption(meta_info["description"])

    info_cols = st.columns(4)
    stars_val = meta_info.get("stargazers")
    forks_val = meta_info.get("forks")
    info_cols[0].metric("Stars", f"{stars_val:,}" if isinstance(stars_val, (int, float)) else "?")
    info_cols[1].metric("Forks", f"{forks_val:,}" if isinstance(forks_val, (int, float)) else "?")
    info_cols[2].metric("Language", meta_info.get("language", "?"))
    info_cols[3].metric("Repo ID", query_repo_id)

    st.divider()

    if query_repo_id not in mappings["item2idx"]:
        st.error(f"Repo {query_repo_id}이 학습 데이터에 없습니다 (cold start)")
    else:
        with st.spinner("추천 생성 중..."):
            candidates = find_similar_repos(query_repo_id, als_model, mappings, n_candidates=200)

            if stage == "Two-Stage (ALS → LGBM)":
                ranked = rerank_candidates(query_repo_id, candidates, als_model, ranker, mappings, meta_dict)
            else:
                ranked = [(rid, score, [score, score, 0, 0, 0, 0, 0, 0]) for rid, score in candidates]

        st.subheader(f"추천 결과 (Top-{top_k})")

        rows = []
        for rank, (rid, score, features) in enumerate(ranked[:top_k], 1):
            rname = name_map.get(rid, f"repo_{rid}")
            rmeta = meta_full.get(rid, {})
            reason = explain_recommendation(features, mappings["feature_names"])

            rows.append({
                "순위": rank,
                "Repo": rname,
                "Score": f"{score:.4f}",
                "Stars": f"{rmeta['stargazers']:,}" if isinstance(rmeta.get("stargazers"), (int, float)) else "-",
                "Language": rmeta.get("language", "-"),
                "Description": (rmeta.get("description") or "-")[:80],
                "추천 이유": reason,
                "Link": f"https://github.com/{rname}",
                "repo_id": rid,
            })

        df_display = pd.DataFrame(rows)

        # Styled table
        st.dataframe(
            df_display[["순위", "Repo", "Score", "Stars", "Language", "Description", "추천 이유"]],
            use_container_width=True,
            hide_index=True,
        )

        # Expandable details
        with st.expander("상세 정보 (링크 포함)"):
            for row in rows:
                st.markdown(
                    f"**{row['순위']}. [{row['Repo']}]({row['Link']})** — "
                    f"Score: {row['Score']} · {row['Stars']} ⭐ · {row['Language']} · "
                    f"{row['추천 이유']}"
                )
                if meta_full.get(row["repo_id"], {}).get("description"):
                    st.caption(f"  {meta_full[row['repo_id']]['description']}")

        # Feature breakdown for top-5
        with st.expander("Top-5 피처 breakdown"):
            for rank, (rid, score, features) in enumerate(ranked[:5], 1):
                rname = name_map.get(rid, f"repo_{rid}")
                st.markdown(f"**{rank}. {rname}** (score={score:.4f})")
                feat_df = pd.DataFrame({
                    "Feature": mappings["feature_names"],
                    "Value": [f"{v:.4f}" for v in features],
                })
                st.dataframe(feat_df, hide_index=True, use_container_width=False)

st.sidebar.markdown("---")
st.sidebar.caption("BDA 2기 · Two-Stage Recommendation")
