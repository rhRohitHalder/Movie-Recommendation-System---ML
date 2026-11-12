import os
import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# =============================
# CONFIG & PATHS
# =============================
DEFAULT_API_BASE = "http://127.0.0.1:8000"

try:
    if "API_BASE" in st.secrets:
        DEFAULT_API_BASE = st.secrets["API_BASE"]
except Exception:
    pass

if os.getenv("API_BASE"):
    DEFAULT_API_BASE = os.getenv("API_BASE")
elif os.getenv("RENDER_EXTERNAL_URL"):
    DEFAULT_API_BASE = os.getenv("RENDER_EXTERNAL_URL")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# =============================
# STYLES (Modern Netflix/IMDb theme)
# =============================
st.markdown(
    """
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
.small-muted { color:#9ca3af; font-size: 0.92rem; }
.movie-title { font-size: 0.92rem; line-height: 1.2rem; font-weight: 600; height: 2.4rem; overflow: hidden; margin-top: 6px; }
.card { border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 16px; background: rgba(255,255,255,0.05); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 6px; background: #e50914; color: white; font-size: 0.8rem; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# =============================
# DIRECT LOCAL DATA ENGINE
# =============================
@st.cache_resource(show_spinner="Loading movie dataset into memory...")
def load_local_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df_path = os.path.join(base_dir, "df.pkl")
    matrix_path = os.path.join(base_dir, "tfidf_matrix.pkl")

    if not os.path.exists(df_path) or not os.path.exists(matrix_path):
        return None, None

    with open(df_path, "rb") as f:
        df = pickle.load(f)
    with open(matrix_path, "rb") as f:
        matrix = pickle.load(f)

    return df, matrix


local_df, local_matrix = load_local_engine()


# =============================
# STATE & ROUTING
# =============================
if "view" not in st.session_state:
    st.session_state.view = "home"
if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None
if "selected_movie_title" not in st.session_state:
    st.session_state.selected_movie_title = None
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = DEFAULT_API_BASE.rstrip("/")

# Initialize TMDB key from env or secrets
if "tmdb_api_key" not in st.session_state:
    env_key = os.getenv("TMDB_API_KEY", "").strip()
    secret_key = ""
    try:
        if "TMDB_API_KEY" in st.secrets:
            secret_key = st.secrets["TMDB_API_KEY"].strip()
    except Exception:
        pass
    st.session_state.tmdb_api_key = env_key or secret_key

qp_view = st.query_params.get("view")
qp_id = st.query_params.get("id")
if qp_view in ("home", "details"):
    st.session_state.view = qp_view
if qp_id:
    try:
        st.session_state.selected_movie_id = int(qp_id)
        st.session_state.view = "details"
    except Exception:
        pass


def goto_home():
    st.session_state.view = "home"
    st.query_params["view"] = "home"
    if "id" in st.query_params:
        del st.query_params["id"]
    st.rerun()


def goto_details(movie_id: int, title: str = ""):
    st.session_state.view = "details"
    st.session_state.selected_movie_id = int(movie_id)
    st.session_state.selected_movie_title = title
    st.query_params["view"] = "details"
    st.query_params["id"] = str(int(movie_id))
    st.rerun()


# =============================
# HELPERS
# =============================
def get_tmdb_key() -> str:
    return st.session_state.get("tmdb_api_key", "").strip()


def make_poster(path):
    if pd.isna(path) or not path or str(path).strip() == "":
        return None
    p = str(path).strip()
    if p.startswith("http"):
        return p
    if not p.startswith("/"):
        p = "/" + p
    return f"{TMDB_IMG}{p}"


# ---------- TMDB DIRECT HELPERS ----------
def tmdb_live_get(path: str, params: dict):
    key = get_tmdb_key()
    if not key:
        return None
    q = dict(params)
    q["api_key"] = key
    try:
        r = requests.get(f"{TMDB_BASE}{path}", params=q, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_home_cards(category: str, limit: int = 24):
    key = get_tmdb_key()

    # 1) Live TMDB
    if key:
        tmdb_path = "/trending/movie/day" if category == "trending" else f"/movie/{category}"
        data = tmdb_live_get(tmdb_path, {"language": "en-US", "page": 1})
        if data and data.get("results"):
            cards = []
            for m in data["results"][:limit]:
                cards.append({
                    "tmdb_id": int(m["id"]),
                    "title": m.get("title") or m.get("name") or "Untitled",
                    "poster_url": make_poster(m.get("poster_path")),
                    "release_date": m.get("release_date", ""),
                    "vote_average": m.get("vote_average"),
                })
            return cards, None

    # 2) Local Dataset Fallback
    if local_df is not None:
        sub = local_df.copy()
        if category in ("popular", "trending"):
            if "popularity" in sub.columns:
                sub["_pop"] = pd.to_numeric(sub["popularity"], errors="coerce").fillna(0)
                sub = sub.sort_values(by="_pop", ascending=False)
        elif category == "top_rated":
            if "vote_average" in sub.columns:
                sub["_vote"] = pd.to_numeric(sub["vote_average"], errors="coerce").fillna(0)
                sub = sub[sub["_vote"] > 0].sort_values(by="_vote", ascending=False)
        elif category in ("upcoming", "now_playing"):
            if "release_date" in sub.columns:
                sub["_date"] = pd.to_datetime(sub["release_date"], errors="coerce")
                sub = sub.dropna(subset=["_date"]).sort_values(by="_date", ascending=False)

        cards = []
        for idx, row in sub.head(limit).iterrows():
            vote = row.get("vote_average")
            cards.append({
                "tmdb_id": int(idx),
                "title": str(row.get("title", "Untitled")),
                "poster_url": make_poster(row.get("poster_path")),
                "release_date": str(row.get("release_date", "")),
                "vote_average": float(vote) if pd.notna(vote) else None,
            })
        return cards, None

    return None, "Dataset not available"


def search_movies(query: str, limit: int = 24):
    key = get_tmdb_key()

    # 1) Live TMDB Search
    if key:
        data = tmdb_live_get("/search/movie", {"query": query, "include_adult": "false", "language": "en-US", "page": 1})
        if data and data.get("results"):
            suggestions = []
            cards = []
            for m in data["results"][:limit]:
                title = m.get("title", "")
                year = (m.get("release_date") or "")[:4]
                label = f"{title} ({year})" if year else title
                m_id = int(m["id"])
                suggestions.append((label, m_id, title))
                cards.append({
                    "tmdb_id": m_id,
                    "title": title,
                    "poster_url": make_poster(m.get("poster_path")),
                    "release_date": m.get("release_date", ""),
                    "vote_average": m.get("vote_average"),
                })
            return suggestions, cards

    # 2) Local Search
    if local_df is not None:
        q_lower = query.strip().lower()
        starts_mask = local_df["title"].astype(str).str.lower().str.startswith(q_lower)
        contains_mask = local_df["title"].astype(str).str.lower().str.contains(q_lower, regex=False)
        matched = pd.concat([local_df[starts_mask], local_df[contains_mask & ~starts_mask]]).head(limit)

        suggestions = []
        cards = []
        for idx, row in matched.iterrows():
            title = str(row["title"])
            year = str(row.get("release_date", ""))[:4]
            label = f"{title} ({year})" if year and year != "nan" else title
            suggestions.append((label, int(idx), title))
            cards.append({
                "tmdb_id": int(idx),
                "title": title,
                "poster_url": make_poster(row.get("poster_path")),
                "release_date": str(row.get("release_date", "")),
                "vote_average": float(row.get("vote_average", 0)) if pd.notna(row.get("vote_average")) else None,
            })
        return suggestions, cards

    return [], []


def get_movie_details(movie_id: int, title_hint: str = ""):
    key = get_tmdb_key()

    # 1) Live TMDB details
    if key and movie_id > 100:
        data = tmdb_live_get(f"/movie/{movie_id}", {"language": "en-US"})
        if data and "title" in data:
            return {
                "tmdb_id": int(data["id"]),
                "title": data.get("title", ""),
                "overview": data.get("overview") or "No overview available.",
                "release_date": data.get("release_date", "-"),
                "poster_url": make_poster(data.get("poster_path")),
                "backdrop_url": make_poster(data.get("backdrop_path")),
                "genres": data.get("genres", []),
                "vote_average": data.get("vote_average"),
                "popularity": data.get("popularity"),
            }, None

    # 2) Local lookup
    if local_df is not None:
        row = None
        if 0 <= movie_id < len(local_df):
            row = local_df.iloc[movie_id]
        elif title_hint:
            t_matches = local_df[local_df["title"].astype(str).str.lower() == title_hint.lower()]
            if not t_matches.empty:
                row = t_matches.iloc[0]

        if row is not None:
            genres_list = [{"name": g.strip()} for g in str(row.get("genres", "")).replace(",", " ").split() if len(g.strip()) > 1]
            poster = make_poster(row.get("poster_path"))
            vote = row.get("vote_average")
            return {
                "tmdb_id": int(movie_id),
                "title": str(row.get("title", "Untitled")),
                "overview": str(row.get("overview", "No overview available.")) if pd.notna(row.get("overview")) else "No overview available.",
                "release_date": str(row.get("release_date", "-")) if pd.notna(row.get("release_date")) else "-",
                "poster_url": poster,
                "backdrop_url": poster,
                "genres": genres_list,
                "vote_average": float(vote) if pd.notna(vote) else None,
            }, None

    return None, "Movie details not found"


def get_recommendations(movie_id: int, title: str = "", top_n: int = 15):
    """Computes TF-IDF cosine similarity recommendations from local matrix and enriches with live posters if TMDB key is active."""
    if local_df is None or local_matrix is None:
        return []

    # Find matrix index
    local_idx = None
    if 0 <= movie_id < len(local_df):
        local_idx = movie_id
    elif title:
        t_clean = title.strip().lower()
        exact = local_df[local_df["title"].astype(str).str.lower() == t_clean]
        if not exact.empty:
            local_idx = exact.index[0]
        else:
            sub = local_df[local_df["title"].astype(str).str.lower().str.contains(t_clean, regex=False)]
            if not sub.empty:
                local_idx = sub.index[0]

    if local_idx is None:
        return []

    qv = local_matrix[local_idx]
    scores = (local_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)

    key = get_tmdb_key()
    recs = []
    for i in order:
        if int(i) == int(local_idx):
            continue
        r_row = local_df.iloc[int(i)]
        r_title = str(r_row["title"])

        poster = make_poster(r_row.get("poster_path"))
        vote = r_row.get("vote_average")

        # If poster is missing and TMDB key is present, fetch live poster
        if not poster and key:
            t_data = tmdb_live_get("/search/movie", {"query": r_title, "language": "en-US", "page": 1})
            if t_data and t_data.get("results"):
                poster = make_poster(t_data["results"][0].get("poster_path"))

        recs.append({
            "tmdb_id": int(i),
            "title": r_title,
            "poster_url": poster,
            "vote_average": float(vote) if pd.notna(vote) else None,
            "score": float(scores[int(i)]),
        })
        if len(recs) >= top_n:
            break

    return recs


def poster_grid(cards, cols=6, key_prefix="grid"):
    if not cards:
        st.info("No movies found.")
        return

    rows = (len(cards) + cols - 1) // cols
    idx = 0
    for r in range(rows):
        colset = st.columns(cols)
        for c in range(cols):
            if idx >= len(cards):
                break
            m = cards[idx]
            idx += 1

            movie_id = m.get("tmdb_id")
            title = m.get("title", "Untitled")
            poster = m.get("poster_url")
            rating = m.get("vote_average")

            with colset[c]:
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.markdown(
                        "<div style='height:220px; background:#1f2937; border-radius:8px; display:flex; align-items:center; justify-content:center; color:#9ca3af; text-align:center; padding:10px;'>🎬 No Poster</div>",
                        unsafe_allow_html=True,
                    )

                if st.button("View", key=f"{key_prefix}_{r}_{c}_{idx}_{movie_id}", use_container_width=True):
                    if movie_id is not None:
                        goto_details(movie_id, title)

                st.markdown(f"<div class='movie-title'>{title}</div>", unsafe_allow_html=True)
                if rating:
                    st.caption(f"⭐ {rating:.1f}")


# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.markdown("## 🎬 Navigation")
    if st.button("🏠 Home", use_container_width=True):
        goto_home()

    st.markdown("---")
    st.markdown("### 🍿 Categories")
    home_category = st.selectbox(
        "Browse By",
        ["popular", "top_rated", "trending", "now_playing", "upcoming"],
        format_func=lambda x: x.replace("_", " ").title(),
        index=0,
    )
    grid_cols = st.slider("Grid Columns", 3, 7, 5)

    st.markdown("---")
    # TMDB API Key Input & Status
    st.markdown("### 🔑 TMDB Integration")
    user_key = st.text_input(
        "Enter TMDB API Key",
        value=st.session_state.tmdb_api_key,
        type="password",
        placeholder="Paste your TMDB v3 API Key...",
        help="Paste your free TMDB API Key here to activate live feeds, new releases, and live posters.",
    )

    if user_key != st.session_state.tmdb_api_key:
        st.session_state.tmdb_api_key = user_key.strip()
        # Save to .env so FastAPI backend can use it too
        with open(".env", "w", encoding="utf-8") as f:
            f.write(f"TMDB_API_KEY={user_key.strip()}\nAPI_BASE={st.session_state.api_base_url}\nPORT=8000\n")
        st.cache_data.clear()
        st.rerun()

    if get_tmdb_key():
        st.success("🟢 TMDB Live API: Connected")
    else:
        st.info("⚡ Mode: Local Data")
        st.caption("Paste your TMDB API Key above anytime to unlock live TMDB feeds and new releases.")


# =============================
# HEADER
# =============================
st.title("🎬 Movie Recommender System")
if get_tmdb_key():
    st.markdown(
        "<div class='small-muted'>Powered by <b>TMDB Live API</b> + <b>Local TF-IDF Vector Space</b> for accurate recommendations.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div class='small-muted'>Content-based filtering using <b>TF-IDF Cosine Similarity</b> on 45,000+ movies. Add your TMDB Key in the sidebar for live feeds.</div>",
        unsafe_allow_html=True,
    )
st.divider()


# ==========================================================
# VIEW: HOME
# ==========================================================
if st.session_state.view == "home":
    typed = st.text_input(
        "🔍 Search by movie title (e.g. Avatar, Inception, Toy Story, Batman...)",
        placeholder="Type a movie title...",
    )

    st.divider()

    # Search Mode
    if typed.strip():
        if len(typed.strip()) < 2:
            st.caption("Type at least 2 characters for suggestions.")
        else:
            suggestions, cards = search_movies(typed.strip(), limit=24)

            if suggestions:
                labels = ["-- Select a movie from suggestions --"] + [s[0] for s in suggestions]
                selected = st.selectbox("🎯 Instant Suggestions", labels, index=0)

                if selected != "-- Select a movie from suggestions --":
                    match = next((s for s in suggestions if s[0] == selected), None)
                    if match:
                        goto_details(match[1], match[2])

            st.markdown(f"### 🔎 Search Results for '{typed.strip()}'")
            poster_grid(cards, cols=grid_cols, key_prefix="search_results")

        st.stop()

    # Home Feed Mode
    st.markdown(f"### 🔥 {home_category.replace('_',' ').title()} Movies")
    home_cards, err = fetch_home_cards(home_category, limit=25)
    if err:
        st.error(f"Error loading movies: {err}")
    else:
        poster_grid(home_cards, cols=grid_cols, key_prefix="home_feed")


# ==========================================================
# VIEW: DETAILS & RECOMMENDATIONS
# ==========================================================
elif st.session_state.view == "details":
    m_id = st.session_state.selected_movie_id
    m_title = st.session_state.selected_movie_title or ""

    if m_id is None:
        st.warning("No movie selected.")
        if st.button("← Back to Home"):
            goto_home()
        st.stop()

    # Back bar
    col_a, col_b = st.columns([4, 1])
    with col_a:
        st.markdown("### 📄 Movie Details")
    with col_b:
        if st.button("← Back to Home", use_container_width=True):
            goto_home()

    movie, err = get_movie_details(m_id, title_hint=m_title)
    if err or not movie:
        st.error(f"Could not load details: {err}")
        st.stop()

    title_val = movie.get("title", m_title)

    # Layout: Poster Left, Metadata Right
    left, right = st.columns([1, 2.5], gap="large")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if movie.get("poster_url"):
            st.image(movie["poster_url"], use_container_width=True)
        else:
            st.markdown(
                "<div style='height:360px; background:#1f2937; border-radius:8px; display:flex; align-items:center; justify-content:center; color:#9ca3af;'>🎬 No Poster Available</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"## {title_val}")
        release = movie.get("release_date") or "-"
        rating = movie.get("vote_average")
        genres = ", ".join([g["name"] for g in movie.get("genres", [])]) or "General"

        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            st.markdown(f"**📅 Release Date:** `{release}`")
            st.markdown(f"**🎭 Genres:** `{genres}`")
        with col_meta2:
            if rating:
                st.markdown(f"**⭐ Rating:** `{rating:.1f} / 10`")

        st.markdown("---")
        st.markdown("### Overview")
        st.write(movie.get("overview") or "No synopsis available.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Recommendations Section
    st.markdown("### 🎯 Recommended Movies (Based on Content & TF-IDF Similarity)")
    recs = get_recommendations(m_id, title=title_val, top_n=15)
    if recs:
        poster_grid(recs, cols=grid_cols, key_prefix="details_recs")
    else:
        st.info("No similar recommendations found for this title.")
