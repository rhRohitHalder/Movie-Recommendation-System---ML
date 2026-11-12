import os
import pickle
import re
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# =========================
# ENV & TMDB CONFIG
# =========================
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

# =========================
# PICKLE GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Dict[str, int] = {}


# =========================
# MODELS
# =========================
class MovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class MovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []
    vote_average: Optional[float] = None
    popularity: Optional[float] = None


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[MovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: MovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[MovieCard]


# =========================
# HELPERS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_poster_url(path: Any) -> Optional[str]:
    if pd.isna(path) or not path or str(path).strip() == "":
        return None
    path_str = str(path).strip()
    if path_str.startswith("http"):
        return path_str
    if not path_str.startswith("/"):
        path_str = "/" + path_str
    return f"{TMDB_IMG_500}{path_str}"


def parse_genres_to_list(genres_val: Any) -> List[dict]:
    if pd.isna(genres_val) or not genres_val:
        return []
    val_str = str(genres_val).strip()
    words = val_str.replace(",", " ").split()
    return [{"id": i, "name": w} for i, w in enumerate(words) if len(w) > 1]


def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    title_to_idx: Dict[str, int] = {}
    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        raise RuntimeError("indices.pkl must be dict or pandas Series-like")


def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    for path, name in [
        (DF_PATH, "df.pkl"),
        (INDICES_PATH, "indices.pkl"),
        (TFIDF_MATRIX_PATH, "tfidf_matrix.pkl"),
        (TFIDF_PATH, "tfidf.pkl"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required model artifact '{name}' not found at {path}. "
                "Ensure pickle files are present or run 'python train.py' to generate them."
            )

    print("[INFO] Loading model artifacts...")
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")

    print(f"[INFO] Model artifacts loaded successfully. Total movies: {len(df)}")


# =========================
# TMDB API ASYNC CLIENT
# =========================
def get_effective_tmdb_key(custom_key: Optional[str] = None) -> str:
    key = (custom_key or "").strip()
    if key:
        return key
    return os.getenv("TMDB_API_KEY", "").strip()


async def tmdb_get(path: str, params: Dict[str, Any], api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    key = get_effective_tmdb_key(api_key)
    if not key:
        return None

    q = dict(params)
    q["api_key"] = key

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None


async def attach_tmdb_card_by_title(title: str, api_key: Optional[str] = None) -> Optional[MovieCard]:
    """Tries TMDB live search first for fresh poster and details, otherwise falls back to local dataset."""
    key = get_effective_tmdb_key(api_key)
    if key:
        data = await tmdb_get("/search/movie", {"query": title, "language": "en-US", "page": 1}, api_key=key)
        if data and data.get("results"):
            m = data["results"][0]
            return MovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or title,
                poster_url=make_poster_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )

    # Local fallback
    if df is not None:
        key_norm = _norm_title(title)
        if key_norm in TITLE_TO_IDX:
            row = df.iloc[TITLE_TO_IDX[key_norm]]
            return row_to_card(TITLE_TO_IDX[key_norm], row)
    return None


# =========================
# STARTUP LIFESPAN
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_pickles()
    yield


# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="Movie Recommender API (TMDB + Local TF-IDF)",
    version="3.0",
    description="Content-based Movie Recommendation System supporting both TMDB live API and local metadata",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_local_idx_by_title(title: str) -> int:
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    for t, idx in TITLE_TO_IDX.items():
        if key in t:
            return idx
    raise HTTPException(status_code=404, detail=f"Title not found in dataset: '{title}'")


def row_to_card(idx: int, row: pd.Series) -> MovieCard:
    vote = row.get("vote_average")
    vote_val = float(vote) if pd.notna(vote) else None
    return MovieCard(
        tmdb_id=int(idx),
        title=str(row.get("title", "Untitled")),
        poster_url=make_poster_url(row.get("poster_path")),
        release_date=str(row.get("release_date")) if pd.notna(row.get("release_date")) else None,
        vote_average=vote_val,
    )


def row_to_details(idx: int, row: pd.Series) -> MovieDetails:
    vote = row.get("vote_average")
    vote_val = float(vote) if pd.notna(vote) else None
    pop = row.get("popularity")
    pop_val = float(pop) if pd.notna(pop) else None
    poster = make_poster_url(row.get("poster_path"))

    return MovieDetails(
        tmdb_id=int(idx),
        title=str(row.get("title", "Untitled")),
        overview=str(row.get("overview", "No overview available.")) if pd.notna(row.get("overview")) else "No overview available.",
        release_date=str(row.get("release_date")) if pd.notna(row.get("release_date")) else None,
        poster_url=poster,
        backdrop_url=poster,
        genres=parse_genres_to_list(row.get("genres")),
        vote_average=vote_val,
        popularity=pop_val,
    )


def tfidf_recommend_indices(idx: int, top_n: int = 10) -> List[Tuple[int, float]]:
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)

    out: List[Tuple[int, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        out.append((int(i), float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


# =========================
# ROUTES
# =========================
@app.get("/")
def root():
    effective_key = get_effective_tmdb_key()
    return {
        "message": "Movie Recommender API (TMDB + Local TF-IDF)",
        "version": "3.0",
        "tmdb_active": bool(effective_key),
        "docs": "/docs",
        "health": "/health",
        "total_movies": len(df) if df is not None else 0,
    }


@app.get("/health")
def health():
    effective_key = get_effective_tmdb_key()
    return {
        "status": "ok",
        "tmdb_configured": bool(effective_key),
        "model_loaded": df is not None and tfidf_matrix is not None,
        "total_movies": len(df) if df is not None else 0,
    }


# ---------- HOME FEED (LIVE TMDB WITH LOCAL FALLBACK) ----------
@app.get("/home", response_model=List[MovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
    tmdb_key: Optional[str] = None,
):
    key = get_effective_tmdb_key(tmdb_key)

    # 1) Try Live TMDB API
    if key:
        tmdb_path = "/trending/movie/day" if category == "trending" else f"/movie/{category}"
        data = await tmdb_get(tmdb_path, {"language": "en-US", "page": 1}, api_key=key)
        if data and data.get("results"):
            cards: List[MovieCard] = []
            for m in data["results"][:limit]:
                cards.append(MovieCard(
                    tmdb_id=int(m["id"]),
                    title=m.get("title") or m.get("name") or "Untitled",
                    poster_url=make_poster_url(m.get("poster_path")),
                    release_date=m.get("release_date"),
                    vote_average=m.get("vote_average"),
                ))
            return cards

    # 2) Fallback to Local Dataset
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    sub = df.copy()
    if category in ("popular", "trending"):
        if "popularity" in sub.columns:
            sub["_pop"] = pd.to_numeric(sub["popularity"], errors="coerce").fillna(0)
            sub = sub.sort_values(by="_pop", ascending=False)
    elif category == "top_rated":
        if "vote_average" in sub.columns:
            sub["_vote"] = pd.to_numeric(sub["vote_average"], errors="coerce").fillna(0)
            sub = sub[sub["_vote"] > 0].sort_values(by="_vote", ascending=False)
    elif category in ("upcoming", "now_playing", "recent"):
        if "release_date" in sub.columns:
            sub["_date"] = pd.to_datetime(sub["release_date"], errors="coerce")
            sub = sub.dropna(subset=["_date"]).sort_values(by="_date", ascending=False)

    return [row_to_card(idx, row) for idx, row in sub.head(limit).iterrows()]


# ---------- SEARCH (LIVE TMDB WITH LOCAL FALLBACK) ----------
@app.get("/tmdb/search")
async def search_endpoint(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
    tmdb_key: Optional[str] = None,
):
    key = get_effective_tmdb_key(tmdb_key)

    # 1) Try Live TMDB Search
    if key:
        data = await tmdb_get("/search/movie", {"query": query, "include_adult": "false", "language": "en-US", "page": page}, api_key=key)
        if data and data.get("results"):
            return data

    # 2) Local Search Fallback
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    q_lower = query.strip().lower()
    starts_mask = df["title"].astype(str).str.lower().str.startswith(q_lower)
    contains_mask = df["title"].astype(str).str.lower().str.contains(q_lower, regex=False)
    matched = pd.concat([df[starts_mask], df[contains_mask & ~starts_mask]]).head(30)

    results = []
    for idx, row in matched.iterrows():
        poster_p = row.get("poster_path")
        poster_str = str(poster_p) if pd.notna(poster_p) and str(poster_p).strip() != "" else None
        results.append({
            "id": int(idx),
            "title": str(row["title"]),
            "poster_path": poster_str,
            "release_date": str(row.get("release_date")) if pd.notna(row.get("release_date")) else "",
            "vote_average": float(row.get("vote_average", 0)) if pd.notna(row.get("vote_average")) else None,
        })
    return {"results": results}


# ---------- MOVIE DETAILS ----------
@app.get("/movie/id/{movie_id}", response_model=MovieDetails)
async def movie_details_route(movie_id: int, tmdb_key: Optional[str] = None):
    key = get_effective_tmdb_key(tmdb_key)

    # 1) Try TMDB API by ID
    if key and movie_id > 100:  # TMDB IDs are typically large integers
        data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"}, api_key=key)
        if data and "title" in data:
            return MovieDetails(
                tmdb_id=int(data["id"]),
                title=data.get("title", ""),
                overview=data.get("overview"),
                release_date=data.get("release_date"),
                poster_url=make_poster_url(data.get("poster_path")),
                backdrop_url=make_poster_url(data.get("backdrop_path")),
                genres=data.get("genres", []),
                vote_average=data.get("vote_average"),
                popularity=data.get("popularity"),
            )

    # 2) Fallback to Local Dataset
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    if 0 <= movie_id < len(df):
        row = df.iloc[movie_id]
        return row_to_details(movie_id, row)

    raise HTTPException(status_code=404, detail="Movie not found")


# ---------- TF-IDF SIMILAR MOVIES ----------
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
    tmdb_key: Optional[str] = None,
):
    idx = get_local_idx_by_title(title)
    rec_pairs = tfidf_recommend_indices(idx, top_n=top_n)

    out = []
    for r_idx, score in rec_pairs:
        row = df.iloc[r_idx]
        card = await attach_tmdb_card_by_title(str(row["title"]), api_key=tmdb_key)
        out.append({
            "title": str(row["title"]),
            "score": score,
            "movie_id": r_idx,
            "poster_url": card.poster_url if card else make_poster_url(row.get("poster_path")),
        })
    return out


# ---------- BUNDLE: Details + TF-IDF Recs + Genre Recs ----------
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
    tmdb_key: Optional[str] = None,
):
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    key = get_effective_tmdb_key(tmdb_key)
    details: Optional[MovieDetails] = None

    # Search TMDB first if key is present
    if key:
        search_data = await tmdb_get("/search/movie", {"query": query, "language": "en-US", "page": 1}, api_key=key)
        if search_data and search_data.get("results"):
            best = search_data["results"][0]
            tmdb_id = int(best["id"])
            d_data = await tmdb_get(f"/movie/{tmdb_id}", {"language": "en-US"}, api_key=key)
            if d_data:
                details = MovieDetails(
                    tmdb_id=tmdb_id,
                    title=d_data.get("title", ""),
                    overview=d_data.get("overview"),
                    release_date=d_data.get("release_date"),
                    poster_url=make_poster_url(d_data.get("poster_path")),
                    backdrop_url=make_poster_url(d_data.get("backdrop_path")),
                    genres=d_data.get("genres", []),
                    vote_average=d_data.get("vote_average"),
                    popularity=d_data.get("popularity"),
                )

    # Local fallback for details
    if details is None:
        idx = get_local_idx_by_title(query)
        row = df.iloc[idx]
        details = row_to_details(idx, row)

    # 1) TF-IDF recommendations
    tfidf_items: List[TFIDFRecItem] = []
    try:
        local_idx = get_local_idx_by_title(details.title)
    except Exception:
        try:
            local_idx = get_local_idx_by_title(query)
        except Exception:
            local_idx = 0

    rec_pairs = tfidf_recommend_indices(local_idx, top_n=tfidf_top_n)
    for r_idx, score in rec_pairs:
        r_row = df.iloc[r_idx]
        card = await attach_tmdb_card_by_title(str(r_row["title"]), api_key=key)
        if card is None:
            card = row_to_card(r_idx, r_row)
        tfidf_items.append(TFIDFRecItem(title=card.title, score=score, tmdb=card))

    # 2) Genre recommendations (Live TMDB discover or local)
    genre_cards: List[MovieCard] = []
    if key and details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get("/discover/movie", {"with_genres": genre_id, "language": "en-US", "sort_by": "popularity.desc"}, api_key=key)
        if discover and discover.get("results"):
            for m in discover["results"][:genre_limit]:
                if int(m["id"]) != details.tmdb_id:
                    genre_cards.append(MovieCard(
                        tmdb_id=int(m["id"]),
                        title=m.get("title") or m.get("name") or "Untitled",
                        poster_url=make_poster_url(m.get("poster_path")),
                        release_date=m.get("release_date"),
                        vote_average=m.get("vote_average"),
                    ))

    if not genre_cards and df is not None:
        target_genre = details.genres[0]["name"].lower() if details.genres else ""
        if target_genre:
            g_matches = df[df["genres"].astype(str).str.lower().str.contains(target_genre, regex=False)].head(genre_limit)
            genre_cards = [row_to_card(g_idx, g_row) for g_idx, g_row in g_matches.iterrows()]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_cards,
    )
