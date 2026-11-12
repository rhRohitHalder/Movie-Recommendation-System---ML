"""
Movie Recommendation Model Training & Pipeline Serialization

Reads raw 'movies_metadata.csv', cleans and preprocesses text metadata,
fits TF-IDF vectorizer, and serializes model artifacts for FastAPI and Streamlit:
  - df.pkl: Processed DataFrame containing movie metadata
  - indices.pkl: Title to DataFrame index mapping
  - tfidf.pkl: Fitted TfidfVectorizer
  - tfidf_matrix.pkl: Sparse TF-IDF matrix (for fast cosine similarity)
"""

import ast
import os
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATASET_PATH = "movies_metadata.csv"


def parse_genres(genre_str):
    """Safely extracts genre names from stringified JSON/list."""
    if pd.isna(genre_str) or not genre_str:
        return ""
    try:
        data = ast.literal_eval(genre_str) if isinstance(genre_str, str) else genre_str
        if isinstance(data, list):
            return " ".join([i["name"] for i in data if isinstance(i, dict) and "name" in i])
        return ""
    except Exception:
        return ""


def clean_text(text: str) -> str:
    """Basic text normalization (lowercase, alphanumeric, normalized whitespace)."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return " ".join(text.split())


def train_and_export():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset '{DATASET_PATH}' not found. Please provide the dataset file.")

    print(f"[1/5] Loading {DATASET_PATH}...")
    df_raw = pd.read_csv(DATASET_PATH, low_memory=False)
    print(f"      Loaded {len(df_raw)} raw rows.")

    print("[2/5] Cleaning and preprocessing metadata...")
    # Drop rows without title
    df = df_raw.dropna(subset=["title"]).copy()
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

    # Fill NA and format fields
    df["overview"] = df["overview"].fillna("").astype(str)
    df["tagline"] = df["tagline"].fillna("").astype(str)
    df["genres_clean"] = df["genres"].apply(parse_genres)

    # Combine text features
    df["tags"] = df["overview"] + " " + df["genres_clean"] + " " + df["tagline"]
    df["tags"] = df["tags"].apply(clean_text)

    # Keep essential columns for production serving
    keep_cols = [
        "title",
        "overview",
        "tagline",
        "genres",
        "genres_clean",
        "vote_average",
        "vote_count",
        "popularity",
        "release_date",
        "poster_path",
    ]
    available_cols = [c for c in keep_cols if c in df.columns]
    df_clean = df[available_cols].copy()

    print(f"      Processed {len(df_clean)} unique movies.")

    print("[3/5] Fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    tfidf_matrix = tfidf.fit_transform(df["tags"])
    print(f"      TF-IDF Matrix shape: {tfidf_matrix.shape}")

    print("[4/5] Building index mapping...")
    indices = pd.Series(df_clean.index, index=df_clean["title"]).drop_duplicates()

    print("[5/5] Serializing artifacts to disk...")
    with open("df.pkl", "wb") as f:
        pickle.dump(df_clean, f)
    with open("indices.pkl", "wb") as f:
        pickle.dump(indices, f)
    with open("tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open("tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)

    print("[SUCCESS] All model artifacts saved successfully:")
    for fn in ["df.pkl", "indices.pkl", "tfidf.pkl", "tfidf_matrix.pkl"]:
        sz_mb = os.path.getsize(fn) / (1024 * 1024)
        print(f"  - {fn}: {sz_mb:.2f} MB")


if __name__ == "__main__":
    train_and_export()

