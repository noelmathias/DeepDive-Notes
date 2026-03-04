import os
import json
import faiss
import numpy as np


# ---------------------------------------------------
# Session Path Helper
# ---------------------------------------------------
def get_session_paths(session_id):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

    base_dir = os.path.join(PROJECT_ROOT, "data", "sessions", session_id)
    os.makedirs(base_dir, exist_ok=True)

    index_path = os.path.join(base_dir, "index.faiss")
    meta_path = os.path.join(base_dir, "metadata.json")

    return index_path, meta_path


# ---------------------------------------------------
# Load / Create Index
# ---------------------------------------------------
def load_index(dimension: int, session_id: str):
    index_path, _ = get_session_paths(session_id)

    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    else:
        return faiss.IndexFlatL2(dimension)


def save_index(index, session_id: str):
    index_path, _ = get_session_paths(session_id)
    faiss.write_index(index, index_path)


# ---------------------------------------------------
# Metadata
# ---------------------------------------------------
def load_metadata(session_id: str):
    _, meta_path = get_session_paths(session_id)

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_metadata(metadata, session_id: str):
    _, meta_path = get_session_paths(session_id)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


# ---------------------------------------------------
# Add Embeddings
# ---------------------------------------------------
def add_embeddings(embeddings: list, metadata_entries: list, session_id: str):

    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]

    index = load_index(dimension, session_id)
    metadata = load_metadata(session_id)

    index.add(embeddings)
    metadata.extend(metadata_entries)

    save_index(index, session_id)
    save_metadata(metadata, session_id)


# ---------------------------------------------------
# Search
# ---------------------------------------------------
def search_similar(query_embedding, session_id: str, top_k=3):

    index_path, _ = get_session_paths(session_id)

    if not os.path.exists(index_path):
        return []

    index = faiss.read_index(index_path)
    metadata = load_metadata(session_id)

    query_embedding = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(distances[0], indices[0]):
        if idx < len(metadata):
            results.append({
                "score": float(score),
                "metadata": metadata[idx]
            })

    return results