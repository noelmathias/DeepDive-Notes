from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once globally (important for performance)
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embedding(text: str) -> np.ndarray:
    """
    Generate embedding vector for given text.
    Returns numpy array of shape (384,)
    """
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding