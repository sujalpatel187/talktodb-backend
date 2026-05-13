from sentence_transformers import SentenceTransformer
from typing import List
from app.config import get_settings

settings = get_settings()

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def get_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector for a single text string.

    Args:
        text: Input text to embed.

    Returns:
        List of floats of length embedding_dimension.
    """
    model = get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embedding vectors for a batch of texts.

    Args:
        texts: List of input strings.

    Returns:
        List of embedding vectors.
    """
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]
