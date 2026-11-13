from .embeddings import create_embedding
from .vector_store import VectorStore


def retrieve(
    question: str,
    vector_store: VectorStore,
    top_k: int = 3
):
    query_vector = create_embedding(question)

    results = vector_store.search(
        query_vector,
        top_k=top_k
    )

    return results