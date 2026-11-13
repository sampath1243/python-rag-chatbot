
import pytest

from rag_core.vector_store import VectorStore


def test_vector_store_adds_item():
    store = VectorStore()

    chunk = "Python is easy"
    vector = [1, 0, 0]

    store.add(chunk, vector)

    assert store.count() == 1


def test_vector_store_count():
    store = VectorStore()

    store.add("Chunk 1", [1, 0, 0])
    store.add("Chunk 2", [0, 1, 0])

    assert store.count() == 2


def test_cosine_similarity_identical_vectors():
    similarity = VectorStore.cosine_similarity(
        [1, 0, 0],
        [1, 0, 0]
    )

    assert similarity == pytest.approx(1.0)


def test_cosine_similarity_perpendicular_vectors():
    similarity = VectorStore.cosine_similarity(
        [1, 0, 0],
        [0, 1, 0]
    )

    assert similarity == pytest.approx(0.0)


def test_vector_store_search_returns_best_match():
    store = VectorStore()

    store.add("Python programming", [1, 0, 0])
    store.add("Java programming", [0, 1, 0])
    store.add("Python Django", [0.9, 0.1, 0])

    results = store.search(
        query_vector=[1, 0, 0],
        top_k=2
    )

    assert len(results) == 2
    assert results[0]["chunk"] == "Python programming"
    assert results[0]["score"] >= results[1]["score"]


def test_vector_store_search_respects_top_k():
    store = VectorStore()

    store.add("Chunk 1", [1, 0, 0])
    store.add("Chunk 2", [0, 1, 0])
    store.add("Chunk 3", [0, 0, 1])

    results = store.search(
        query_vector=[1, 0, 0],
        top_k=1
    )

    assert len(results) == 1
