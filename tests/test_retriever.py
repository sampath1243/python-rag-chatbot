
from unittest.mock import patch

from rag_core.retriever import retrieve
from rag_core.vector_store import VectorStore


def test_retrieve_returns_matching_chunks():
    store = VectorStore()

    store.add("Python basics", [1, 0, 0])
    store.add("Java basics", [0, 1, 0])

    with patch(
        "rag_core.retriever.create_embedding",
        return_value=[1, 0, 0]
    ) as mock_embedding:

        results = retrieve(
            "Explain Python",
            store,
            top_k=1
        )

    assert mock_embedding.called
    assert len(results) == 1
    assert results[0]["chunk"] == "Python basics"


def test_retrieve_passes_question_to_embedding():
    store = VectorStore()

    store.add("Python basics", [1, 0, 0])

    with patch(
        "rag_core.retriever.create_embedding",
        return_value=[1, 0, 0]
    ) as mock_embedding:

        retrieve("What is Python?", store)

    mock_embedding.assert_called_once_with("What is Python?")


def test_retrieve_respects_top_k():
    store = VectorStore()

    store.add("Chunk 1", [1, 0, 0])
    store.add("Chunk 2", [0, 1, 0])
    store.add("Chunk 3", [0, 0, 1])

    with patch(
        "rag_core.retriever.create_embedding",
        return_value=[1, 0, 0]
    ):

        results = retrieve(
            "Python question",
            store,
            top_k=2
        )

    assert len(results) == 2
