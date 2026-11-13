
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import run


def test_main_runs_rag_pipeline():

    # Mock document
    document = SimpleNamespace(
        source="sample.txt",
        text="Python is a programming language."
    )

    # Mock chunk
    chunk = SimpleNamespace(
        source="sample.txt",
        chunk_id=0,
        text="Python is a programming language."
    )

    # Mock retrieved results
    results = [
        {
            "chunk": chunk,
            "score": 0.95
        }
    ]

    # Mock vector store
    mock_vector_store = MagicMock()

    # Mock all external components
    with patch("run.load_documents", return_value=[document]) as mock_loader, \
         patch("run.chunk_document", return_value=[chunk]) as mock_chunker, \
         patch("run.create_embedding", return_value=[0.1, 0.2, 0.3]) as mock_embedding, \
         patch("run.VectorStore", return_value=mock_vector_store), \
         patch("run.retrieve", return_value=results) as mock_retriever, \
         patch("run.generate_answer", return_value="Python is a programming language.") as mock_generator, \
         patch("builtins.input", return_value="What is Python?"):

        # Execute application
        run.main()

    # Verify each component was called
    mock_loader.assert_called_once()

    mock_chunker.assert_called_once_with(
        document,
        chunk_size=500,
        overlap=50
    )

    mock_embedding.assert_called_once_with(
        chunk.text
    )

    mock_vector_store.add.assert_called_once_with(
        chunk,
        [0.1, 0.2, 0.3]
    )

    mock_retriever.assert_called_once_with(
        "What is Python?",
        mock_vector_store,
        top_k=3
    )

    mock_generator.assert_called_once_with(
        "What is Python?",
        results
    )


def test_main_function_exists():

    assert callable(run.main)
