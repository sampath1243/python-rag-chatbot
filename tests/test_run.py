
from unittest.mock import patch

from run import main


@patch("run.generate_answer")
@patch("run.retrieve")
@patch("run.create_embedding")
@patch("run.load_documents")
def test_main_runs_rag_pipeline(
    mock_load_documents,
    mock_create_embedding,
    mock_retrieve,
    mock_generate_answer
):
    from rag_core.loader import Document
    from rag_core.chunker import Chunk

    # Arrange: mock document
    mock_load_documents.return_value = [
        Document(
            source="sample.txt",
            text="Python is a programming language."
        )
    ]

    mock_create_embedding.return_value = [0.1, 0.2, 0.3]

    mock_chunk = Chunk(
        source="sample.txt",
        chunk_id=0,
        text="Python is a programming language."
    )

    mock_retrieve.return_value = [
        {
            "chunk": mock_chunk,
            "score": 0.95
        }
    ]

    mock_generate_answer.return_value = (
        "Python is a programming language."
    )

    # Simulate one question followed by exit
    with patch(
        "builtins.input",
        side_effect=["What is Python?", "exit"]
    ):
        main()

    # Assert
    mock_load_documents.assert_called_once()
    mock_create_embedding.assert_called_once()
    mock_retrieve.assert_called_once()
    mock_generate_answer.assert_called_once()
