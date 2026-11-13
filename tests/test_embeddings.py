
from unittest.mock import MagicMock, patch

from rag_core.embeddings import create_embedding


@patch("rag_core.embeddings.client")
def test_create_embedding_returns_vector(mock_client):
    mock_response = MagicMock()

    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3])
    ]

    mock_client.embeddings.create.return_value = mock_response

    result = create_embedding("Python programming")

    assert result == [0.1, 0.2, 0.3]


@patch("rag_core.embeddings.client")
def test_create_embedding_uses_correct_model(mock_client):
    mock_response = MagicMock()

    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3])
    ]

    mock_client.embeddings.create.return_value = mock_response

    create_embedding("Python programming")

    mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input="Python programming"
    )
