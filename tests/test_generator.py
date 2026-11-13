
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag_core.generator import generate_answer


@patch("rag_core.generator.client")
def test_generate_answer_returns_llm_response(mock_client):
    mock_response = MagicMock()
    mock_response.output_text = "Python is a programming language."

    mock_client.responses.create.return_value = mock_response

    results = [
        {
            "chunk": SimpleNamespace(
                text="Python is a programming language."
            )
        }
    ]

    answer = generate_answer(
        "What is Python?",
        results
    )

    assert answer == "Python is a programming language."


@patch("rag_core.generator.client")
def test_generate_answer_sends_question_and_context(mock_client):
    mock_response = MagicMock()
    mock_response.output_text = "Python is easy."

    mock_client.responses.create.return_value = mock_response

    results = [
        {
            "chunk": SimpleNamespace(text="Python is easy to learn.")
        },
        {
            "chunk": SimpleNamespace(text="Python is widely used.")
        }
    ]

    generate_answer("Is Python easy?", results)

    call_args = mock_client.responses.create.call_args

    assert call_args.kwargs["model"] == "gpt-5.6"
    assert "Is Python easy?" in call_args.kwargs["input"]
    assert "Python is easy to learn." in call_args.kwargs["input"]
    assert "Python is widely used." in call_args.kwargs["input"]


@patch("rag_core.generator.client")
def test_generate_answer_uses_instructions(mock_client):
    mock_response = MagicMock()
    mock_response.output_text = "Information not available."

    mock_client.responses.create.return_value = mock_response

    generate_answer("What is Django?", [])

    instructions = mock_client.responses.create.call_args.kwargs[
        "instructions"
    ]

    assert "provided context" in instructions
    assert "not available" in instructions
