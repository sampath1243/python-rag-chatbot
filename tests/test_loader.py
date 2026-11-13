
import pytest

from rag_core.loader import load_documents


def test_load_documents_reads_text_files(tmp_path):
    knowledge_file = tmp_path / "sample.txt"

    knowledge_file.write_text(
        "Python is a programming language.",
        encoding="utf-8"
    )

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].text == "Python is a programming language."
    assert documents[0].source == "sample.txt"


def test_load_documents_ignores_unsupported_files(tmp_path):
    pdf_file = tmp_path / "sample.pdf"

    pdf_file.write_text(
        "This file should be ignored.",
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_documents(tmp_path)


def test_load_documents_raises_error_for_missing_folder(tmp_path):
    missing_folder = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        load_documents(missing_folder)


def test_load_documents_reads_markdown_files(tmp_path):
    markdown_file = tmp_path / "notes.md"

    markdown_file.write_text(
        "# Python Basics",
        encoding="utf-8"
    )

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].text == "# Python Basics"
    assert documents[0].source == "notes.md"
