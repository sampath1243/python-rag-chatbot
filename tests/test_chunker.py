
from types import SimpleNamespace

from rag_core.chunker import chunk_document


def test_chunk_document_splits_text():
    doc = SimpleNamespace(
        source="sample.txt",
        text="ABCDEFGHIJKLMN"
    )

    chunks = chunk_document(
        doc,
        chunk_size=5,
        overlap=2
    )

    assert len(chunks) == 5
    assert chunks[0].text == "ABCDE"
    assert chunks[1].text == "DEFGH"
    assert chunks[2].text == "GHIJK"
    assert chunks[3].text == "JKLMN"
    assert chunks[4].text == "MN"


def test_chunk_document_preserves_source():
    doc = SimpleNamespace(
        source="sample.txt",
        text="ABCDEFGHIJKLMN"
    )

    chunks = chunk_document(
        doc,
        chunk_size=5,
        overlap=2
    )

    for chunk in chunks:
        assert chunk.source == "sample.txt"


def test_chunk_document_rejects_zero_chunk_size():
    doc = SimpleNamespace(
        source="sample.txt",
        text="ABCDEFGHIJKLMN"
    )

    try:
        chunk_document(doc, chunk_size=0, overlap=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass



def test_chunk_document_rejects_negative_overlap():
    doc = SimpleNamespace(
        source="sample.txt",
        text="ABCDEFGHIJKLMN"
    )

    try:
        chunk_document(doc, chunk_size=5, overlap=-1)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_chunk_document_rejects_overlap_equal_to_chunk_size():
    doc = SimpleNamespace(
        source="sample.txt",
        text="ABCDEFGHIJKLMN"
    )

    try:
        chunk_document(doc, chunk_size=5, overlap=5)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_chunk_document_handles_empty_text():
    doc = SimpleNamespace(
        source="sample.txt",
        text=""
    )

    chunks = chunk_document(doc)

    assert chunks == []
