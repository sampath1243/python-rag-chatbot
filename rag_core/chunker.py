from dataclasses import dataclass

from .loader import Document


@dataclass(frozen=True)
class Chunk:
    source: str
    chunk_id: int
    text: str


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[Chunk]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []

    start = 0
    chunk_id = 0

    while start < len(document.text):

        end = start + chunk_size

        text = document.text[start:end].strip()

        if text:
            chunks.append(
                Chunk(
                    source=document.source,
                    chunk_id=chunk_id,
                    text=text
                )
            )

        chunk_id += 1

        start += chunk_size - overlap

    return chunks

    