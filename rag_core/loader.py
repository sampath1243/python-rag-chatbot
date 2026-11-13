from dataclasses import dataclass
from pathlib import Path

from .config import KNOWLEDGE_DIR, SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class Document:
    source: str
    text: str


def load_documents(folder: Path = KNOWLEDGE_DIR) -> list[Document]:

    if not folder.exists():
        raise FileNotFoundError(
            f"Knowledge folder does not exist: {folder}"
        )

    documents = []

    for path in sorted(folder.rglob("*")):

        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:

            try:
                text = path.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                text = path.read_text(encoding="cp1252").strip()

            if text:
                documents.append(
                    Document(
                        source=path.relative_to(folder).as_posix(),
                        text=text
                    )
                )

    if not documents:
        raise ValueError(
            f"No knowledge files found in: {folder}"
        )

    return documents