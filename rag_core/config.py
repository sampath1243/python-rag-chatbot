from pathlib import Path


# Root folder of our project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folder containing our knowledge documents
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

# File types that our loader can read
SUPPORTED_EXTENSIONS = {".txt", ".md"}