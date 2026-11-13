
from pathlib import Path
from dotenv import load_dotenv

# Root folder of our project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load environment variables from the project's .env file
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

# Folder containing our knowledge documents
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

# File types that our loader can read
SUPPORTED_EXTENSIONS = {".txt", ".md"}
