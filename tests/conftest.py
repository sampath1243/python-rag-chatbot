
import os

# Dummy key for automated tests.
# No real OpenAI API calls will be made.
os.environ.setdefault(
    "OPENAI_API_KEY",
    "test-key-not-real"
)
