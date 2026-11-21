import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env (no-op when the file does not exist, e.g. on a hosted demo)
load_dotenv()


def get_api_key() -> str | None:
    """Gemini key from GOOGLE_API_KEY or GEMINI_API_KEY. Read at call time so a
    key added to the environment / Streamlit secrets is picked up without code changes."""
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if key and key.strip() and key.strip() != "API_KEY":
        return key.strip()
    return None


def has_api_key() -> bool:
    return get_api_key() is not None


# Embeddings: "local" (FastEmbed / ONNX, no key needed) or "gemini"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "strategies"
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", str(BASE_DIR / "vectorstore")))
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", str(BASE_DIR / ".cache" / "models")))

# One collection per embedding provider, so switching providers never mixes vector sizes
COLLECTION_NAME = f"smc_strategies_{EMBEDDING_PROVIDER}"
