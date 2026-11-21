from typing import List

from langchain_core.embeddings import Embeddings

from .config import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    LOCAL_EMBEDDING_MODEL,
    MODEL_CACHE_DIR,
    get_api_key,
)


class MissingAPIKeyError(RuntimeError):
    """Raised when a Gemini call is needed but no API key is configured."""

    def __init__(self, what: str = "this step"):
        super().__init__(
            f"A Gemini API key is required for {what}. Set GOOGLE_API_KEY (or GEMINI_API_KEY) "
            "in your .env file or hosting secrets. Get a free key at https://aistudio.google.com/apikey"
        )


class GeminiEmbeddings(Embeddings):
    def __init__(self, model: str):
        import google.genai as genai

        key = get_api_key()
        if not key:
            raise MissingAPIKeyError("Gemini embeddings")
        self.model = model
        self._client = genai.Client(api_key=key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        result = self._client.models.embed_content(
            model=self.model,
            contents=[text or " "],
        )
        return list(result.embeddings[0].values)


class LocalEmbeddings(Embeddings):
    """Small ONNX embedding model (FastEmbed). Runs on CPU, needs no API key."""

    def __init__(self, model: str):
        from fastembed import TextEmbedding

        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(model_name=model, cache_dir=str(MODEL_CACHE_DIR))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self._model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> List[float]:
        return next(iter(self._model.query_embed(text or " "))).tolist()


_cached: Embeddings | None = None


def get_embeddings() -> Embeddings:
    global _cached
    if _cached is None:
        if EMBEDDING_PROVIDER == "gemini":
            _cached = GeminiEmbeddings(model=EMBEDDING_MODEL)
        else:
            _cached = LocalEmbeddings(model=LOCAL_EMBEDDING_MODEL)
    return _cached
