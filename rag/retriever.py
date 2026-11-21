from langchain_chroma import Chroma

from .config import COLLECTION_NAME, VECTORSTORE_DIR
from .embeddings import get_embeddings

_db: Chroma | None = None


def get_vectorstore(build_if_missing: bool = True) -> Chroma:
    """
    Open the persisted Chroma collection. Builds it from data/strategies on first use,
    so a fresh clone or a fresh cloud container works with no manual step.
    """
    global _db
    if _db is not None:
        return _db

    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(VECTORSTORE_DIR),
    )
    if db._collection.count() == 0:
        if not build_if_missing:
            raise RuntimeError(
                f"Vector store is empty at '{VECTORSTORE_DIR}'. Run `python -m rag.ingest` first."
            )
        from .ingest import run_ingestion

        db = run_ingestion()
    _db = db
    return _db


def get_retriever(k: int = 4):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


def retrieve_with_scores(question: str, k: int = 4):
    """Return [(Document, relevance 0..1)] for the question."""
    return get_vectorstore().similarity_search_with_relevance_scores(question, k=k)
