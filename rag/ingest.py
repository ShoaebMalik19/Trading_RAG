from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import COLLECTION_NAME, DATA_DIR, VECTORSTORE_DIR
from .embeddings import get_embeddings


def load_documents():
    """
    Load all .md files from data/strategies as LangChain Documents.
    """
    docs = [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path)})
        for path in sorted(DATA_DIR.glob("*.md"))
    ]
    print(f"[INGEST] Loaded {len(docs)} documents from {DATA_DIR}")
    return docs


def split_documents(docs):
    """
    Split docs into smaller chunks with overlap for better retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n##", "\n#", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[INGEST] Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks):
    """
    Create (or rebuild from scratch) the Chroma collection with the given chunks.
    """
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()

    print(f"[INGEST] Building Chroma collection '{COLLECTION_NAME}' at {VECTORSTORE_DIR}")
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )
    # Re-running ingestion should replace the notes, not duplicate them
    vectordb.reset_collection()
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    vectordb.add_documents(chunks, ids=ids)
    print("[INGEST] Vector store built and persisted.")
    return vectordb


def run_ingestion():
    docs = load_documents()
    chunks = split_documents(docs)
    return build_vectorstore(chunks)


if __name__ == "__main__":
    run_ingestion()
