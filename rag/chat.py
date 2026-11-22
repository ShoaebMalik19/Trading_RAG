from dataclasses import dataclass, field
from pathlib import Path

from .config import CHAT_MODEL, get_api_key
from .embeddings import MissingAPIKeyError
from .retriever import retrieve_with_scores

SYSTEM_PROMPT = """You are a professional SMC trading assistant.

You ONLY answer based on the provided context from SMC strategy documents
(BOS, liquidity grabs, FVGs, etc).

Rules:
- If the context is not enough, say you are not sure and ask for more details.
- Do NOT invent new strategies outside the given context.
- Explain in simple, practical language.
- Focus on structure, liquidity, and risk management.
"""

USER_TEMPLATE = """User trade idea / question:
{question}

Use the context to:
- Identify which SMC concepts are relevant (BOS, liquidity grab, FVG, etc).
- Explain how they apply.
- Mention risk considerations (SL placement, over-risking, etc).

Answer clearly in 1-3 short paragraphs.

Context:
{context}
"""


@dataclass
class Source:
    file: str
    content: str
    score: float

    @property
    def section(self) -> str:
        """First Markdown heading in the chunk, e.g. 'Entry Rules'."""
        for line in self.content.splitlines():
            if line.lstrip().startswith("#"):
                return line.lstrip("# ").strip()
        return ""

    @property
    def title(self) -> str:
        return self.file.removesuffix(".md").removeprefix("smc_").replace("_", " ").upper()


@dataclass
class RagResult:
    question: str
    answer: str | None
    sources: list[Source] = field(default_factory=list)
    error: str | None = None
    missing_key: bool = False


_client = None


def _get_client():
    global _client
    key = get_api_key()
    if not key:
        raise MissingAPIKeyError("answer generation")
    if _client is None:
        import google.genai as genai

        _client = genai.Client(api_key=key)
    return _client


def _format_context(chunks) -> str:
    if not chunks:
        return "No relevant context found in the knowledge base."
    return "\n\n---\n\n".join(doc.page_content for doc in chunks)


def _format_prompt(question: str, context: str) -> str:
    return USER_TEMPLATE.format(question=question, context=context)


def _call_llm(question: str, context: str) -> str:
    from google.genai import types

    response = _get_client().models.generate_content(
        model=CHAT_MODEL,
        contents=_format_prompt(question, context),
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    text = response.text
    if not text:
        raise RuntimeError("Gemini response did not include any text output.")
    return text


def retrieve(question: str, k: int = 4) -> list[Source]:
    return [
        Source(
            file=Path(doc.metadata.get("source", "unknown")).name,
            content=doc.page_content,
            score=float(score),
        )
        for doc, score in retrieve_with_scores(question, k=k)
    ]


def stream_answer(question: str, sources: list[Source]):
    """Yield the Gemini answer in pieces as it is generated.
    Raises MissingAPIKeyError before yielding anything when no key is set."""
    from google.genai import types

    client = _get_client()
    context = "\n\n---\n\n".join(s.content for s in sources) or "No relevant context found in the knowledge base."

    def gen():
        for chunk in client.models.generate_content_stream(
            model=CHAT_MODEL,
            contents=_format_prompt(question, context),
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        ):
            if chunk.text:
                yield chunk.text

    return gen()


def answer_question(question: str, k: int = 4) -> RagResult:
    """Retrieve notes, then generate a grounded answer. Never raises for a missing key:
    retrieval still runs and the result says the LLM step needs a key."""
    sources = retrieve(question, k=k)
    result = RagResult(question=question, answer=None, sources=sources)
    try:
        result.answer = _call_llm(question, "\n\n---\n\n".join(s.content for s in sources))
    except MissingAPIKeyError as exc:
        result.missing_key = True
        result.error = str(exc)
    except Exception as exc:  # surface API errors to the UI instead of crashing
        result.error = f"Gemini call failed: {exc}"
    return result


def ask_rag(question: str) -> str:
    """Backwards-compatible helper used by the CLI."""
    result = answer_question(question)
    if result.answer is None:
        raise RuntimeError(result.error)
    return result.answer
