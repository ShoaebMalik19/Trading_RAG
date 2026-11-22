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


_model: str | None = None  # the model that actually works for this key (resolved lazily)


def _is_not_found(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 404 or "NOT_FOUND" in str(exc)


def _newest_flash(client) -> str | None:
    """Pick the newest general-purpose Flash model this key can call."""
    import re

    best, best_ver = None, ()
    for m in client.models.list():
        name = (m.name or "").removeprefix("models/")
        actions = getattr(m, "supported_actions", None) or []
        if actions and "generateContent" not in actions:
            continue
        if not re.fullmatch(r"gemini-\d+(\.\d+)?-flash", name):  # skips lite/image/tts/live/preview variants
            continue
        ver = tuple(int(x) for x in re.findall(r"\d+", name))
        if ver > best_ver:
            best, best_ver = name, ver
    return best


def _with_model(call):
    """Run call(model). If the configured model was retired (404), switch to the newest Flash once."""
    global _model
    client = _get_client()
    try:
        return call(client, _model or CHAT_MODEL)
    except Exception as exc:
        if not _is_not_found(exc) or _model:
            raise
        fallback = _newest_flash(client)
        if not fallback or fallback == CHAT_MODEL:
            raise
        _model = fallback
        return call(client, _model)


def active_model() -> str:
    return _model or CHAT_MODEL


def _call_llm(question: str, context: str) -> str:
    from google.genai import types

    response = _with_model(lambda client, model: client.models.generate_content(
        model=model,
        contents=_format_prompt(question, context),
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    ))
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

    _get_client()  # raises MissingAPIKeyError up front
    context = "\n\n---\n\n".join(s.content for s in sources) or "No relevant context found in the knowledge base."

    def open_stream(client, model):
        it = iter(client.models.generate_content_stream(
            model=model,
            contents=_format_prompt(question, context),
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        ))
        first = next(it, None)  # surfaces a 404 here, so the fallback can retry
        return first, it

    def gen():
        first, it = _with_model(open_stream)
        for chunk in ([first] if first is not None else []):
            if chunk.text:
                yield chunk.text
        for chunk in it:
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
