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


def _is_busy(exc: Exception) -> bool:
    return getattr(exc, "code", None) in (429, 500, 503) or any(
        s in str(exc) for s in ("UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded", "high demand")
    )


class GeminiBusyError(RuntimeError):
    pass


_flash_cache: list[str] | None = None


def _flash_models(client) -> list[str]:
    """General-purpose Flash and Flash-Lite models this key can call, newest first."""
    global _flash_cache
    if _flash_cache is None:
        import re

        found = []
        try:
            for m in client.models.list():
                name = (m.name or "").removeprefix("models/")
                actions = getattr(m, "supported_actions", None) or []
                if actions and "generateContent" not in actions:
                    continue
                if re.fullmatch(r"gemini-\d+(\.\d+)?-flash(-lite)?", name):  # skips image/tts/live/preview variants
                    ver = tuple(int(x) for x in re.findall(r"\d+", name))
                    found.append((ver, "lite" not in name, name))
        except Exception:
            pass
        _flash_cache = [n for _, _, n in sorted(found, reverse=True)]
    return _flash_cache


def _with_model(call):
    """Run call(client, model) with resilience for a public demo:
    - transient overload (429/5xx): retry with backoff, then try other Flash models
    - retired model (404): switch to the newest available Flash model and remember it"""
    import time

    global _model
    client = _get_client()
    first = _model or CHAT_MODEL

    def candidates():
        yield first
        for m in _flash_models(client):  # only listed if the first model fails
            if m != first:
                yield m

    last_exc = None
    for i, model in enumerate(candidates()):
        if i >= 4:
            break
        for attempt in range(3 if i == 0 else 1):
            try:
                result = call(client, model)
                if model != first and _is_not_found(last_exc or Exception()):
                    _model = model  # configured model is retired: stick with the replacement
                return result
            except Exception as exc:
                last_exc = exc
                if _is_not_found(exc):
                    break
                if not _is_busy(exc):
                    raise
                if i == 0 and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
    if last_exc is not None and _is_busy(last_exc):
        raise GeminiBusyError("Gemini is overloaded right now. Please try again in a few seconds.") from last_exc
    raise last_exc


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
