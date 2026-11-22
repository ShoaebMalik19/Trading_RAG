"""Trading RAG web demo.  Run:  streamlit run app.py"""

import html
import json
import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# Streamlit Cloud / HF secrets -> env vars, before rag.config reads them
try:
    for _k in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no secrets file locally: fine

from rag.chat import RagResult, retrieve, stream_answer  # noqa: E402
from rag.config import CHAT_MODEL, DATA_DIR, EMBEDDING_PROVIDER, LOCAL_EMBEDDING_MODEL, has_api_key  # noqa: E402
from rag.embeddings import MissingAPIKeyError  # noqa: E402
from rag.retriever import get_vectorstore  # noqa: E402

st.set_page_config(page_title="Trading RAG · SMC Assistant", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")

UI = Path(__file__).parent / "ui"
GITHUB = "https://github.com/ShoaebMalik19/Trading_RAG"
EXAMPLES = [
    "How do I enter after a bullish break of structure?",
    "Where should my stop loss go on a liquidity grab trade?",
    "What makes a fair value gap worth trading?",
    "Price swept equal lows then closed back inside the range. Long?",
]
TONES = {"bos": "var(--up)", "liquidity": "var(--cyan)", "fvg": "var(--amber)"}

first_run = not st.session_state.get("booted")
st.session_state.booted = True
if "history" not in st.session_state:
    st.session_state.history = []


@st.cache_resource(show_spinner=False)
def load_store():
    return get_vectorstore()._collection.count()


@st.cache_data(show_spinner=False)
def read_ui(name: str) -> str:
    return (UI / name).read_text(encoding="utf-8")


def md(s: str):
    # Collapse to one line: blank lines / indentation would end Streamlit's HTML block
    st.markdown(" ".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def kinetic(text: str, start: int, cls: str = "") -> str:
    spans = "".join(f'<span class="ch {cls}" style="--i:{start + i}">{html.escape(c)}</span>' for i, c in enumerate(text))
    return f'<span class="row">{spans}</span>'


# ---------- styles + motion layer ----------
st.markdown(f"<style>{read_ui('styles.css')}</style>", unsafe_allow_html=True)
if first_run:
    md('<div class="smc-cover"></div>')

try:
    n_chunks = load_store()
except Exception as exc:
    st.error(f"Could not build the vector store: {exc}")
    st.stop()

live = has_api_key()
notes = sorted(DATA_DIR.glob("*.md"))
embed_name = LOCAL_EMBEDDING_MODEL.split("/")[-1] if EMBEDDING_PROVIDER == "local" else "gemini-embedding-001"
dims = 384 if EMBEDDING_PROVIDER == "local" else 3072
cfg = {"first": first_run, "notes": len(notes), "chunks": n_chunks, "embed": embed_name, "live": live, "llm": CHAT_MODEL}
components.html(
    "<script>(function(){var P=window.parent,D=P.document;"
    f"P.__SMC_CFG={json.dumps(cfg)};"
    "if(P.__smcFxReady){P.__smcFxEnhance&&P.__smcFxEnhance();return;}"
    f"var s=D.createElement('script');s.textContent={json.dumps(read_ui('fx.js'))};D.head.appendChild(s);"
    "})();</script>",
    height=0,
)

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Engine")
    color, label = ("#00e59b", "LIVE · Gemini answers") if live else ("#ffb020", "RETRIEVAL ONLY · no key")
    md(f'<div class="status"><span class="dot" style="background:{color}"></span>{label}</div>')
    md(f'<div class="status">LLM&nbsp;&nbsp;&nbsp;{CHAT_MODEL}<br>EMBED&nbsp;{embed_name}<br>STORE&nbsp;ChromaDB · {n_chunks} chunks</div>')
    k = st.slider("Chunks to retrieve (k)", 2, 8, 4)
    st.markdown("### 📚 Knowledge base")
    md("".join(f'<span class="tag">{html.escape(f.name)}</span>' for f in notes))
    if st.button("Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ---------- hero ----------
intro = "intro" if first_run else ""
base = "2.35s" if first_run else "0s"
md(f"""
<div class="hero">
  <div class="eyebrow fade-up" style="--d:{base}"><span class="live"></span><span class="scramble" data-delay="2.4">RETRIEVAL-AUGMENTED · SMART MONEY CONCEPTS</span></div>
  <div class="title {intro}" role="heading" aria-level="1" style="--d:{base}">{kinetic("Trading RAG", 0)}{kinetic("SMC Assistant", 11, "grad")}</div>
  <p class="sub fade-up" style="--d:calc({base} + .5s)">Ask about <b>break of structure</b>, <b>liquidity grabs</b> and <b>fair value gaps</b>.
  Every answer is grounded in my own trading notes, and you see exactly which notes it came from.<span class="caret"></span></p>
</div>
""")

tape_items = [("u", "▲ BREAK OF STRUCTURE"), ("d", "▼ LIQUIDITY GRAB"), ("a", "◆ FAIR VALUE GAP"), ("c", "$$$ EQUAL LOWS SWEPT"),
              ("", "WAIT FOR THE RETEST"), ("u", "RISK 0.5–1% PER TRADE"), ("", "SL BEYOND THE SWEEP WICK"),
              ("a", "TRADE WITH HTF BIAS"), ("d", "NO FVG TRADES IN CHOP")]
tape = "".join(f'<span class="{c}">{t}</span><span style="color:#2a333d">/</span>' for c, t in tape_items)
md(f'<div class="tape fade-up" style="--d:calc({base} + .7s)"><div class="tape-track">{tape}{tape}</div></div>')

status_tile = ('<div class="stat live"><div class="k">ENGINE</div><div class="v">● LIVE</div></div>' if live else
               '<div class="stat warn"><div class="k">ENGINE</div><div class="v">◐ RETRIEVAL</div></div>')
delay = 2.6 if first_run else 0
md(f"""
<div class="stats fade-up" style="--d:calc({base} + .85s)">
  <div class="stat" style="--d:0s"><div class="k">STRATEGY NOTES</div><div class="v"><span data-count="{len(notes)}" data-delay="{delay}">{len(notes)}</span></div></div>
  <div class="stat" style="--d:.6s"><div class="k">INDEXED CHUNKS</div><div class="v"><span data-count="{n_chunks}" data-delay="{delay}">{n_chunks}</span></div></div>
  <div class="stat" style="--d:1.2s"><div class="k">VECTOR DIMS</div><div class="v"><span data-count="{dims}" data-delay="{delay}">{dims}</span><small>d</small></div></div>
  {status_tile}
</div>
""")


# ---------- renderers ----------
def chunk_html(text: str) -> str:
    out = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            out.append(f'<span class="src-h">{html.escape(line.lstrip("# ").strip())}</span>')
        else:
            out.append(html.escape(line.rstrip()))
    return "<br>".join(out)


def render_sources(sources):
    if not sources:
        st.caption("No matching notes found.")
        return
    cards = []
    for i, s in enumerate(sources):
        pct = max(0, min(100, round(s.score * 100)))
        tone = next((v for key, v in TONES.items() if key in s.file), "var(--up)")
        section = f" · {html.escape(s.section)}" if s.section else ""
        cards.append(
            f'<div class="src" style="--i:{i};--tone:{tone}"><div class="rank">{i + 1:02d}</div>'
            f'<div class="src-head"><div class="src-title"><b>{html.escape(s.title)}</b>{section} '
            f'<span class="f">{html.escape(s.file)}</span></div>'
            f'<div class="meter">{pct}%<div class="bar"><i style="--w:{pct}%;--i:{i}"></i></div></div></div>'
            f'<div class="src-body">{chunk_html(s.content)}</div></div>'
        )
    md(f'<div class="srcs-h">▍ Grounded in {len(sources)} note chunks</div>' + "".join(cards))


def render_keybox():
    md("""<div class="keybox"><b>⚡ Plug in a Gemini key to get the written answer.</b><br>
    Retrieval already ran: the notes below are exactly what the answer would be grounded in.
    Set <code>GOOGLE_API_KEY</code> in <code>.env</code> or your host's secrets.
    Free key at <a href="https://aistudio.google.com/apikey" target="_blank">aistudio.google.com/apikey</a></div>""")


def render_result(r: RagResult):
    if r.answer:
        st.markdown(r.answer)
    elif r.missing_key:
        render_keybox()
    else:
        st.error(r.error)
    render_sources(r.sources)


# ---------- landing ----------
pending = None
landing = st.empty()
if not st.session_state.history:
    with landing.container():
        md(f'<div class="label fade-up" style="--d:calc({base} + 1s)">01 · Try a setup</div>')
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLES):
            if cols[i % 2].button(f"{i + 1:02d}   {q}", key=f"ex{i}", use_container_width=True):
                pending = q
        md(f"""
        <div class="label" style="margin-top:28px">02 · How it works</div>
        <div class="pipe">
          <div class="node" style="--n:0"><div class="ic">?</div><div class="t">QUESTION</div><div class="s">your setup</div></div>
          <div class="wire" style="--n:0"></div>
          <div class="node" style="--n:1"><div class="ic">∿</div><div class="t">EMBED</div><div class="s">{html.escape(embed_name)}</div></div>
          <div class="wire" style="--n:1"></div>
          <div class="node" style="--n:2"><div class="ic">◫</div><div class="t">CHROMADB</div><div class="s">top-k similar notes</div></div>
          <div class="wire" style="--n:2"></div>
          <div class="node" style="--n:3"><div class="ic">✦</div><div class="t">GEMINI</div><div class="s">{html.escape(CHAT_MODEL)}</div></div>
          <div class="wire" style="--n:3"></div>
          <div class="node" style="--n:4"><div class="ic">▲</div><div class="t">ANSWER</div><div class="s">+ cited sources</div></div>
        </div>
        """)

# ---------- chat ----------
for item in st.session_state.history:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(item.question)
    with st.chat_message("assistant", avatar="📈"):
        render_result(item)

typed = st.chat_input("Describe a setup or ask an SMC question…")
question = typed or pending
if question:
    landing.empty()
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="📈"):
        status = st.empty()
        with status.container():
            md(f'<div class="scan"><div>› embedding query with {html.escape(embed_name)}</div>'
               f'<div>› scanning {n_chunks} chunks in chromadb</div><div>› ranking top {k} by similarity</div></div>')
        sources = retrieve(question, k=k)
        result = RagResult(question=question, answer=None, sources=sources)
        answer_slot = st.empty()
        render_sources(sources)
        try:
            stream = stream_answer(question, sources)
            status.empty()
            result.answer = answer_slot.write_stream(stream)
        except MissingAPIKeyError as exc:
            status.empty()
            result.missing_key, result.error = True, str(exc)
            with answer_slot.container():
                render_keybox()
        except Exception as exc:
            status.empty()
            result.error = f"Gemini call failed: {exc}"
            answer_slot.error(result.error)
    st.session_state.history.append(result)

md(f'<div class="foot">BUILT BY MOHAMMED SHOAEB MALIK · <a href="{GITHUB}" target="_blank">SOURCE ON GITHUB</a> · '
   'FOR EDUCATION, NOT FINANCIAL ADVICE</div>')
