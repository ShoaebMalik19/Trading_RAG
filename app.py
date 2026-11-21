"""Trading RAG web demo.  Run:  streamlit run app.py"""

import html
import os

import streamlit as st

# Streamlit Cloud / HF secrets -> env vars, before rag.config reads them
try:
    for _k in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no secrets file locally: fine

from rag.chat import answer_question  # noqa: E402
from rag.config import CHAT_MODEL, DATA_DIR, EMBEDDING_PROVIDER, LOCAL_EMBEDDING_MODEL, has_api_key  # noqa: E402
from rag.retriever import get_vectorstore  # noqa: E402

st.set_page_config(page_title="Trading RAG · SMC Assistant", page_icon="📈", layout="wide")

EXAMPLES = [
    "How do I enter after a bullish break of structure?",
    "Where should my stop loss go on a liquidity grab trade?",
    "What makes a fair value gap worth trading?",
    "Price swept equal lows then closed back inside the range. Long?",
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stChatMessage { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; max-width: 1100px; }
[data-testid="stSidebar"] { background: #0f1318; border-right: 1px solid #1f262f; }
.mono { font-family: 'JetBrains Mono', monospace; }
.hero { border: 1px solid #1f262f; border-radius: 14px; padding: 22px 26px; margin-bottom: 18px;
  background: radial-gradient(1200px 200px at 0% 0%, rgba(22,199,132,.10), transparent 60%), #10151b; }
.hero h1 { font-size: 1.9rem; margin: 0 0 6px 0; letter-spacing: -.02em; }
.hero h1 span { color: #16c784; }
.hero p { color: #9aa4b2; margin: 0; }
.tape { display:flex; gap:8px; flex-wrap:wrap; margin-top: 14px; }
.tag { font-family:'JetBrains Mono',monospace; font-size:.72rem; padding:4px 9px; border-radius:6px;
  border:1px solid #243040; color:#c9d1d9; background:#0b0f14; }
.tag.up { color:#16c784; border-color:rgba(22,199,132,.35); }
.tag.dn { color:#ea3943; border-color:rgba(234,57,67,.35); }
.tag.am { color:#f0b90b; border-color:rgba(240,185,11,.35); }
.status { font-family:'JetBrains Mono',monospace; font-size:.8rem; padding:10px 12px; border-radius:8px;
  border:1px solid #243040; background:#0b0f14; margin-bottom: 8px; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px; }
.src { border:1px solid #1f262f; border-left:3px solid #16c784; border-radius:8px; padding:12px 14px;
  margin:8px 0; background:#0f1419; }
.src-head { display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:8px; }
.src-title { font-family:'JetBrains Mono',monospace; font-size:.78rem; color:#16c784; }
.src-title b { color:#e6e8eb; font-weight:600; }
.bar { width:90px; height:6px; background:#1d242d; border-radius:4px; overflow:hidden; }
.bar > i { display:block; height:100%; background:linear-gradient(90deg,#0e8f5f,#16c784); }
.src-body { color:#b4bdc8; font-size:.86rem; line-height:1.55; }
.src-h { color:#e6e8eb; font-weight:600; font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; }
.hero h1, .hero h1 * { color:#e6e8eb !important; }
.hero h1 span { color:#16c784 !important; }
.keybox { border:1px solid rgba(240,185,11,.4); background:rgba(240,185,11,.06); border-radius:10px;
  padding:14px 16px; color:#e8d7a2; font-size:.9rem; }
.keybox code { color:#f0b90b; }
.foot { color:#5f6b7a; font-size:.75rem; margin-top: 28px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Indexing strategy notes…")
def load_store():
    db = get_vectorstore()
    return db._collection.count()


def render_sources(sources):
    if not sources:
        st.caption("No matching notes found.")
        return
    for s in sources:
        pct = max(0, min(100, round(s.score * 100)))
        section = f" · {html.escape(s.section)}" if s.section else ""
        st.markdown(
            # Single line on purpose: blank lines or indentation would end Streamlit's HTML block
            '<div class="src"><div class="src-head">'
            f'<div class="src-title"><b>{html.escape(s.title)}</b>{section} '
            f'<span style="color:#5f6b7a">({html.escape(s.file)})</span></div>'
            '<div style="display:flex;align-items:center;gap:8px">'
            f'<span class="mono" style="font-size:.72rem;color:#9aa4b2">{pct}%</span>'
            f'<div class="bar"><i style="width:{pct}%"></i></div></div></div>'
            f'<div class="src-body">{chunk_html(s.content)}</div></div>',
            unsafe_allow_html=True,
        )


def chunk_html(text: str) -> str:
    """Note chunk -> compact HTML: sub-headings bold, bullets kept, no blank lines."""
    out = []
    for line in text.strip().splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            out.append(f'<span class="src-h">{html.escape(line.lstrip("# ").strip())}</span>')
        else:
            out.append(html.escape(line).replace(" ", "&nbsp;", len(line) - len(line.lstrip())))
    return "<br>".join(out)


def render_result(result):
    if result.answer:
        st.markdown(result.answer)
    elif result.missing_key:
        st.markdown(
            """<div class="keybox"><b>Add your Gemini key to get written answers.</b><br>
            Retrieval worked: the notes below are what the answer would be grounded in.
            Set <code>GOOGLE_API_KEY</code> in <code>.env</code> (local) or in your host's secrets.
            Free key: <a href="https://aistudio.google.com/apikey" target="_blank">aistudio.google.com/apikey</a></div>""",
            unsafe_allow_html=True,
        )
    else:
        st.error(result.error)
    with st.expander(f"Grounded in {len(result.sources)} note chunks", expanded=not result.answer):
        render_sources(result.sources)


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Engine")
    live = has_api_key()
    color, label = ("#16c784", "LIVE · Gemini answers") if live else ("#f0b90b", "RETRIEVAL ONLY · no key")
    st.markdown(f'<div class="status"><span class="dot" style="background:{color}"></span>{label}</div>', unsafe_allow_html=True)
    embed = LOCAL_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "local" else "gemini-embedding-001"
    st.markdown(
        f'<div class="status">LLM&nbsp;&nbsp;&nbsp;{CHAT_MODEL}<br>EMBED&nbsp;{embed}<br>STORE&nbsp;ChromaDB</div>',
        unsafe_allow_html=True,
    )
    k = st.slider("Chunks to retrieve (k)", 2, 8, 4)
    st.markdown("### 📚 Knowledge base")
    for f in sorted(DATA_DIR.glob("*.md")):
        st.markdown(f'<span class="tag">{html.escape(f.name)}</span>', unsafe_allow_html=True)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ---------- header ----------
st.markdown(
    """<div class="hero"><h1>Trading RAG · <span>SMC Assistant</span></h1>
    <p>Ask about Smart Money Concepts. Answers come only from my own strategy notes, with the exact sources shown.</p>
    <div class="tape"><span class="tag up">▲ BOS</span><span class="tag dn">▼ LIQUIDITY GRAB</span>
    <span class="tag am">◆ FVG</span><span class="tag">LangChain</span><span class="tag">ChromaDB</span><span class="tag">Gemini</span></div></div>""",
    unsafe_allow_html=True,
)

try:
    n_chunks = load_store()
except Exception as exc:
    st.error(f"Could not build the vector store: {exc}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

pending = None
examples_box = st.empty()
if not st.session_state.history:
    with examples_box.container():
        st.caption(f"{n_chunks} note chunks indexed. Try one:")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLES):
            if cols[i % 2].button(q, key=f"ex{i}", use_container_width=True):
                pending = q

for item in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(item.question)
    with st.chat_message("assistant", avatar="📈"):
        render_result(item)

typed = st.chat_input("Describe a setup or ask an SMC question…")
question = typed or pending
if question:
    examples_box.empty()
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="📈"):
        with st.spinner("Searching notes…"):
            result = answer_question(question, k=k)
        render_result(result)
    st.session_state.history.append(result)

st.markdown('<div class="foot">For education and research. Not financial advice.</div>', unsafe_allow_html=True)
