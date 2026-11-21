<div align="center">

# Trading RAG · SMC Assistant

**A RAG assistant (web demo + terminal) that answers Smart Money Concepts trading questions strictly from my own strategy notes, and shows the notes each answer came from.**

**Live demo:** _coming soon, see [Deploy](#-deploy-free)_

[![Python](https://img.shields.io/badge/Python_3.10+-1e2a3a?style=for-the-badge&logo=python&logoColor=FFD43B)](https://www.python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1a2e2a?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1a1a2e?style=for-the-badge&logoColor=white)](https://www.trychroma.com)
[![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-1a1f36?style=for-the-badge&logo=googlegemini&logoColor=8E75B2)](https://ai.google.dev)
[![RAG](https://img.shields.io/badge/RAG-Retrieval_Augmented-111827?style=for-the-badge)](#-how-it-works)

</div>

---

## ✨ Overview

Trading RAG is a web and command-line assistant for **Smart Money Concepts (SMC)**: Break of Structure (BOS), liquidity grabs, Fair Value Gaps (FVGs) and similar ideas. It uses Retrieval-Augmented Generation, so instead of relying on a model's training data it first searches the Markdown files in [`data/strategies/`](data/strategies), then asks Gemini to answer **only** from what it found. If the context isn't enough, it says so instead of guessing.

## 🚀 Features

- **Grounded answers:** a strict system prompt limits Gemini to the retrieved chunks and forbids inventing strategies.
- **Bring your own knowledge base:** drop `.md` strategy notes into `data/strategies/` and re-ingest. Ships with notes on BOS, FVG and liquidity grabs.
- **Web demo:** dark Streamlit UI that shows the answer plus the source note chunks and their relevance.
- **Works without a key for retrieval:** local embeddings (FastEmbed), so only the final answer needs Gemini.
- **Automatic first-run ingestion:** if the vector store is empty, the app builds it on start.
- **Local, persistent vector store:** ChromaDB on disk, so ingestion only runs once.
- **Risk-aware responses:** answers cover which SMC concepts apply, how, and risk considerations such as stop-loss placement.
- **Configurable models** through environment variables.

## 🧠 How It Works

```
You type a question
        │
        ▼
[Embeddings]     question → vector (local bge-small, or Gemini)
        │
        ▼
[ChromaDB]       similarity search → top 4 document chunks
        │
        ▼
[Prompt builder] question + retrieved chunks → strict prompt
        │
        ▼
[Gemini 2.5 Flash] answers only from the retrieved context
        │
        ▼
Answer + source notes (web UI or terminal)
```

1. **Ingestion:** every `.md` file in `data/strategies/` is split into overlapping chunks (600 characters, 80 overlap, split on headings first), embedded (locally by default), and saved to a local ChromaDB store.
2. **Retrieval:** the question is embedded with the same model and the 4 closest chunks are returned.
3. **Generation:** those chunks are injected into the prompt, and Gemini writes a short, practical answer.

## 🧱 Tech Stack

- **Language:** Python 3.10+
- **Web UI:** Streamlit (dark theme)
- **Orchestration:** LangChain (`langchain-community`, `langchain-text-splitters`, `langchain-chroma`)
- **Vector store:** ChromaDB
- **LLM:** Google GenAI SDK (`gemini-2.5-flash`)
- **Embeddings:** FastEmbed `BAAI/bge-small-en-v1.5` locally (default) or `gemini-embedding-001`
- **Config:** python-dotenv

## 🗂️ Project Structure

```
.
├── data/strategies/     # Knowledge base: smc_bos.md, smc_fvg.md, smc_liquidity_grab.md
├── rag/
│   ├── config.py        # Loads .env, model names and paths
│   ├── embeddings.py    # Local (FastEmbed) or Gemini embeddings for LangChain
│   ├── ingest.py        # Load docs → split → embed → save to ChromaDB
│   ├── retriever.py     # Load the persisted store and return a retriever
│   └── chat.py          # Retrieve context → build prompt → call Gemini
├── app.py               # Streamlit web demo: answer + source notes
├── main.py              # Terminal chat
├── Dockerfile           # Hugging Face Spaces / container deploy
├── .streamlit/          # Dark theme config
├── requirements.txt
└── .env.example
```

## ⚡ Run Locally

Requires Python 3.10+ (tested on 3.11).

```bash
git clone https://github.com/ShoaebMalik19/Trading_RAG.git
cd Trading_RAG
python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1    macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add GOOGLE_API_KEY (free: https://aistudio.google.com/apikey)
```

**Web demo (one command):**

```bash
streamlit run app.py
```

It opens http://localhost:8501 and builds the vector store on first start.

**Terminal chat:** `python main.py`, or one-shot: `python main.py "how do I trade a BOS retest?"`

**No key yet?** It still runs. Retrieval uses a small local embedding model (`BAAI/bge-small-en-v1.5` via FastEmbed, CPU only), so you can search the notes and see which chunks match. Only the written Gemini answer needs `GOOGLE_API_KEY`, and the UI says so instead of crashing.

### Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `GOOGLE_API_KEY` | *(none)* | Gemini key for answers (`GEMINI_API_KEY` also works) |
| `CHAT_MODEL` | `gemini-2.5-flash` | Model that writes the answers |
| `EMBEDDING_PROVIDER` | `local` | `local` (FastEmbed, no key) or `gemini` |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Used when `EMBEDDING_PROVIDER=gemini` |
| `VECTORSTORE_DIR` | `./vectorstore` | Where ChromaDB stores the database |

### Add your own notes

Put `.md` files in `data/strategies/`, then run `python -m rag.ingest`. It rebuilds the store without duplicating chunks.

## 🌐 Deploy (free)

### Option A: Streamlit Community Cloud (recommended, about 2 minutes)

1. Push the code to GitHub: `git push -u origin feat/demo` (or merge `feat/demo` into `main` and push).
2. Open [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, then **Create app → Deploy a public app from GitHub**.
3. Repository `ShoaebMalik19/Trading_RAG`, branch `main` (or `feat/demo`), main file path `app.py`.
4. **App URL:** choose a subdomain, for example `trading-rag-smc`.
5. **Advanced settings:** Python version `3.11`. In **Secrets** paste:
   ```toml
   GOOGLE_API_KEY = "your-key-here"
   ```
6. Click **Deploy**. The link is `https://trading-rag-smc.streamlit.app` (your chosen subdomain).

First boot downloads the ~70 MB embedding model and indexes the notes (under a minute). Free apps sleep after a period without visitors and wake on the next visit.

### Option B: Hugging Face Spaces (Docker)

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space): SDK **Docker**, template **Blank**, hardware **CPU basic (free)**.
2. In the Space: **Settings → Variables and secrets → New secret** named `GOOGLE_API_KEY`.
3. The Space's `README.md` must start with this header (add it on the branch you push to the Space):
   ```yaml
   ---
   title: Trading RAG SMC Assistant
   emoji: 📈
   colorFrom: green
   colorTo: gray
   sdk: docker
   app_port: 7860
   ---
   ```
   Then push:
   ```bash
   git remote add space https://huggingface.co/spaces/<hf-username>/trading-rag
   git push space feat/demo:main
   ```
4. The link is `https://huggingface.co/spaces/<hf-username>/trading-rag` (direct app: `https://<hf-username>-trading-rag.hf.space`).

The `Dockerfile` downloads the embedding model and builds the vector store at image build time, so the Space starts fast.

> This tool is for education and research. It is not financial advice.

## 👤 Author

**Mohammed Shoaeb Malik** · AI & Full-Stack Developer, Bengaluru

[![GitHub](https://img.shields.io/badge/GitHub-ShoaebMalik19-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ShoaebMalik19)
