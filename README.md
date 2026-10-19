<div align="center">

# Trading RAG · SMC Assistant

**A terminal chatbot that answers Smart Money Concepts trading questions strictly from your own strategy notes.**

[![Python](https://img.shields.io/badge/Python_3.10+-1e2a3a?style=for-the-badge&logo=python&logoColor=FFD43B)](https://www.python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1a2e2a?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1a1a2e?style=for-the-badge&logoColor=white)](https://www.trychroma.com)
[![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-1a1f36?style=for-the-badge&logo=googlegemini&logoColor=8E75B2)](https://ai.google.dev)
[![RAG](https://img.shields.io/badge/RAG-Retrieval_Augmented-111827?style=for-the-badge)](#-how-it-works)

</div>

---

## ✨ Overview

Trading RAG is a command-line assistant for **Smart Money Concepts (SMC)**: Break of Structure (BOS), liquidity grabs, Fair Value Gaps (FVGs) and similar ideas. It uses Retrieval-Augmented Generation, so instead of relying on a model's training data it first searches the Markdown files in [`data/strategies/`](data/strategies), then asks Gemini to answer **only** from what it found. If the context isn't enough, it says so instead of guessing.

## 🚀 Features

- **Grounded answers:** a strict system prompt limits Gemini to the retrieved chunks and forbids inventing strategies.
- **Bring your own knowledge base:** drop `.md` strategy notes into `data/strategies/` and re-ingest. Ships with notes on BOS, FVG and liquidity grabs.
- **Automatic first-run ingestion:** if no vector store exists, `main.py` builds one before starting the chat.
- **Local, persistent vector store:** ChromaDB on disk, so ingestion only runs once.
- **Risk-aware responses:** answers cover which SMC concepts apply, how, and risk considerations such as stop-loss placement.
- **Configurable models** through environment variables.

## 🧠 How It Works

```
You type a question
        │
        ▼
[Embeddings]     question → vector (gemini-embedding-001)
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
Answer printed in your terminal
```

1. **Ingestion:** every `.md` file in `data/strategies/` is split into overlapping chunks (600 characters, 80 overlap, split on headings first), embedded with Gemini, and saved to a local ChromaDB store.
2. **Retrieval:** the question is embedded with the same model and the 4 closest chunks are returned.
3. **Generation:** those chunks are injected into the prompt, and Gemini writes a short, practical answer.

## 🧱 Tech Stack

- **Language:** Python 3.10+
- **Orchestration:** LangChain (`langchain-community`, `langchain-text-splitters`, `langchain-chroma`)
- **Vector store:** ChromaDB
- **LLM & embeddings:** Google GenAI SDK (`gemini-2.5-flash`, `gemini-embedding-001`)
- **Config:** python-dotenv

## 🗂️ Project Structure

```
.
├── data/strategies/     # Knowledge base: smc_bos.md, smc_fvg.md, smc_liquidity_grab.md
├── rag/
│   ├── config.py        # Loads .env, model names and paths
│   ├── embeddings.py    # Gemini embeddings wrapped for LangChain
│   ├── ingest.py        # Load docs → split → embed → save to ChromaDB
│   ├── retriever.py     # Load the persisted store and return a retriever
│   └── chat.py          # Retrieve context → build prompt → call Gemini
├── main.py              # Entry point: ensures the vector store, runs the chat loop
├── requirements.txt
└── .env.example
```

## ⚡ Getting Started

### Prerequisites

- Python 3.10 or higher
- A Google AI Studio API key ([aistudio.google.com](https://aistudio.google.com))

### Setup

```bash
git clone https://github.com/ShoaebMalik19/Trading_RAG.git
cd Trading_RAG

python -m venv venv
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # then add your API key
```

### Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `GOOGLE_API_KEY` | *(none)* | **Required.** Google AI Studio key |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Model used to embed documents and queries |
| `CHAT_MODEL` | `gemini-2.5-flash` | Gemini model used to generate answers |
| `VECTORSTORE_DIR` | `./vectorstore` | Where ChromaDB stores the database |

### Run

```bash
python main.py
```

On the first run the bot builds the vector store, then starts the chat:

```
=== Trading RAG SMC Assistant ===

[INFO] No vector store found. Running ingestion now...
[INGEST] Loaded 3 documents from .../data/strategies
[INGEST] Split into 9 chunks
[INGEST] Vector store built and persisted.

Type your trade idea or question. Type 'exit' to quit.

You: what is a BOS and how do I enter after one?
```

Type `exit` or `quit` to stop.

### Add your own documents

1. Put a new `.md` file in `data/strategies/`.
2. Rebuild the vector store:

   ```bash
   python -m rag.ingest
   ```

3. Run `python main.py` again.

> This tool is for education and research. It is not financial advice.

## 👤 Author

**Mohammed Shoaeb Malik** · AI & Full-Stack Developer, Bengaluru

[![GitHub](https://img.shields.io/badge/GitHub-ShoaebMalik19-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ShoaebMalik19)
