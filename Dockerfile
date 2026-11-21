# Container image for Hugging Face Spaces (Docker SDK) or any container host.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE_DIR=/app/.cache/models

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Pre-download the embedding model and build the vector store at image build time
RUN python -m rag.ingest && chmod -R a+rwX /app

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
