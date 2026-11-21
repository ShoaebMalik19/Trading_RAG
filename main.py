"""Terminal chat. For the web demo run: streamlit run app.py"""

import sys

from rag.chat import answer_question
from rag.config import has_api_key
from rag.retriever import get_vectorstore


def print_result(result):
    if result.answer:
        print("\nBot:", result.answer, "\n")
    else:
        print(f"\n[!] {result.error}\n")
    print("Sources:")
    for s in result.sources:
        preview = " ".join(s.content.split())[:110]
        print(f"  - {s.file}  (relevance {s.score:.2f})  {preview}...")
    print()


def chat_loop():
    print("Type your trade idea or question. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye")
            break
        if question.lower() in {"exit", "quit"}:
            print("Bot: Goodbye")
            break
        if question:
            print_result(answer_question(question))


def main():
    print("=== Trading RAG SMC Assistant ===\n")
    if not has_api_key():
        print("[!] No GOOGLE_API_KEY set: retrieval works, answers need a key (see .env.example).\n")
    get_vectorstore()  # builds the store on first run

    if len(sys.argv) > 1:  # one-shot: python main.py "what is a BOS?"
        print_result(answer_question(" ".join(sys.argv[1:])))
    else:
        chat_loop()


if __name__ == "__main__":
    main()
