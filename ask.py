"""Ask questions against the index built by ingest.py.

Usage:
    python ask.py "What is RAG?"          # show the top matching chunks
    python ask.py "What is RAG?" --answer # also generate an answer (needs OPENAI_API_KEY)
    python ask.py                         # interactive mode
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("index.json")
TOP_K = 3


def load_index():
    if not INDEX_PATH.exists():
        print("No index.json found. Run `python ingest.py` first.")
        sys.exit(1)
    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    records = data["records"]
    embeddings = np.array([r["embedding"] for r in records], dtype=np.float32)
    return data["model"], records, embeddings


def search(query: str, model: SentenceTransformer, records, embeddings, k: int = TOP_K):
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ q  # cosine similarity, since vectors are normalized
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), records[i]) for i in top]


def generate_answer(question: str, hits) -> str:
    """Send the retrieved chunks to an OpenAI model and return its answer."""
    if not os.environ.get("OPENAI_API_KEY"):
        return "(Skipped: set OPENAI_API_KEY to generate an answer.)"
    from openai import OpenAI

    client = OpenAI()
    context = "\n\n".join(
        f"[{Path(h['source']).name}]\n{h['text']}" for _, h in hits
    )
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using only the context below. "
                    "Cite the file names in square brackets. "
                    "If the context does not contain the answer, say so."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content


def show(question: str, model, records, embeddings, use_answer: bool) -> None:
    hits = search(question, model, records, embeddings)
    print("\nTop matches:")
    for score, rec in hits:
        preview = rec["text"][:200].replace("\n", " ")
        print(f"- [{score:.3f}] {rec['source']} (chunk {rec['chunk_id']}): {preview}...")

    if use_answer:
        print("\nAnswer:")
        print(generate_answer(question, hits))
    print()


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--answer"]
    use_answer = "--answer" in sys.argv[1:]

    model_name, records, embeddings = load_index()
    model = SentenceTransformer(model_name)

    if args:
        show(" ".join(args), model, records, embeddings, use_answer)
        return

    print("Ask a question (empty line to quit).")
    while True:
        question = input("> ").strip()
        if not question:
            break
        show(question, model, records, embeddings, use_answer)


if __name__ == "__main__":
    main()
