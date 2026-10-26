"""Read notes, split them into chunks, embed them, and save an index to index.json."""

import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

NOTES_DIR = Path("notes")
INDEX_PATH = Path("index.json")
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_CHARS = 800
OVERLAP_CHARS = 150


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Group paragraphs into chunks of roughly `size` characters, with a small overlap."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # Very long paragraphs get split by characters so no chunk is too big.
        while len(para) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:size])
            para = para[size - overlap:]

        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{para}" if tail else para

    if current.strip():
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


def main() -> None:
    if not NOTES_DIR.exists():
        print(f"Folder '{NOTES_DIR}' not found. Create it and add .md or .txt files.")
        sys.exit(1)

    files = sorted(NOTES_DIR.rglob("*.md")) + sorted(NOTES_DIR.rglob("*.txt"))
    if not files:
        print(f"No .md or .txt files found in '{NOTES_DIR}'.")
        sys.exit(1)

    records = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            records.append({"source": str(path), "chunk_id": i, "text": chunk})

    print(f"Embedding {len(records)} chunks from {len(files)} files with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        [r["text"] for r in records],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for record, emb in zip(records, embeddings):
        record["embedding"] = emb.tolist()

    INDEX_PATH.write_text(
        json.dumps({"model": MODEL_NAME, "records": records}),
        encoding="utf-8",
    )
    print(f"Saved index to {INDEX_PATH}")


if __name__ == "__main__":
    main()
