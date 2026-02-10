import os
import hashlib
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import chromadb
from chromadb.utils import embedding_functions
from docx import Document

DOC_PATH = Path("Philosophical robot interview.  .docx")
DB_DIR = "chroma_db"
COLLECTION = "darwin_single_doc"
EMBED_MODEL = "text-embedding-3-small"


def read_docx(path: Path) -> str:
    doc = Document(str(path))
    parts = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts)


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def stable_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    if not DOC_PATH.exists():
        raise SystemExit(f"Doc not found: {DOC_PATH.resolve()}")

    text = read_docx(DOC_PATH)
    chunks = chunk_text(text)

    if not chunks:
        raise SystemExit("No text extracted from the .docx (is it empty?)")

    # Embedding function (OpenAI)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBED_MODEL,
    )

    # Persistent vector DB
    client = chromadb.PersistentClient(path=DB_DIR)
    col = client.get_or_create_collection(name=COLLECTION, embedding_function=openai_ef)

    docs, metadatas, ids = [], [], []
    for i, chunk in enumerate(chunks):
        ids.append(stable_id(f"{DOC_PATH.resolve()}::{i}::{chunk[:200]}"))
        docs.append(chunk)
        metadatas.append(
            {
                "source_file": DOC_PATH.name,
                "chunk_index": i,
            }
        )

    col.upsert(documents=docs, metadatas=metadatas, ids=ids)
    print(f"✅ Ingested {len(ids)} chunks into {DB_DIR}/{COLLECTION}")


if __name__ == "__main__":
    main()
