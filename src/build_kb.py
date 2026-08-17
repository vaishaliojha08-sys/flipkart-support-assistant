"""
Build the Flipkart-style policy RAG knowledge base.

This script:

1. Loads policy documents from data/policy_kb.json
2. Splits each document sentence-by-sentence
3. Embeds every chunk using all-MiniLM-L6-v2
4. Builds a FAISS cosine-similarity index
5. Saves the index and chunk metadata

No API key is required.
"""

from pathlib import Path
import json
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

KB_FILE = ROOT / "data" / "policy_kb.json"

INDEX_DIR = ROOT / "indexes"
INDEX_FILE = INDEX_DIR / "policy.index"
METADATA_FILE = INDEX_DIR / "metadata.json"

INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# SENTENCE CHUNKING
# ---------------------------------------------------------

def sentence_chunk(text):
    """
    Split a policy document into sentence-level chunks.
    """

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("BUILDING POLICY RAG INDEX")
    print("=" * 70)

    # Load documents
    with open(KB_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Policy documents loaded: {len(documents)}")

    # Create sentence-level chunks
    chunks = []

    for document in documents:

        sentences = sentence_chunk(document["text"])

        for sentence_number, sentence in enumerate(sentences):

            chunks.append({
                "chunk_id": f"{document['doc_id']}_CH{sentence_number + 1}",
                "doc_id": document["doc_id"],
                "title": document["title"],
                "text": sentence
            })

    print(f"Total chunks created: {len(chunks)}")

    # -----------------------------------------------------
    # LOAD EMBEDDING MODEL
    # -----------------------------------------------------

    print("\nLoading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    # -----------------------------------------------------
    # CREATE EMBEDDINGS
    # -----------------------------------------------------

    texts = [chunk["text"] for chunk in chunks]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = embeddings.astype("float32")

    print("Embedding shape:", embeddings.shape)

    # -----------------------------------------------------
    # BUILD FAISS INDEX
    # -----------------------------------------------------

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print("FAISS vectors:", index.ntotal)

    # -----------------------------------------------------
    # SAVE INDEX
    # -----------------------------------------------------

    faiss.write_index(index, str(INDEX_FILE))

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print("\nIndex saved:")
    print(INDEX_FILE)

    print("\nMetadata saved:")
    print(METADATA_FILE)

    print("\nBUILD COMPLETE")


if __name__ == "__main__":
    main()