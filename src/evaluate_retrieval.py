"""
Part 3 - Retrieval evaluation.

Evaluation is performed at the DOCUMENT level.

Each query has an expected set of relevant document IDs.

Retrieved chunks are mapped back to their parent document
and deduplicated before Precision@3 and Recall@3 are calculated.
"""

from pathlib import Path
import json

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

INDEX_FILE = ROOT / "indexes" / "policy.index"
METADATA_FILE = ROOT / "indexes" / "metadata.json"

OUTPUT_FILE = ROOT / "retrieval_evaluation.txt"

MODEL_NAME = "all-MiniLM-L6-v2"


# =========================================================
# EVALUATION QUERIES
# =========================================================

EVALUATION_QUERIES = [

    {
        "query": "How many days do I have to return footwear?",
        "relevant_docs": ["POL002"]
    },

    {
        "query": "What is the return period for electronics?",
        "relevant_docs": ["POL003"]
    },

    {
        "query": "How long does a COD refund take?",
        "relevant_docs": ["POL005"]
    },

    {
        "query": "Can my returned product be collected from my address?",
        "relevant_docs": ["POL009"]
    },

    {
        "query": "What should I do if my order arrives damaged?",
        "relevant_docs": ["POL012"]
    },

    {
        "query": "What happens if I receive the wrong product?",
        "relevant_docs": ["POL013"]
    }
]


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    model = SentenceTransformer(
        MODEL_NAME
    )

    all_results = []

    for item in EVALUATION_QUERIES:

        query = item["query"]

        relevant_docs = set(
            item["relevant_docs"]
        )

        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = index.search(
            query_embedding,
            3
        )

        retrieved_chunks = []

        for idx in indices[0]:

            if idx < 0:
                continue

            retrieved_chunks.append(
                metadata[idx]
            )

        # -------------------------------------------------
        # Map chunks to documents
        # -------------------------------------------------

        retrieved_docs = []

        for chunk in retrieved_chunks:

            doc_id = chunk["doc_id"]

            if doc_id not in retrieved_docs:

                retrieved_docs.append(
                    doc_id
                )

        retrieved_docs = retrieved_docs[:3]

        retrieved_set = set(
            retrieved_docs
        )

        # -------------------------------------------------
        # Precision@3
        # -------------------------------------------------

        relevant_retrieved = (
            retrieved_set
            & relevant_docs
        )

        precision_at_3 = (
            len(relevant_retrieved)
            / 3
        )

        # -------------------------------------------------
        # Recall@3
        # -------------------------------------------------

        recall_at_3 = (
            len(relevant_retrieved)
            / len(relevant_docs)
        )

        result = {
            "query": query,
            "expected": sorted(
                relevant_docs
            ),
            "retrieved": retrieved_docs,
            "precision_at_3":
                precision_at_3,
            "recall_at_3":
                recall_at_3
        }

        all_results.append(result)

        print("\nQuery:")
        print(query)

        print(
            "Expected documents:",
            sorted(relevant_docs)
        )

        print(
            "Retrieved documents:",
            retrieved_docs
        )

        print(
            f"Precision@3 = "
            f"{len(relevant_retrieved)}/3 "
            f"= {precision_at_3:.4f}"
        )

        print(
            f"Recall@3 = "
            f"{len(relevant_retrieved)}/"
            f"{len(relevant_docs)} "
            f"= {recall_at_3:.4f}"
        )

    # -----------------------------------------------------
    # Averages
    # -----------------------------------------------------

    avg_precision = np.mean([
        result["precision_at_3"]
        for result in all_results
    ])

    avg_recall = np.mean([
        result["recall_at_3"]
        for result in all_results
    ])

    print("\n" + "=" * 70)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 70)

    print(
        f"Average Precision@3: "
        f"{avg_precision:.4f}"
    )

    print(
        f"Average Recall@3: "
        f"{avg_recall:.4f}"
    )

    # -----------------------------------------------------
    # Save report
    # -----------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for result in all_results:

            f.write(
                f"Query: {result['query']}\n"
            )

            f.write(
                f"Expected: {result['expected']}\n"
            )

            f.write(
                f"Retrieved: {result['retrieved']}\n"
            )

            f.write(
                f"Precision@3: "
                f"{result['precision_at_3']:.4f}\n"
            )

            f.write(
                f"Recall@3: "
                f"{result['recall_at_3']:.4f}\n"
            )

            f.write("\n")

        f.write(
            f"Average Precision@3: "
            f"{avg_precision:.4f}\n"
        )

        f.write(
            f"Average Recall@3: "
            f"{avg_recall:.4f}\n"
        )

    print(
        f"\nReport saved to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()