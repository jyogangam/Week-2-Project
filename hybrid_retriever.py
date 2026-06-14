from rank_bm25 import BM25Okapi
import re

from Load_Documents import load_markdown_documents
from chunk_docs import chunk_documents
from embed_docs import embed_chunks
from vector_store import build_faiss_index, search_index


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    def __init__(self, docs_folder="sample_docs"):
        print("Loading documents...")
        self.documents = load_markdown_documents(docs_folder)

        print("Chunking documents...")
        self.chunks = chunk_documents(self.documents)

        print("Embedding chunks...")
        self.embedded_chunks = embed_chunks(self.chunks)

        print("Building FAISS index...")
        self.faiss_index = build_faiss_index(self.embedded_chunks)

        print("Building BM25 index...")
        tokenized_chunks = [tokenize(chunk["text"]) for chunk in self.chunks]
        self.bm25_index = BM25Okapi(tokenized_chunks)

        print("Hybrid retriever is ready.")

    def dense_retrieve(self, query, top_k=5):
        results = search_index(
            query=query,
            index=self.faiss_index,
            embedded_chunks=self.embedded_chunks,
            top_k=top_k,
        )

        for result in results:
            result["retrieval_method"] = "dense"

        return results

    def bm25_retrieve(self, query, top_k=5):
        tokenized_query = tokenize(query)

        scores = self.bm25_index.get_scores(tokenized_query)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

        for idx in ranked_indexes[:top_k]:
            result = {
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "chunk_id": self.chunks[idx]["chunk_id"],
                "bm25_score": float(scores[idx]),
                "retrieval_method": "bm25",
            }

            results.append(result)

        return results

    def retrieve(self, query, dense_k=5, bm25_k=5):
        dense_results = self.dense_retrieve(query, top_k=dense_k)
        bm25_results = self.bm25_retrieve(query, top_k=bm25_k)

        combined_results = dense_results + bm25_results

        unique_results = {}

        for result in combined_results:
            key = (result["source"], result["chunk_id"])

            if key not in unique_results:
                unique_results[key] = result
            else:
                unique_results[key]["retrieval_method"] += "+bm25"

        return list(unique_results.values())


if __name__ == "__main__":
    retriever = HybridRetriever()

    query = "What is the MFA and VPN policy?"
    results = retriever.retrieve(query, dense_k=3, bm25_k=3)

    print("\nQuery:", query)
    print("\nHybrid retrieval results:")

    for result in results:
        print("\nSource:", result["source"])
        print("Chunk ID:", result["chunk_id"])
        print("Method:", result["retrieval_method"])

        if "distance" in result:
            print("Dense distance:", result["distance"])

        if "bm25_score" in result:
            print("BM25 score:", result["bm25_score"])

        print(result["text"][:500])
