from sentence_transformers import CrossEncoder

from hybrid_retriever import HybridRetriever


class Reranker:
    def __init__(self):
        print("Loading reranker model...")
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Reranker is ready.")

    def rerank(self, query, candidates, top_k=5):
        pairs = []

        for candidate in candidates:
            pairs.append((query, candidate["text"]))

        scores = self.model.predict(pairs)

        reranked_results = []

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
            reranked_results.append(candidate)

        reranked_results = sorted(
            reranked_results,
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return reranked_results[:top_k]


if __name__ == "__main__":
    retriever = HybridRetriever()
    reranker = Reranker()

    query = "What is the MFA and VPN policy?"

    candidates = retriever.retrieve(query, dense_k=5, bm25_k=5)

    print(f"\nRetrieved {len(candidates)} candidate chunks")

    top_results = reranker.rerank(query, candidates, top_k=3)

    print("\nTop reranked results:")

    for result in top_results:
        print("\nSource:", result["source"])
        print("Chunk ID:", result["chunk_id"])
        print("Method:", result["retrieval_method"])
        print("Rerank score:", result["rerank_score"])
        print(result["text"][:500])