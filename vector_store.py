import faiss
import numpy as np

from embed_docs import embed_chunks, embed_text
from chunk_docs import chunk_documents
from Load_Documents import load_markdown_documents


def build_faiss_index(embedded_chunks):
    embeddings = [chunk["embedding"] for chunk in embedded_chunks]

    embedding_matrix = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embedding_matrix)

    dimension = embedding_matrix.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embedding_matrix)

    return index


def search_index(query, index, embedded_chunks, top_k=3):
    query_embedding = embed_text(query)

    query_vector = np.array([query_embedding]).astype("float32")

    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        result = {
            "text": embedded_chunks[idx]["text"],
            "source": embedded_chunks[idx]["source"],
            "chunk_id": embedded_chunks[idx]["chunk_id"],
            "title": embedded_chunks[idx].get("title"),
            "metadata": embedded_chunks[idx].get("metadata"),
            "similarity_score": float(score),
        }

        results.append(result)

    return results


if __name__ == "__main__":
    docs = load_markdown_documents("sample_docs")
    chunks = chunk_documents(docs)

    print(f"Created {len(chunks)} chunks")
    print("Embedding all chunks...")

    embedded_chunks = embed_chunks(chunks)

    print(f"Embedded {len(embedded_chunks)} chunks")
    print("Building FAISS cosine-similarity index...")

    index = build_faiss_index(embedded_chunks)

    query = "How many days can employees work remotely?"
    results = search_index(query, index, embedded_chunks, top_k=3)

    print("\nQuery:", query)
    print("\nTop results:")

    for result in results:
        print("\nTitle:", result["title"])
        print("Source:", result["source"])
        print("Chunk ID:", result["chunk_id"])
        print("Similarity score:", result["similarity_score"])
        print("Metadata:", result["metadata"])
        print(result["text"][:500])