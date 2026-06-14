from sentence_transformers import SentenceTransformer

from chunk_docs import chunk_documents
from Load_Documents import load_markdown_documents


model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")


def embed_text(text: str):
    embedding = model.encode(text)
    return embedding.tolist()


def embed_chunks(chunks):
    embedded_chunks = []
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=False)

    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunk = {
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            #"title": chunk["title"],
            "metadata": chunk["metadata"],
            "embedding": embedding,
        }

        embedded_chunks.append(embedded_chunk)

    return embedded_chunks


if __name__ == "__main__":
    docs = load_markdown_documents("sample_docs")
    chunks = chunk_documents(docs)

    #print(f"Created {len(chunks)} chunks")
    #print("Creating local embeddings for first 3 chunks...")

    embedded_chunks = embed_chunks(chunks)
    if len(embedded_chunks) == len(chunks):
        print("All chunks have been successfully embedded.")
    else:
        print("Error: Not all chunks were successfully embedded.")


    #print(f"Embedded {len(embedded_chunks)} chunks")
    #print("Embedding vector length:", len(embedded_chunks[0]["embedding"]))
    #("Title:", embedded_chunks[0]["title"])
    #print("Source:", embedded_chunks[0]["source"])
