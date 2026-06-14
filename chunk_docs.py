from langchain_text_splitters import RecursiveCharacterTextSplitter

from Load_Documents import load_markdown_documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for doc in documents:
        split_texts = splitter.split_text(doc["text"])

        for i, chunk_text in enumerate(split_texts):
            chunk = {
                "text": chunk_text,
                "source": doc["source"],
                "chunk_id": i,
                "metadata":doc["metadata"]
            }

            chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    docs = load_markdown_documents("sample_docs")
    chunks = chunk_documents(docs)

    #print(f"Loaded {len(docs)} documents")
    #print(f"Created {len(chunks)} chunks")

    print("\nFirst chunk preview:")
    print("Source:", chunks[0]["source"])
    print("Chunk ID:", chunks[0]["chunk_id"])
    #print(chunks[0]["text"][:700])
    print(chunks[0]["text"])

