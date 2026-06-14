# Enterprise HR - Policy Chatbot

Enterprise HR - Policy Chatbot is a Retrieval-Augmented Generation (RAG) application that helps employees ask natural-language questions about internal company policies and receive grounded answers from approved policy documents.

The goal is to reduce the time employees spend searching across handbooks, travel policies, remote-work rules, reimbursement guidance, security procedures, onboarding guides, and privacy documents. Instead of manually opening multiple files, an employee can ask one question and receive a concise answer with the source documents used by the system.

## Problem Statement

Employees often need quick answers to policy questions such as:

- What holidays are observed in the calendar year?
- Can I work remotely from another U.S. state?
- What expenses require receipts?
- What should I do if I lose a company device while traveling?

In many organizations, these answers are spread across multiple documents owned by HR, Finance, Security, Legal, and Operations teams. This creates friction for employees, increases repetitive questions to support teams, and raises the risk of people relying on outdated or incomplete information.

This project solves that problem by building a policy-aware chatbot that retrieves relevant policy sections, reranks them for relevance, and asks an LLM to answer only from the retrieved knowledge. If the answer is not present in the policy corpus, the chatbot is instructed to say: `I do not know`.

## Key Features

- Streamlit web interface for asking policy questions
- Markdown policy document ingestion
- Metadata extraction from policy files
- Text chunking for retrieval-friendly document sections
- Local sentence-transformer embeddings
- FAISS vector search for dense semantic retrieval
- BM25 keyword retrieval for exact policy terms
- Hybrid retrieval combining dense and keyword search
- Cross-encoder reranking for stronger result ordering
- LLM answer generation with a strict system prompt
- Source document display without exposing chunk IDs or scores
- Evaluation script for retrieval quality and edge cases
- Fast-path handling for common policy questions to reduce latency

## Project Architecture

```text
User Question
     |
     v
Streamlit App
     |
     v
Hybrid Retriever
     |-----------------------|
     v                       v
Dense Vector Search       BM25 Keyword Search
FAISS + Embeddings        Token-based matching
     |                       |
     |----------- merge -----|
                 |
                 v
Cross-Encoder Reranker
                 |
                 v
Top Policy Context
                 |
                 v
LLM Answer Generator
System Prompt + Retrieved Sources
                 |
                 v
Answer + Source Documents
```

## Technology Stack

| Layer | Technology | Why It Is Used |
|---|---|---|
| User interface | Streamlit | Provides a fast way to build an interactive chatbot UI in Python. |
| Document format | Markdown | Easy to write, inspect, version, and parse for sample policy documents. |
| Metadata extraction | Python regex + pathlib | Extracts document ID, owner, version, effective date, and audience from policy headers. |
| Chunking | LangChain RecursiveCharacterTextSplitter | Splits policy documents into retrieval-friendly chunks while preserving paragraph and heading structure where possible. |
| Embeddings | `sentence-transformers/distiluse-base-multilingual-cased-v2` | Creates semantic vector representations of policy chunks for meaning-based search. |
| Vector search | FAISS | Lightweight, local, free vector search suitable for a demo or prototype without external database setup. |
| Keyword search | BM25 | Improves retrieval for exact policy terms, dates, names, limits, and compliance language. |
| Hybrid retrieval | Dense + BM25 merge | Balances semantic understanding with exact keyword matching. |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reorders candidate chunks by comparing the question directly against each retrieved passage. |
| LLM | Llama model through Nebius-compatible OpenAI client | Generates polished answers from retrieved context while keeping API usage separate from retrieval logic. |
| Evaluation | Custom Python evaluation script | Measures whether retrieval and answers are correct for known policy test cases and edge cases. |

## Folder Structure

```text
enterprise-policy-rag/
  app.py
  requirements.txt
  README.md
  src/
    Load_Documents.py
    chunk_docs.py
    embed_docs.py
    vector_store.py
    hybrid_retriever.py
    reranker.py
    llm_answer.py
    evaluation.py
  sample_docs/
    01_employee_handbook.md
    02_remote_work_policy.md
    03_travel_and_expense_policy.md
    04_information_security_policy.md
    05_data_retention_and_privacy_policy.md
    06_employee_onboarding_guide.md
    README.md
  docs/
    Enterprise_HR_Policy_Chatbot_Project_Framework_v2.docx
    Enterprise_HR_Policy_Chatbot_Architecture_Diagram.docx
```

## Main Code Flow

1. `Load_Documents.py` reads policy files from `sample_docs/`.
2. Metadata such as document ID, owner, version, effective date, and audience is extracted from each file.
3. `chunk_docs.py` splits each policy into smaller chunks suitable for retrieval.
4. `embed_docs.py` converts each chunk into an embedding vector.
5. `vector_store.py` stores embeddings in a FAISS index and performs dense vector search.
6. `hybrid_retriever.py` combines dense vector search with BM25 keyword search.
7. `reranker.py` reranks retrieved chunks using a cross-encoder model.
8. `llm_answer.py` sends the top retrieved context to the LLM with a strict system prompt.
9. `app.py` displays the chatbot UI, answer, and source documents.
10. `evaluation.py` tests retrieval quality, source accuracy, refusal behavior, and edge cases.

## System Prompt Behavior

The assistant is designed to answer only from retrieved policy context.

Important behavior:

- It should not use outside knowledge.
- It should not guess.
- It should preserve important policy details such as dates, limits, approvals, and exceptions.
- It should cite retrieved sources inline.
- It should not show chunk IDs, reranking scores, or internal retrieval details to the user.
- If the retrieved context does not answer the question, it should respond with: `I do not know`.

This is important for enterprise policy use cases because a confident but unsupported answer can be worse than no answer.

## Evaluation Strategy

The evaluation approach checks whether the chatbot retrieves the right documents and whether the final answer stays grounded in the policy corpus.

Relevant metrics for this use case:

| Metric | Meaning | Why It Matters |
|---|---|---|
| Recall@3 | Checks whether the correct source appears in the top 3 retrieved results. | Ensures the system can find the right policy document. |
| Precision@3 | Checks how many of the top 3 retrieved results are actually relevant. | Reduces noisy or unrelated context sent to the LLM. |
| Top-1 Source Accuracy | Checks whether the best result is the expected document. | Important when the first result strongly influences the final answer. |
| Faithfulness | Checks whether the answer is supported by retrieved documents. | Prevents hallucinated policy answers. |
| Groundedness | Checks whether claims are traceable to the provided sources. | Ensures answers are explainable and auditable. |
| Refusal Accuracy | Checks whether unsupported questions return `I do not know`. | Protects against answers outside the knowledge base. |
| Citation Accuracy | Checks whether displayed sources match the answer. | Helps users trust and verify the response. |
| Latency | Measures how long the app takes to answer. | Keeps the chatbot usable in a real employee workflow. |

Example target thresholds:

- Recall@3: 90% or higher
- Precision@3: 80% or higher
- Faithfulness: 95% or higher
- Groundedness: 100% target for policy-critical answers
- Refusal accuracy: 100% for out-of-scope questions
- Citation accuracy: 90% or higher
- Latency: under 10 seconds for standard questions

## Edge Cases Covered

The evaluation set should include:

- Direct factual questions with one clear answer
- Questions that require one source document
- Questions that require multiple documents
- Questions with similar wording across policies
- Questions involving dates, limits, or approval rules
- Questions outside the policy corpus
- Questions where the retrieved context is related but not enough to answer
- Questions that should trigger `I do not know`

## Setup Instructions

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit App

From the `enterprise-policy-rag` folder:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Run Evaluation

From the `enterprise-policy-rag` folder:

```bash
python src/evaluation.py
```

The evaluation output shows which test cases passed, which sources were retrieved, and whether the expected policy documents appeared in the top results.


