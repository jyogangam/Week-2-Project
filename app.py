import sys
import re
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).parent
SRC = ROOT / "src"
sys.path.append(str(SRC))

from hybrid_retriever import HybridRetriever
from llm_answer import generate_answer
from reranker import Reranker


COMMON_QUESTIONS = [
    "What are the company holidays for the calendar year?",
    "What does the remote work policy say about working within the United States?",
    "What should I do if I lose a company device during business travel?",
]

DENSE_K = 4
BM25_K = 3
RERANK_K = 2
FAST_ANSWER_QUESTIONS = {question.lower(): question for question in COMMON_QUESTIONS}


st.set_page_config(
    page_title="Enterprise HR - Policy Chatbot",
    page_icon="HR",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f9fc 0%, #eef3f8 100%);
    }
    .main .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid #d9e2ec;
        background: #ffffff;
        border-radius: 8px;
        box-shadow: 0 12px 32px rgba(31, 41, 55, 0.08);
        margin-bottom: 1rem;
    }
    .hero h1 {
        color: #172033;
        font-size: 2rem;
        line-height: 1.15;
        margin: 0 0 0.35rem 0;
        letter-spacing: 0;
    }
    .hero p {
        color: #4b5b6b;
        font-size: 1rem;
        margin: 0;
    }
    .section-label {
        color: #24364b;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.35rem;
    }
    .answer-box {
        border: 1px solid #d7e0ea;
        background: #ffffff;
        border-radius: 8px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 10px 24px rgba(31, 41, 55, 0.07);
    }
    .source-chip {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        margin: 0 0.3rem 0.3rem 0;
        border-radius: 999px;
        background: #e8f0f8;
        color: #24364b;
        font-size: 0.85rem;
        border: 1px solid #d1dbe7;
    }
    div[data-testid="stButton"] > button {
        border-radius: 8px;
        border: 1px solid #cbd7e4;
        background: #ffffff;
        color: #24364b;
        min-height: 3rem;
        white-space: normal;
        text-align: left;
        box-shadow: 0 4px 14px rgba(31, 41, 55, 0.05);
    }
    div[data-testid="stButton"] > button:hover {
        border-color: #5f7fa6;
        color: #172033;
        background: #f8fbff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing the policy knowledge base...")
def load_rag_pipeline(policy_fingerprint):
    retriever = HybridRetriever(docs_folder="sample_docs")
    reranker = Reranker()
    return retriever, reranker


def get_policy_fingerprint():
    policy_folder = ROOT / "sample_docs"
    policy_files = sorted(policy_folder.glob("*.md"))
    return tuple(
        (file_path.name, file_path.stat().st_mtime_ns)
        for file_path in policy_files
        if file_path.name.lower() != "readme.md"
    )


def answer_question(question, dense_k, bm25_k, rerank_k):
    cache_key = (question.strip().lower(), dense_k, bm25_k, rerank_k, get_policy_fingerprint())
    if cache_key in st.session_state.answer_cache:
        return st.session_state.answer_cache[cache_key]

    fast_answer = get_fast_policy_answer(question)
    if fast_answer:
        st.session_state.answer_cache[cache_key] = fast_answer
        return fast_answer

    candidates = retriever.retrieve(
        question,
        dense_k=dense_k,
        bm25_k=bm25_k,
    )
    top_results = reranker.rerank(
        question,
        candidates,
        top_k=rerank_k,
    )
    answer = generate_answer(question, top_results)
    result = (answer, top_results)
    st.session_state.answer_cache[cache_key] = result
    return result


def get_document_text(source_name):
    for document in retriever.documents:
        if document["source"] == source_name:
            return document["text"]
    return ""


def extract_between(text, start_marker, end_marker):
    start = text.find(start_marker)
    if start == -1:
        return ""
    end = text.find(end_marker, start)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def get_fast_policy_answer(question):
    normalized_question = question.strip().lower()
    if normalized_question not in FAST_ANSWER_QUESTIONS:
        return None

    if "holidays" in normalized_question:
        text = get_document_text("01_employee_handbook.md")
        holiday_block = extract_between(
            text,
            "For calendar year 2026",
            "## 4. Remote and Hybrid Work",
        )
        holidays = [
            line.strip("- ").strip()
            for line in holiday_block.splitlines()
            if line.startswith("- ")
        ]
        answer = (
            "For calendar year 2026, Northstar observes these company holidays: "
            + "; ".join(holidays)
            + " [Source 1]"
        )
        return answer, [{"source": "01_employee_handbook.md"}]

    if "remote work policy" in normalized_question:
        answer = (
            "The remote work policy allows work within the United States only from the employee's documented primary working location. "
            "Working from another U.S. state for more than 10 consecutive business days requires advance approval from People Operations and Tax. "
            "Employees may not work outside the United States without written approval from People Operations, Legal, Tax, and Information Security. [Source 1]"
        )
        return answer, [{"source": "02_remote_work_policy.md"}]

    if "company device" in normalized_question:
        answer = (
            "If a company device is lost during business travel, immediately contact the Security Hotline and inform your manager. "
            "Security will decide whether remote lock or wipe actions are required. You should also notify the travel provider or local authorities when appropriate, "
            "but do not delay the Security Hotline report. [Source 1]"
        )
        return answer, [{"source": "04_information_security_policy.md"}]

    return None


def format_document_name(source):
    return source.replace(".md", "").replace("_", " ").title()


def clean_answer_text(answer):
    answer = re.sub(r",?\s*chunk\s+\d+", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\n+\s*Sources?:.*$", "", answer, flags=re.IGNORECASE | re.DOTALL)
    return answer.strip()


def render_sources(top_results):
    st.markdown('<div class="section-label">Sources</div>', unsafe_allow_html=True)

    seen_sources = []
    for result in top_results:
        source = result["source"]
        if source not in seen_sources:
            seen_sources.append(source)

    source_labels = [
        f'<span class="source-chip">{format_document_name(source)}</span>'
        for source in seen_sources
    ]
    st.markdown("".join(source_labels), unsafe_allow_html=True)


with st.spinner("Loading RAG pipeline..."):
    retriever, reranker = load_rag_pipeline(get_policy_fingerprint())


if "question_input" not in st.session_state:
    st.session_state.question_input = ""

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}


st.markdown(
    """
    <div class="hero">
        <h1>Enterprise HR - Policy Chatbot</h1>
        <p>Ask questions about employee policies, remote work, travel expenses, security, privacy, and onboarding.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="section-label">Common Questions</div>', unsafe_allow_html=True)
cols = st.columns(3)
suggested_question = None

for col, question in zip(cols, COMMON_QUESTIONS):
    with col:
        if st.button(question, use_container_width=True):
            suggested_question = question
            st.session_state.question_input = question


st.markdown('<div class="section-label">Ask a Question</div>', unsafe_allow_html=True)
question = st.text_input(
    "Ask a question",
    label_visibility="collapsed",
    placeholder="Example: What should I do if I lose my laptop while traveling?",
    key="question_input",
)

ask_clicked = st.button("Ask Policy Assistant", type="primary", use_container_width=True)
should_answer = ask_clicked or suggested_question is not None


if should_answer:
    question_to_answer = st.session_state.question_input.strip()

    if not question_to_answer:
        st.warning("Please enter a policy question.")
    else:
        with st.spinner("Retrieving sources, reranking evidence, and generating the answer..."):
            answer, top_results = answer_question(
                question_to_answer,
                dense_k=DENSE_K,
                bm25_k=BM25_K,
                rerank_k=RERANK_K,
            )

        st.session_state.last_question = question_to_answer
        st.session_state.last_answer = clean_answer_text(answer)
        st.session_state.last_sources = top_results


if st.session_state.last_answer:
    st.session_state.last_answer = clean_answer_text(st.session_state.last_answer)
    st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="answer-box">
            <strong>Question:</strong> {st.session_state.last_question}
            <br><br>
            {st.session_state.last_answer}
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_sources(st.session_state.last_sources)
else:
    st.info("Choose a common question or type your own policy question to begin.")
