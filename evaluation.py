import argparse
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from hybrid_retriever import HybridRetriever
from reranker import Reranker


TEST_CASES = [
    {
        "id": "direct_pto_carryover",
        "category": "direct factual",
        "question": "How many PTO days can carry over?",
        "expected_sources": ["01_employee_handbook.md"],
        "expected_keywords": ["five", "unused PTO", "carry"],
        "should_refuse": False,
    },
    {
        "id": "direct_remote_state",
        "category": "direct factual",
        "question": "Can I work remotely from another U.S. state for more than 10 consecutive business days?",
        "expected_sources": ["02_remote_work_policy.md"],
        "expected_keywords": ["10 consecutive business days", "People Operations", "Tax"],
        "should_refuse": False,
    },
    {
        "id": "direct_receipt_limit",
        "category": "numeric exact",
        "question": "When do I need an itemized receipt for an expense?",
        "expected_sources": ["03_travel_and_expense_policy.md"],
        "expected_keywords": ["itemized receipt", "$25"],
        "should_refuse": False,
    },
    {
        "id": "direct_vpn",
        "category": "exact keyword",
        "question": "When do employees need to use VPN?",
        "expected_sources": ["02_remote_work_policy.md"],
        "expected_keywords": ["VPN", "public", "shared network"],
        "should_refuse": False,
    },
    {
        "id": "policy_code_sec003",
        "category": "policy code",
        "question": "What does SEC-003 say to do for lost or stolen equipment?",
        "expected_sources": ["04_information_security_policy.md", "02_remote_work_policy.md"],
        "expected_keywords": ["Security Hotline", "lost", "stolen"],
        "should_refuse": False,
    },
    {
        "id": "policy_precedence_petcare",
        "category": "policy precedence",
        "question": "Can my manager approve reimbursement for pet care?",
        "expected_sources": ["03_travel_and_expense_policy.md"],
        "expected_keywords": ["pet care", "does not reimburse", "Finance approval"],
        "should_refuse": False,
    },
    {
        "id": "multi_lost_laptop_travel",
        "category": "multi-hop",
        "question": "What should I do if I lose my laptop while traveling?",
        "expected_sources": ["04_information_security_policy.md"],
        "expected_keywords": ["Security Hotline", "manager", "travel provider", "local authorities"],
        "should_refuse": False,
    },
    {
        "id": "multi_security_retention",
        "category": "multi-hop",
        "question": "How long are records from a security investigation kept, and what should I do before deleting them?",
        "expected_sources": ["05_data_retention_and_privacy_policy.md"],
        "expected_keywords": ["three years", "legal hold", "written release"],
        "should_refuse": False,
    },
    {
        "id": "ambiguous_new_employee_remote",
        "category": "ambiguous boundary",
        "question": "Can a new employee work remotely?",
        "expected_sources": ["02_remote_work_policy.md", "06_employee_onboarding_guide.md"],
        "expected_keywords": ["90 calendar days", "onboarding", "occasional work from home"],
        "should_refuse": False,
    },
    {
        "id": "ambiguous_time_off",
        "category": "ambiguous broad",
        "question": "What is the policy on time off?",
        "expected_sources": ["01_employee_handbook.md"],
        "expected_keywords": ["PTO", "sick leave"],
        "should_refuse": False,
    },
    {
        "id": "unanswerable_dental",
        "category": "unanswerable",
        "question": "What dental insurance plan does Northstar offer?",
        "expected_sources": [],
        "expected_keywords": [],
        "should_refuse": True,
    },
    {
        "id": "unanswerable_stock",
        "category": "unanswerable",
        "question": "What is the company stock option vesting schedule?",
        "expected_sources": [],
        "expected_keywords": [],
        "should_refuse": True,
    },
    {
        "id": "unanswerable_visa",
        "category": "unanswerable",
        "question": "Does Northstar sponsor H-1B visas?",
        "expected_sources": [],
        "expected_keywords": [],
        "should_refuse": True,
    },
]


def rank_of_first_expected(results, expected_sources):
    if not expected_sources:
        return None

    for rank, result in enumerate(results, start=1):
        if result.get("source") in expected_sources:
            return rank

    return None


def all_expected_in_top_k(results, expected_sources, k):
    if not expected_sources:
        return None

    retrieved_sources = [result.get("source") for result in results[:k]]
    return all(source in retrieved_sources for source in expected_sources)


def top_1_source_match(results, expected_sources):
    if not expected_sources:
        return None

    return bool(results) and results[0].get("source") in expected_sources


def reciprocal_rank(rank):
    if rank is None:
        return 0.0

    return 1.0 / rank


def answer_contains_keywords(answer, expected_keywords):
    if not expected_keywords:
        return None

    answer_lower = answer.lower()
    return all(keyword.lower() in answer_lower for keyword in expected_keywords)


def is_refusal(answer):
    return answer.strip().lower() == "i do not know"


def format_result(result):
    score = result.get("rerank_score")
    score_text = "" if score is None else f", rerank={score:.2f}"
    return (
        f"{result.get('source')}#chunk-{result.get('chunk_id')}"
        f"[{result.get('retrieval_method')}{score_text}]"
    )


def evaluate(args):
    print("Loading retriever...")
    retriever = HybridRetriever(docs_folder=args.docs_folder)

    reranker = None
    if not args.no_rerank:
        print("Loading reranker...")
        reranker = Reranker()

    answer_generator = None
    unknown_answer = "I do not know"
    if args.answers:
        from llm_answer import UNKNOWN_ANSWER, generate_answer

        answer_generator = generate_answer
        unknown_answer = UNKNOWN_ANSWER

    rows = []

    for test_case in TEST_CASES:
        candidates = retriever.retrieve(
            test_case["question"],
            dense_k=args.dense_k,
            bm25_k=args.bm25_k,
        )

        if reranker:
            top_results = reranker.rerank(
                test_case["question"],
                candidates,
                top_k=args.top_k,
            )
        else:
            top_results = candidates[: args.top_k]

        answer = None
        keyword_pass = None
        refusal_pass = None

        if answer_generator:
            answer = answer_generator(test_case["question"], top_results)
            keyword_pass = answer_contains_keywords(answer, test_case["expected_keywords"])
            if test_case["should_refuse"]:
                refusal_pass = answer.strip().lower() == unknown_answer.lower()

        rank = rank_of_first_expected(top_results, test_case["expected_sources"])

        rows.append(
            {
                "case": test_case,
                "top_results": top_results,
                "top1": top_1_source_match(top_results, test_case["expected_sources"]),
                "recall3": all_expected_in_top_k(top_results, test_case["expected_sources"], 3),
                "recall5": all_expected_in_top_k(top_results, test_case["expected_sources"], 5),
                "mrr": reciprocal_rank(rank),
                "answer": answer,
                "keyword_pass": keyword_pass,
                "refusal_pass": refusal_pass,
            }
        )

    print("\nRAG EVALUATION RESULTS")
    print("=" * 100)

    for row in rows:
        test_case = row["case"]
        print(f"\n{test_case['id']} [{test_case['category']}]")
        print("Question:", test_case["question"])
        expected = test_case["expected_sources"] or "(none: should answer I do not know)"
        print("Expected sources:", expected)
        print("Top results:")

        for index, result in enumerate(row["top_results"], start=1):
            print(f"  {index}. {format_result(result)}")

        if test_case["expected_sources"]:
            print("Top-1 source accuracy:", "PASS" if row["top1"] else "FAIL")
            print("Recall@3:", "PASS" if row["recall3"] else "FAIL")
            print("Recall@5:", "PASS" if row["recall5"] else "FAIL")
            print(f"MRR contribution: {row['mrr']:.2f}")
        else:
            print("Unanswerable retrieval note: nearest chunks are still returned.")
            print("Expected final behavior: I do not know")

        if row["answer"] is not None:
            if test_case["expected_keywords"]:
                print("Answer keyword accuracy:", "PASS" if row["keyword_pass"] else "FAIL")
            if test_case["should_refuse"]:
                print("Refusal accuracy:", "PASS" if row["refusal_pass"] else "FAIL")
            print("Answer:", row["answer"].replace("\n", " ")[:700])

    answerable_rows = [row for row in rows if row["case"]["expected_sources"]]
    top1_score = sum(1 for row in answerable_rows if row["top1"]) / len(answerable_rows)
    recall3_score = sum(1 for row in answerable_rows if row["recall3"]) / len(answerable_rows)
    recall5_score = sum(1 for row in answerable_rows if row["recall5"]) / len(answerable_rows)
    mrr_score = sum(row["mrr"] for row in answerable_rows) / len(answerable_rows)

    print("\nSUMMARY")
    print("=" * 100)
    print(f"Answerable cases: {len(answerable_rows)}")
    print(f"Top-1 source accuracy: {top1_score:.2f}")
    print(f"Recall@3: {recall3_score:.2f}")
    print(f"Recall@5: {recall5_score:.2f}")
    print(f"MRR: {mrr_score:.2f}")
    print(f"Unanswerable cases checked: {sum(1 for row in rows if row['case']['should_refuse'])}")

    if args.answers:
        keyword_rows = [row for row in rows if row["keyword_pass"] is not None]
        refusal_rows = [row for row in rows if row["refusal_pass"] is not None]

        if keyword_rows:
            keyword_score = sum(1 for row in keyword_rows if row["keyword_pass"]) / len(keyword_rows)
            print(f"Answer keyword accuracy: {keyword_score:.2f}")

        if refusal_rows:
            refusal_score = sum(1 for row in refusal_rows if row["refusal_pass"]) / len(refusal_rows)
            print(f"Refusal accuracy: {refusal_score:.2f}")

    print("\nOBSERVATIONS TO CHECK")
    print("- If README.md appears in results, exclude it from document loading.")
    print("- For unanswerable questions, retrieval will still return nearest chunks.")
    print("- The final answer step must refuse unsupported answers with: I do not know")
    print("- Multi-source questions should be evaluated with Recall@5 or query decomposition.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the enterprise policy RAG pipeline.")
    parser.add_argument("--docs-folder", default="sample_docs")
    parser.add_argument("--dense-k", type=int, default=6)
    parser.add_argument("--bm25-k", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--answers", action="store_true", help="Also call the LLM answer generator.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
