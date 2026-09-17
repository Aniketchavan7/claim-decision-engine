"""End-to-End Evaluation Script for Policy-Aware Multi-Agent RAG Claim Decision Engine.

Measures:
1. Decision Quality: Accuracy against explicit ground truth
2. Retrieval Quality: Expected policy section recall@k per dimension
3. Citation Correctness: Percentage of cited chunks having non-empty text and valid chunk_id
4. Abstention Fidelity: Correct identification of NEEDS_REVIEW cases
5. Latency and Agent Trace Statistics
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from tabulate import tabulate

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.policy_evidence import set_retriever
from src.config import get_settings
from src.ingestion.indexer import load_or_build
from src.models.claim import ClaimCase
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.sparse import SparseRetriever
from src.workflow.graph import create_claim_workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_evaluation(output_dir: str = "evaluation/results") -> None:
    settings = get_settings()
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Initializing indexing and multi-agent workflow for evaluation...")
    pdf_path = str(settings.resolve_path(settings.policy_pdf_path))
    index_dir = str(settings.resolve_path(settings.index_dir))

    chunks, faiss_index, bm25_retriever = load_or_build(pdf_path, index_dir)
    dense = DenseRetriever(
        index=faiss_index,
        chunks=chunks,
        model_name=settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
    )
    sparse = SparseRetriever(retriever=bm25_retriever, chunks=chunks)
    reranker = Reranker(model_name=settings.reranker_model)
    hybrid_retriever = HybridRetriever(
        dense=dense, sparse=sparse, reranker=reranker, settings=settings
    )
    set_retriever(hybrid_retriever)

    workflow = create_claim_workflow()

    # Load test cases and ground truth
    with open("candidate_data/public_test_cases.json", "r", encoding="utf-8") as f:
        public_cases = json.load(f)

    with open("evaluation/custom_test_cases.json", "r", encoding="utf-8") as f:
        custom_cases = json.load(f)

    with open("evaluation/expected_outcomes.json", "r", encoding="utf-8") as f:
        expected_outcomes = json.load(f)

    all_cases = public_cases + custom_cases
    logger.info("Loaded %d test cases (%d public, %d custom).", len(all_cases), len(public_cases), len(custom_cases))

    results_table = []
    case_details = []

    correct_decisions = 0
    total_cases = len(all_cases)
    needs_review_expected = 0
    needs_review_correct = 0
    total_citations = 0
    valid_citations = 0

    section_hits = 0
    section_queries = 0

    for idx, case in enumerate(all_cases, 1):
        case_id = case["case_id"]
        logger.info("[%d/%d] Adjudicating Case: %s ...", idx, total_cases, case_id)

        expected = expected_outcomes.get(case_id, {})
        exp_decision = expected.get("expected_decision", "UNKNOWN")
        exp_sections = expected.get("expected_policy_sections", [])

        if exp_decision == "NEEDS_REVIEW":
            needs_review_expected += 1

        t0 = time.time()
        initial_state = {
            "claim_case": case,
            "trace": [],
            "retry_count": 0,
        }

        try:
            final_state = workflow.invoke(initial_state)
            elapsed = time.time() - t0
            actual_decision = final_state.get("decision", "NEEDS_REVIEW")
            confidence = final_state.get("confidence", 0.0)
            citations = final_state.get("citations", [])
            val_status = final_state.get("validation", {}).get("status", "PASS")
        except Exception as e:
            logger.error("Error evaluating %s: %s", case_id, e)
            actual_decision = "ERROR"
            confidence = 0.0
            citations = []
            val_status = "ERROR"
            elapsed = time.time() - t0

        is_decision_correct = actual_decision == exp_decision
        if is_decision_correct:
            correct_decisions += 1
            if exp_decision == "NEEDS_REVIEW":
                needs_review_correct += 1

        # Check citations
        for cit in citations:
            total_citations += 1
            if cit.get("chunk_id") and cit.get("chunk_text"):
                valid_citations += 1

        # Check section recall
        retrieved_dims = final_state.get("evidence_by_dimension", {})
        all_retrieved_text = ""
        for dim_data in retrieved_dims.values():
            for r in dim_data.get("results", []):
                chunk = r.get("chunk", {})
                all_retrieved_text += f" {chunk.get('section', '')} {chunk.get('text', '')}"

        found_sections = []
        for sec in exp_sections:
            section_queries += 1
            if sec.lower() in all_retrieved_text.lower():
                section_hits += 1
                found_sections.append(sec)

        recall_rate = len(found_sections) / len(exp_sections) if exp_sections else 1.0

        results_table.append([
            case_id,
            exp_decision,
            actual_decision,
            "✅ PASS" if is_decision_correct else "❌ FAIL",
            f"{confidence:.2f}",
            val_status,
            f"{recall_rate*100:.0f}%",
            f"{elapsed:.2f}s",
        ])

        case_details.append({
            "case_id": case_id,
            "expected_decision": exp_decision,
            "actual_decision": actual_decision,
            "is_correct": is_decision_correct,
            "confidence": confidence,
            "validation_status": val_status,
            "key_findings": final_state.get("key_findings", []),
            "applicable_limits": final_state.get("applicable_limits", []),
            "citations_count": len(citations),
            "expected_sections": exp_sections,
            "matched_sections": found_sections,
            "elapsed_seconds": elapsed,
        })

    # Metrics computation
    decision_accuracy = (correct_decisions / total_cases) * 100
    abstention_accuracy = (
        (needs_review_correct / needs_review_expected) * 100
        if needs_review_expected > 0
        else 100.0
    )
    citation_correctness = (
        (valid_citations / total_citations) * 100 if total_citations > 0 else 100.0
    )
    retrieval_recall = (
        (section_hits / section_queries) * 100 if section_queries > 0 else 100.0
    )

    headers = [
        "Case ID",
        "Expected",
        "System Decision",
        "Match",
        "Confidence",
        "Validation",
        "Section Recall",
        "Latency",
    ]
    report_table_str = tabulate(results_table, headers=headers, tablefmt="github")

    summary_metrics = {
        "total_cases_evaluated": total_cases,
        "decision_accuracy_pct": round(decision_accuracy, 2),
        "correct_decisions": correct_decisions,
        "abstention_accuracy_pct": round(abstention_accuracy, 2),
        "needs_review_expected": needs_review_expected,
        "needs_review_correct": needs_review_correct,
        "retrieval_section_recall_pct": round(retrieval_recall, 2),
        "citation_validity_pct": round(citation_correctness, 2),
    }

    print("\n" + "=" * 80)
    print("EVALUATION RUN COMPLETE")
    print("=" * 80)
    print(report_table_str)
    print("\nSummary Metrics:")
    for k, v in summary_metrics.items():
        print(f"  {k}: {v}")

    # Write results to output files
    summary_file = Path(output_dir) / "evaluation_summary.json"
    details_file = Path(output_dir) / "evaluation_details.json"
    markdown_file = Path(output_dir) / "evaluation_report.md"

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    with open(details_file, "w", encoding="utf-8") as f:
        json.dump(case_details, f, indent=2)

    with open(markdown_file, "w", encoding="utf-8") as f:
        f.write("# Evaluation Report: Policy-Aware Multi-Agent RAG Claim Decision Engine\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Overall Metrics\n\n")
        f.write(f"- **Decision Quality (Accuracy):** {decision_accuracy:.1f}%\n")
        f.write(f"- **Retrieval Section Recall@k:** {retrieval_recall:.1f}%\n")
        f.write(f"- **Citation Correctness Rate:** {citation_correctness:.1f}%\n")
        f.write(f"- **Abstention Accuracy (NEEDS_REVIEW):** {abstention_accuracy:.1f}%\n\n")
        f.write("## Detailed Case Results\n\n")
        f.write(report_table_str)
        f.write("\n")

    logger.info("Evaluation artifacts written to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evaluation/results", help="Results directory")
    args = parser.parse_args()
    run_evaluation(output_dir=args.output)
