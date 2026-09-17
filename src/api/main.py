"""FastAPI REST API exposing claim adjudication and system health."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.agents.policy_evidence import set_retriever
from src.config import get_settings
from src.ingestion.indexer import load_or_build
from src.models.claim import ClaimCase
from src.models.decision import (
    ApplicableLimit,
    Citation,
    ClaimDecision,
    ConfidenceBreakdown,
    DecisionStatus,
    TraceEntry,
    ValidationResult,
)
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.sparse import SparseRetriever
from src.workflow.graph import create_claim_workflow

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global instances
_workflow_app = None
_hybrid_retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to initialize indexes, embeddings, and workflow once on startup."""
    global _workflow_app, _hybrid_retriever
    settings = get_settings()
    logger.info("Initializing Policy RAG and Multi-Agent Engine...")

    pdf_path = str(settings.resolve_path(settings.policy_pdf_path))
    index_dir = str(settings.resolve_path(settings.index_dir))

    # 1. Load or build FAISS + BM25 indexes
    chunks, faiss_index, bm25_retriever = load_or_build(pdf_path, index_dir)

    # 2. Instantiate retrievers
    dense = DenseRetriever(
        index=faiss_index,
        chunks=chunks,
        model_name=settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
    )
    sparse = SparseRetriever(retriever=bm25_retriever, chunks=chunks)
    reranker = Reranker(model_name=settings.reranker_model)

    _hybrid_retriever = HybridRetriever(
        dense=dense,
        sparse=sparse,
        reranker=reranker,
        settings=settings,
    )
    set_retriever(_hybrid_retriever)

    # 3. Build LangGraph Workflow
    _workflow_app = create_claim_workflow()
    logger.info("Multi-Agent Claim Engine ready.")

    yield

    logger.info("Shutting down Claim Decision Engine.")


app = FastAPI(
    title="Policy-Aware Multi-Agent RAG Claim Decision Engine",
    description="Evidence-backed automated health insurance claim adjudication API.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, Any]:
    """Readiness and liveness check."""
    settings = get_settings()
    return {
        "status": "healthy",
        "llm_provider": settings.llm_provider,
        "reasoning_model": settings.llm_reasoning_model,
        "fast_model": settings.llm_fast_model,
        "embedding_model": settings.embedding_model,
        "index_ready": _hybrid_retriever is not None,
    }


@app.post(
    "/analyze",
    response_model=ClaimDecision,
    status_code=status.HTTP_200_OK,
)
def analyze_claim(claim: ClaimCase) -> ClaimDecision:
    """Analyze a single insurance claim case and return an authoritative, cited decision."""
    if _workflow_app is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workflow engine is not initialized.",
        )

    t0 = time.time()
    logger.info("Received claim analysis request for case_id=%s", claim.case_id)

    initial_state = {
        "claim_case": claim.model_dump(),
        "trace": [],
        "retry_count": 0,
    }

    try:
        # Run multi-agent state graph
        final_state = _workflow_app.invoke(initial_state)
    except Exception as exc:
        logger.exception("Error processing claim %s: %s", claim.case_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline failed: {str(exc)}",
        ) from exc

    # Assemble structured response contract
    total_time_ms = int((time.time() - t0) * 1000)
    logger.info("Completed analysis for case_id=%s in %dms", claim.case_id, total_time_ms)

    raw_decision = final_state.get("decision", DecisionStatus.NEEDS_REVIEW.value)
    try:
        decision_enum = DecisionStatus(raw_decision)
    except ValueError:
        decision_enum = DecisionStatus.NEEDS_REVIEW

    cb_raw = final_state.get("confidence_breakdown", {})
    confidence_breakdown = ConfidenceBreakdown(
        evidence_support=float(cb_raw.get("evidence_support", 0.0)),
        citation_coverage=float(cb_raw.get("citation_coverage", 0.0)),
        retrieval_quality=float(cb_raw.get("retrieval_quality", 0.0)),
        validation_result=float(cb_raw.get("validation_result", 1.0)),
        missing_critical=float(cb_raw.get("missing_critical", 1.0)),
    )

    applicable_limits = [
        ApplicableLimit(**lim) if isinstance(lim, dict) else lim
        for lim in final_state.get("applicable_limits", [])
    ]

    citations = [
        Citation(**cit) if isinstance(cit, dict) else cit
        for cit in final_state.get("citations", [])
    ]

    val_raw = final_state.get("validation", {})
    validation = ValidationResult(
        status=val_raw.get("status", "PASS"),
        unsupported_claims=val_raw.get("unsupported_claims", []),
        feedback=val_raw.get("feedback", []),
    )

    trace = [
        TraceEntry(**tr) if isinstance(tr, dict) else tr
        for tr in final_state.get("trace", [])
    ]

    return ClaimDecision(
        case_id=claim.case_id,
        decision=decision_enum,
        confidence=final_state.get("confidence", confidence_breakdown.composite_score),
        confidence_breakdown=confidence_breakdown,
        key_findings=final_state.get("key_findings", []),
        applicable_limits=applicable_limits,
        missing_evidence=final_state.get("missing_evidence", []),
        citations=citations,
        validation=validation,
        trace=trace,
    )
