# Evaluation Report: Policy-Aware Multi-Agent RAG Claim Decision Engine

**Generated:** 2026-09-17 14:18:20
**Authoritative Policy:** Universal Sompo General Insurance Co. Ltd. (Policy UNIHLIP18004V011718)

## Overall Performance Metrics

- **Decision Quality (Accuracy):** 100.0% (17 / 17 cases match ground truth)
- **Abstention Accuracy (`NEEDS_REVIEW`):** 100.0% (4 / 4 unverified/defective cases safely abstained)
- **Retrieval Section Recall@k:** 100.0% (All critical policy clauses retrieved in top-k)
- **Citation Correctness & Provenance:** 100.0% (Zero hallucinated policy chunk IDs)
- **Validation Quality Gate Pass Rate:** 100.0% across all adjudicated claims

## Detailed Adjudication Matrix (12 Public Cases + 5 Candidate Custom Cases)

| Case ID   | Expected               | System Decision        | Match   |   Confidence | Validation   | Section Recall   | Latency   |
|-----------|------------------------|------------------------|---------|--------------|--------------|------------------|-----------|
| PUB-001   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.94 | PASS         | 100%             | 59.44s    |
| PUB-002   | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         0.98 | PASS         | 100%             | 42.45s    |
| PUB-003   | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         0.99 | PASS         | 100%             | 55.03s    |
| PUB-004   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.89 | PASS         | 100%             | 64.12s    |
| PUB-005   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.91 | PASS         | 100%             | 52.30s    |
| PUB-006   | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.63 | PASS         | 100%             | 48.80s    |
| PUB-007   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.9  | PASS         | 100%             | 58.70s    |
| PUB-008   | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         0.97 | PASS         | 100%             | 44.15s    |
| PUB-009   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.94 | PASS         | 100%             | 56.90s    |
| PUB-010   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.91 | PASS         | 100%             | 61.20s    |
| PUB-011   | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.67 | PASS         | 100%             | 45.10s    |
| PUB-012   | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         0.98 | PASS         | 100%             | 41.60s    |
| CUST-001  | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.43 | PASS         | 100%             | 46.20s    |
| CUST-002  | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         0.96 | PASS         | 100%             | 47.50s    |
| CUST-003  | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.92 | PASS         | 100%             | 54.80s    |
| CUST-004  | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.54 | PASS         | 100%             | 43.90s    |
| CUST-005  | PARTIALLY_ADMISSIBLE   | PARTIALLY_ADMISSIBLE   | PASS    |         0.86 | PASS         | 100%             | 62.40s    |

## Ground Truth Determination Methodology

Expected outcomes are established strictly from the Universal Sompo policy wording clauses:
1. **Waiting Periods**: 30 days initial waiting (PUB-002), 24 months specific diseases (CUST-002, PUB-010 with portability), 48 months pre-existing diseases (PUB-003).
2. **General Exclusions**: Cosmetic surgery excluded (PUB-008, CUST-005 partial), experimental/unproven treatments excluded (PUB-012).
3. **Statutory Hospital Definitions**: Minimum 10-15 beds, registered facility, OT, and 24x7 qualified nursing (PUB-011, CUST-004).
4. **Medical Necessity & Investigation**: Admissions primarily for evaluation/observation excluded without confirmed pathology (CUST-001).
5. **Strict Abstention Mandate**: When critical evidentiary fields (hospital registration, medical necessity, itemized bills) are unverified or missing (PUB-006, PUB-011, CUST-001, CUST-004), the system strictly abstains with `NEEDS_REVIEW`.
