# Evaluation Report: Policy-Aware Multi-Agent RAG Claim Decision Engine

**Date:** 2026-09-17 17:03:22

## Overall Metrics

- **Decision Quality (Label Accuracy):** 70.6%
- **Validation Gate Pass Rate:** 52.9% (9/17 cases passed gate)
- **Retrieval Section Recall@k:** 100.0%
- **Canonical Citation Resolver Accuracy:** 95.1% (provenance, chunk_id existence, text grounding)
- **Material Finding Citation Coverage:** 86.5% (key findings supported by policy citations)
- **Abstention Accuracy (NEEDS_REVIEW):** 100.0%

## Detailed Case Results

| Case ID   | Expected               | System Decision        | Match   |   Confidence | Validation   | Section Recall   | Finding Coverage   | Latency   |
|-----------|------------------------|------------------------|---------|--------------|--------------|------------------|--------------------|-----------|
| PUB-001   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.94 | PASS         | 100%             | 100%               | 72.19s    |
| PUB-002   | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         1    | PASS         | 100%             | 100%               | 70.26s    |
| PUB-003   | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         1    | PASS         | 100%             | 100%               | 103.87s   |
| PUB-004   | ADMISSIBLE_WITH_LIMITS | NEEDS_REVIEW           | FAIL    |         0.65 | FAIL         | 100%             | 86%                | 157.60s   |
| PUB-005   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.96 | PASS         | 100%             | 100%               | 117.73s   |
| PUB-006   | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.61 | FAIL         | 100%             | 100%               | 90.31s    |
| PUB-007   | ADMISSIBLE_WITH_LIMITS | NEEDS_REVIEW           | FAIL    |         0.65 | FAIL         | 100%             | 86%                | 167.91s   |
| PUB-008   | NOT_ADMISSIBLE         | NEEDS_REVIEW           | FAIL    |         0.55 | FAIL         | 100%             | 80%                | 101.76s   |
| PUB-009   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.99 | PASS         | 100%             | 100%               | 173.70s   |
| PUB-010   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.98 | PASS         | 100%             | 80%                | 63.08s    |
| PUB-011   | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.69 | PASS         | 100%             | 67%                | 71.36s    |
| PUB-012   | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         1    | PASS         | 100%             | 67%                | 57.71s    |
| CUST-001  | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.6  | FAIL         | 100%             | 75%                | 66.07s    |
| CUST-002  | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         1    | PASS         | 100%             | 75%                | 178.86s   |
| CUST-003  | ADMISSIBLE_WITH_LIMITS | NEEDS_REVIEW           | FAIL    |         0.65 | FAIL         | 100%             | 62%                | 164.46s   |
| CUST-004  | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.68 | FAIL         | 100%             | 75%                | 59.37s    |
| CUST-005  | PARTIALLY_ADMISSIBLE   | NEEDS_REVIEW           | FAIL    |         0.73 | FAIL         | 100%             | 100%               | 79.45s    |

## Investigation of Retrieval Recall & Section Naming

In early iterations, naive string matching on synthetic benchmark labels (e.g. searching for literal `"Hospital Definition"`) reported 0% recall because the policy wording uses statutory clause headings like `UNIVERSAL SOMPO GENERAL INSURANCE CO LTD > Hospital` (`chunk_3_036`).
Inspection of raw `retrieved_chunk_ids` revealed that the hybrid retriever DID successfully retrieve `chunk_3_036` (Hospital Definition) and `chunk_4_052` (Medically Necessary) in the top 15 results. Mapping evaluation section targets to canonical policy chunk IDs provides a true reflection of retrieval performance.
