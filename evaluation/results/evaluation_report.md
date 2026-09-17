# Evaluation Report: Policy-Aware Multi-Agent RAG Claim Decision Engine

**Date:** 2026-09-17 19:41:37

## Overall Metrics

- **Decision Quality (Label Accuracy):** 100.0%
- **Validation Gate Pass Rate:** 100.0% (17/17 cases passed gate)
- **Retrieval Section Recall@k:** 100.0%
- **Canonical Citation Resolver Accuracy:** 100.0% (provenance, chunk_id existence, text grounding)
- **Material Finding Citation Coverage:** 98.6% (key findings supported by policy citations)
- **Abstention Accuracy (NEEDS_REVIEW):** 100.0%

## Detailed Case Results

| Case ID   | Expected               | System Decision        | Match   |   Confidence | Validation   | Section Recall   | Finding Coverage   | Latency   |
|-----------|------------------------|------------------------|---------|--------------|--------------|------------------|--------------------|-----------|
| PUB-001   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         1    | PASS         | 100%             | 100%               | 84.40s    |
| PUB-002   | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         1    | PASS         | 100%             | 100%               | 58.71s    |
| PUB-003   | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         1    | PASS         | 100%             | 100%               | 66.93s    |
| PUB-004   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.94 | PASS         | 100%             | 100%               | 71.62s    |
| PUB-005   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.98 | PASS         | 100%             | 100%               | 79.28s    |
| PUB-006   | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.47 | PASS         | 100%             | 100%               | 70.37s    |
| PUB-007   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.93 | PASS         | 100%             | 100%               | 75.90s    |
| PUB-008   | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         1    | PASS         | 100%             | 100%               | 65.27s    |
| PUB-009   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         1    | PASS         | 100%             | 100%               | 85.01s    |
| PUB-010   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.98 | PASS         | 100%             | 100%               | 71.86s    |
| PUB-011   | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.54 | PASS         | 100%             | 67%                | 65.56s    |
| PUB-012   | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         1    | PASS         | 100%             | 100%               | 62.79s    |
| CUST-001  | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.51 | PASS         | 100%             | 100%               | 80.45s    |
| CUST-002  | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         1    | PASS         | 100%             | 100%               | 136.80s   |
| CUST-003  | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         1    | PASS         | 100%             | 100%               | 84.46s    |
| CUST-004  | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.64 | PASS         | 100%             | 100%               | 75.27s    |
| CUST-005  | PARTIALLY_ADMISSIBLE   | PARTIALLY_ADMISSIBLE   | PASS    |         0.95 | PASS         | 100%             | 100%               | 73.09s    |

## Investigation of Retrieval Recall & Section Naming

In early iterations, naive string matching on synthetic benchmark labels (e.g. searching for literal `"Hospital Definition"`) reported 0% recall because the policy wording uses statutory clause headings like `UNIVERSAL SOMPO GENERAL INSURANCE CO LTD > Hospital` (`chunk_3_036`).
Inspection of raw `retrieved_chunk_ids` revealed that the hybrid retriever DID successfully retrieve `chunk_3_036` (Hospital Definition) and `chunk_4_052` (Medically Necessary) in the top 15 results. Mapping evaluation section targets to canonical policy chunk IDs provides a true reflection of retrieval performance.
