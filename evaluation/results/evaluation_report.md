# Evaluation Report: Policy-Aware Multi-Agent RAG Claim Decision Engine

**Date:** 2026-09-17 18:36:53

## Overall Metrics

- **Decision Quality (Label Accuracy):** 94.1%
- **Validation Gate Pass Rate:** 70.6% (12/17 cases passed gate)
- **Retrieval Section Recall@k:** 100.0%
- **Canonical Citation Resolver Accuracy:** 100.0% (provenance, chunk_id existence, text grounding)
- **Material Finding Citation Coverage:** 95.5% (key findings supported by policy citations)
- **Abstention Accuracy (NEEDS_REVIEW):** 100.0%

## Detailed Case Results

| Case ID   | Expected               | System Decision        | Match   |   Confidence | Validation   | Section Recall   | Finding Coverage   | Latency   |
|-----------|------------------------|------------------------|---------|--------------|--------------|------------------|--------------------|-----------|
| PUB-001   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.98 | PASS         | 100%             | 100%               | 89.71s    |
| PUB-002   | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         1    | PASS         | 100%             | 100%               | 60.70s    |
| PUB-003   | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         1    | PASS         | 100%             | 100%               | 80.08s    |
| PUB-004   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.94 | PASS         | 100%             | 100%               | 77.09s    |
| PUB-005   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.94 | PASS         | 100%             | 100%               | 117.35s   |
| PUB-006   | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.81 | FAIL         | 100%             | 83%                | 105.07s   |
| PUB-007   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         1    | PASS         | 100%             | 100%               | 80.34s    |
| PUB-008   | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         1    | PASS         | 100%             | 75%                | 58.01s    |
| PUB-009   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         1    | PASS         | 100%             | 100%               | 96.82s    |
| PUB-010   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.94 | PASS         | 100%             | 100%               | 129.71s   |
| PUB-011   | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.67 | FAIL         | 100%             | 100%               | 79.91s    |
| PUB-012   | NOT_ADMISSIBLE         | NEEDS_REVIEW           | FAIL    |         0.55 | FAIL         | 100%             | 75%                | 108.73s   |
| CUST-001  | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.59 | FAIL         | 100%             | 100%               | 69.91s    |
| CUST-002  | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         1    | PASS         | 100%             | 100%               | 80.70s    |
| CUST-003  | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.98 | PASS         | 100%             | 100%               | 75.71s    |
| CUST-004  | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.64 | FAIL         | 100%             | 100%               | 84.53s    |
| CUST-005  | PARTIALLY_ADMISSIBLE   | PARTIALLY_ADMISSIBLE   | PASS    |         0.94 | PASS         | 100%             | 80%                | 71.06s    |

## Investigation of Retrieval Recall & Section Naming

In early iterations, naive string matching on synthetic benchmark labels (e.g. searching for literal `"Hospital Definition"`) reported 0% recall because the policy wording uses statutory clause headings like `UNIVERSAL SOMPO GENERAL INSURANCE CO LTD > Hospital` (`chunk_3_036`).
Inspection of raw `retrieved_chunk_ids` revealed that the hybrid retriever DID successfully retrieve `chunk_3_036` (Hospital Definition) and `chunk_4_052` (Medically Necessary) in the top 15 results. Mapping evaluation section targets to canonical policy chunk IDs provides a true reflection of retrieval performance.
