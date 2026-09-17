# Evaluation Report: Policy-Aware Multi-Agent RAG Claim Decision Engine

**Date:** 2026-09-17 15:59:36

## Overall Metrics

- **Decision Quality (Accuracy):** 100.0%
- **Retrieval Section Recall@k:** 100.0%
- **Canonical Citation Resolver Accuracy:** 97.7% (provenance, chunk_id existence, text grounding)
- **Material Finding Citation Coverage:** 89.7% (key findings supported by policy citations)
- **Abstention Accuracy (NEEDS_REVIEW):** 100.0%

## Detailed Case Results

| Case ID   | Expected               | System Decision        | Match   |   Confidence | Validation   | Section Recall   | Finding Coverage   | Latency   |
|-----------|------------------------|------------------------|---------|--------------|--------------|------------------|--------------------|-----------|
| PUB-001   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.92 | PASS         | 100%             | 100%               | 7.36s     |
| PUB-002   | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         0.98 | PASS         | 100%             | 100%               | 42.14s    |
| PUB-003   | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         1    | PASS         | 100%             | 100%               | 52.74s    |
| PUB-004   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.91 | FAIL         | 100%             | 83%                | 126.89s   |
| PUB-005   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         1    | PASS         | 100%             | 80%                | 103.87s   |
| PUB-006   | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.86 | FAIL         | 100%             | 100%               | 58.12s    |
| PUB-007   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.98 | FAIL         | 100%             | 83%                | 157.00s   |
| PUB-008   | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         1    | PASS         | 100%             | 100%               | 40.42s    |
| PUB-009   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.92 | FAIL         | 100%             | 86%                | 180.37s   |
| PUB-010   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         1    | PASS         | 100%             | 80%                | 51.89s    |
| PUB-011   | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.61 | FAIL         | 100%             | 80%                | 64.75s    |
| PUB-012   | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         1    | PASS         | 100%             | 100%               | 69.36s    |
| CUST-001  | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.62 | FAIL         | 100%             | 80%                | 56.45s    |
| CUST-002  | NOT_ADMISSIBLE         | NOT_ADMISSIBLE         | PASS    |         0.99 | PASS         | 100%             | 100%               | 98.71s    |
| CUST-003  | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.83 | FAIL         | 100%             | 100%               | 187.25s   |
| CUST-004  | NEEDS_REVIEW           | NEEDS_REVIEW           | PASS    |         0.79 | FAIL         | 100%             | 83%                | 57.06s    |
| CUST-005  | PARTIALLY_ADMISSIBLE   | PARTIALLY_ADMISSIBLE   | PASS    |         0.95 | FAIL         | 100%             | 83%                | 146.12s   |

## Investigation of Retrieval Recall & Section Naming

In early iterations, naive string matching on synthetic benchmark labels (e.g. searching for literal `"Hospital Definition"`) reported 0% recall because the policy wording uses statutory clause headings like `UNIVERSAL SOMPO GENERAL INSURANCE CO LTD > Hospital` (`chunk_3_036`).
Inspection of raw `retrieved_chunk_ids` revealed that the hybrid retriever DID successfully retrieve `chunk_3_036` (Hospital Definition) and `chunk_4_052` (Medically Necessary) in the top 15 results. Mapping evaluation section targets to canonical policy chunk IDs provides a true reflection of retrieval performance.
