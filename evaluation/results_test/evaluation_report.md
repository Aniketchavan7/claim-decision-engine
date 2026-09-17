# Evaluation Report: Policy-Aware Multi-Agent RAG Claim Decision Engine

**Date:** 2026-09-17 19:16:17

## Overall Metrics

- **Decision Quality (Label Accuracy):** 100.0%
- **Validation Gate Pass Rate:** 100.0% (5/5 cases passed gate)
- **Retrieval Section Recall@k:** 100.0%
- **Canonical Citation Resolver Accuracy:** 100.0% (provenance, chunk_id existence, text grounding)
- **Material Finding Citation Coverage:** 100.0% (key findings supported by policy citations)
- **Abstention Accuracy (NEEDS_REVIEW):** 100.0%

## Detailed Case Results

| Case ID   | Expected       | System Decision   | Match   |   Confidence | Validation   | Section Recall   | Finding Coverage   | Latency   |
|-----------|----------------|-------------------|---------|--------------|--------------|------------------|--------------------|-----------|
| PUB-006   | NEEDS_REVIEW   | NEEDS_REVIEW      | PASS    |         0.57 | PASS         | 100%             | 100%               | 89.89s    |
| PUB-011   | NEEDS_REVIEW   | NEEDS_REVIEW      | PASS    |         0.54 | PASS         | 100%             | 100%               | 94.34s    |
| PUB-012   | NOT_ADMISSIBLE | NOT_ADMISSIBLE    | PASS    |         1    | PASS         | 100%             | 100%               | 62.71s    |
| CUST-001  | NEEDS_REVIEW   | NEEDS_REVIEW      | PASS    |         0.51 | PASS         | 100%             | 100%               | 128.67s   |
| CUST-004  | NEEDS_REVIEW   | NEEDS_REVIEW      | PASS    |         0.59 | PASS         | 100%             | 100%               | 77.23s    |

## Investigation of Retrieval Recall & Section Naming

In early iterations, naive string matching on synthetic benchmark labels (e.g. searching for literal `"Hospital Definition"`) reported 0% recall because the policy wording uses statutory clause headings like `UNIVERSAL SOMPO GENERAL INSURANCE CO LTD > Hospital` (`chunk_3_036`).
Inspection of raw `retrieved_chunk_ids` revealed that the hybrid retriever DID successfully retrieve `chunk_3_036` (Hospital Definition) and `chunk_4_052` (Medically Necessary) in the top 15 results. Mapping evaluation section targets to canonical policy chunk IDs provides a true reflection of retrieval performance.
