# Evaluation Report: Policy-Aware Multi-Agent RAG Claim Decision Engine

**Date:** 2026-09-17 17:58:21

## Overall Metrics

- **Decision Quality (Label Accuracy):** 100.0%
- **Validation Gate Pass Rate:** 100.0% (1/1 cases passed gate)
- **Retrieval Section Recall@k:** 100.0%
- **Canonical Citation Resolver Accuracy:** 100.0% (provenance, chunk_id existence, text grounding)
- **Material Finding Citation Coverage:** 100.0% (key findings supported by policy citations)
- **Abstention Accuracy (NEEDS_REVIEW):** 100.0%

## Detailed Case Results

| Case ID   | Expected               | System Decision        | Match   |   Confidence | Validation   | Section Recall   | Finding Coverage   | Latency   |
|-----------|------------------------|------------------------|---------|--------------|--------------|------------------|--------------------|-----------|
| PUB-004   | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | PASS    |         0.96 | PASS         | 100%             | 100%               | 85.79s    |

## Investigation of Retrieval Recall & Section Naming

In early iterations, naive string matching on synthetic benchmark labels (e.g. searching for literal `"Hospital Definition"`) reported 0% recall because the policy wording uses statutory clause headings like `UNIVERSAL SOMPO GENERAL INSURANCE CO LTD > Hospital` (`chunk_3_036`).
Inspection of raw `retrieved_chunk_ids` revealed that the hybrid retriever DID successfully retrieve `chunk_3_036` (Hospital Definition) and `chunk_4_052` (Medically Necessary) in the top 15 results. Mapping evaluation section targets to canonical policy chunk IDs provides a true reflection of retrieval performance.
