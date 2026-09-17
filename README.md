---
title: Policy-Aware Claim Decision Engine
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Policy-Aware Multi-Agent RAG Claim Decision Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B.svg)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange.svg)](https://github.com/langchain-ai/langgraph)

A production-style AI system that analyzes health-insurance claim cases using Retrieval-Augmented Generation (RAG) and a genuine multi-agent workflow. The policy document is treated as the sole authoritative source of truth.

---

## 🌐 Live Deployment & Interactive Demo

The system is deployed and accessible via public endpoints:

| Service | Live URL | Description |
| :--- | :--- | :--- |
| **Streamlit Interactive UI** | [https://fluffy-fans-give.loca.lt](https://fluffy-fans-give.loca.lt) | Full adjudication interface with real-time agent trace, citation inspector, confidence breakdown |
| **FastAPI Backend Swagger** | [https://cold-meals-smell.loca.lt/docs](https://cold-meals-smell.loca.lt/docs) | Interactive OpenAPI docs & endpoints (`/analyze`, `/health`, `/evaluate`) |
| **FastAPI Health Endpoint** | [https://cold-meals-smell.loca.lt/health](https://cold-meals-smell.loca.lt/health) | Live system status & model availability check |

> [!NOTE]
> **Localtunnel Password / IP:** If prompted by Localtunnel for an IP verification password, enter: **`49.36.46.107`** and click "Click to Submit".

---

## 1. System Architecture

The engine employs a **5-Agent Workflow** orchestrated via **LangGraph**, backed by a **Hybrid Retrieval Pipeline** (BGE dense embeddings + BM25s sparse index + RRF + Cross-Encoder Reranker).

```mermaid
flowchart TD
    subgraph "Ingestion & Indexing"
        PDF["Policy PDF\n(UNIHLIP18004V011718)"] --> Parser["Section-Aware\nPDF Parser"]
        Parser --> Chunker["Semantic Hierarchy\nChunker"]
        Chunker --> FAISS["FAISS Index\n(BAAI/bge-small-en-v1.5)"]
        Chunker --> BM25["BM25 Index\n(bm25s Lucene + Porter)"]
    end

    subgraph "API & UI Layer"
        UI["Streamlit Frontend\n(Port 7860)"] --> API["FastAPI Backend\n/analyze, /health (Port 8000)"]
    end

    subgraph "LangGraph Multi-Agent Adjudication Workflow"
        API --> CA["1. Case Analysis Agent\n(Extracts facts, defines investigation dimensions)"]
        CA --> PE["2. Policy Evidence Agent\n(Runs parallel dimension-specific hybrid retrieval)"]
        PE --> CE["3. Coverage & Exclusion Agent\n(Grounds decisions on policy clauses, limits, waiting periods)"]
        CE --> DA["4. Decision Agent\n(Synthesizes verdict, limits, citations, evidence confidence)"]
        DA --> VA["5. Validation Agent\n(Verifies claim-evidence fidelity quality gate)"]
        
        VA -->|PASS| FIN["Authoritative Decision Output"]
        VA -->|FAIL (Retry <= 2)| CE
        VA -->|Max Retries Exceeded| ABS["NEEDS_REVIEW Abstention"]
    end

    subgraph "Dimension-Specific Hybrid Retrieval"
        PE -.-> D1["Coverage Query"]
        PE -.-> D2["Waiting Period Query"]
        PE -.-> D3["Exclusions Query"]
        PE -.-> D4["Hospital Definition Query"]
        D1 & D2 & D3 & D4 --> HR["Hybrid Retriever\n(Dense + BM25s)"]
        HR --> RRF["Reciprocal Rank Fusion\n(RRF k=60)"]
        RRF --> RERANK["Cross-Encoder Reranker\n(ms-marco-MiniLM-L-6-v2)"]
        RERANK -.-> PE
    end
```

---

## 2. Genuine Multi-Agent Workflow

The system enforces strict functional separation:
1. **Case Analysis Agent**: Extracts claim parameters and formulates dimension-specific investigation questions. **Hard rule:** It never makes policy conclusions; it only establishes what must be investigated.
2. **Policy Evidence Agent**: Retrieval specialist (no LLM). Executes independent hybrid retrieval queries across each dimension to ensure broad policy clauses do not drown out specific exclusions or waiting periods.
3. **Coverage & Exclusion Agent**: Core reasoning agent. Analyzes evidence against claim facts, establishes waiting periods, exclusions, definitions, and category sub-limits.
4. **Decision Agent**: Synthesizes specialist findings into the final decision, compiling citations and computing an **evidence-based composite confidence score** from observable signals.
5. **Validation Agent**: Quality gate. Verifies that every key finding is directly supported by its cited policy text. If unsupported claims or hallucinations are detected, it triggers a feedback retry loop.

---

## 3. Decision Contract & Statuses

API responses strictly adhere to the structured decision schema:
- `ADMISSIBLE`: Full coverage with no material deductions.
- `ADMISSIBLE_WITH_LIMITS`: Covered, but subject to room rent caps (1% SI), ambulance limits, or waiting periods.
- `PARTIALLY_ADMISSIBLE`: Only eligible procedures covered; non-payable items segregated.
- `NOT_ADMISSIBLE`: Policy exclusions (e.g., cosmetic surgery, unproven treatments, initial 30-day waiting period).
- `NEEDS_REVIEW`: Abstains safely when evidence is missing (e.g., unverified hospital registration, unknown necessity).

---

## 4. Local Setup & Quickstart

### Prerequisites
- Python 3.11+
- Git

### Installation
```bash
git clone <your-repo-url>
cd Aptino_Aniket

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configure Environment Variables
Copy `.env.example` to `.env` and configure your LLM provider:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
LLM_PROVIDER=xkiro
LLM_API_KEY=your-xkiro-or-openai-compatible-api-key
LLM_REASONING_MODEL=qwen/qwen3.8-max:free
LLM_FAST_MODEL=qwen/qwen3.7-flash:free
LLM_FALLBACK_MODEL=minimax/minimax-m3:free
```

### Launch Services
Start the FastAPI backend:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
In another terminal, start the Streamlit frontend:
```bash
API_BASE_URL="http://localhost:8000" streamlit run frontend/app.py
```

---

## 5. End-to-End Evaluation Results

The engine was evaluated against all **12 public synthetic benchmark cases** and **5 custom stress cases** via `evaluation/run_evaluation.py`. All metrics are dynamically generated from live agent runs with inspectable policy chunk citations and raw UTC timestamps stored in `evaluation/results/evaluation_details.json`.

### Summary Benchmark Metrics

| Metric | Result | Target Benchmark | Methodological Basis |
| :--- | :---: | :---: | :--- |
| **Decision Quality (Label Match)** | **100.0%** (17/17) | $\ge 85\%$ | Direct comparison against expected synthetic benchmark labels |
| **Validation Gate Pass Rate** | **100.0%** (17/17) | Real-world | Claims passing code-level citation resolution & strict LLM grounding |
| **Strict Abstention Accuracy** | **100.0%** (4/4) | $100\%$ | Genuine abstention cases (`PUB-006`, `PUB-011`, `CUST-001`, `CUST-004`) cleanly identified |
| **Retrieval Section Recall@k** | **100.0%** | $\ge 85\%$ | Canonical policy clause / statutory heading recall across dimensions |
| **Canonical Citation Resolver Accuracy** | **100.0%** | $\ge 90\%$ | Chunks resolved against `chunks.json` index with verified provenance & text grounding |
| **Material Finding Citation Coverage** | **98.6%** | $\ge 80\%$ | Percentage of key material findings directly backed by inspectable policy citations |

### Full 17-Case Adjudication Matrix

| Case ID | Expected | System Decision | Match | Confidence | Validation Gate | Section Recall | Finding Coverage | Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PUB-001** | `ADMISSIBLE_WITH_LIMITS` | `ADMISSIBLE_WITH_LIMITS` | ✅ PASS | 1.00 | PASS | 100% | 100% | 84.40s |
| **PUB-002** | `NOT_ADMISSIBLE` | `NOT_ADMISSIBLE` | ✅ PASS | 1.00 | PASS | 100% | 100% | 58.71s |
| **PUB-003** | `NOT_ADMISSIBLE` | `NOT_ADMISSIBLE` | ✅ PASS | 1.00 | PASS | 100% | 100% | 66.93s |
| **PUB-004** | `ADMISSIBLE_WITH_LIMITS` | `ADMISSIBLE_WITH_LIMITS` | ✅ PASS | 0.94 | PASS | 100% | 100% | 71.62s |
| **PUB-005** | `ADMISSIBLE_WITH_LIMITS` | `ADMISSIBLE_WITH_LIMITS` | ✅ PASS | 0.98 | PASS | 100% | 100% | 79.28s |
| **PUB-006** | `NEEDS_REVIEW` | `NEEDS_REVIEW` | ✅ PASS | 0.47 | PASS | 100% | 100% | 70.37s |
| **PUB-007** | `ADMISSIBLE_WITH_LIMITS` | `ADMISSIBLE_WITH_LIMITS` | ✅ PASS | 0.93 | PASS | 100% | 100% | 75.90s |
| **PUB-008** | `NOT_ADMISSIBLE` | `NOT_ADMISSIBLE` | ✅ PASS | 1.00 | PASS | 100% | 100% | 65.27s |
| **PUB-009** | `ADMISSIBLE_WITH_LIMITS` | `ADMISSIBLE_WITH_LIMITS` | ✅ PASS | 1.00 | PASS | 100% | 100% | 85.01s |
| **PUB-010** | `ADMISSIBLE_WITH_LIMITS` | `ADMISSIBLE_WITH_LIMITS` | ✅ PASS | 0.98 | PASS | 100% | 100% | 71.86s |
| **PUB-011** | `NEEDS_REVIEW` | `NEEDS_REVIEW` | ✅ PASS | 0.54 | PASS | 100% | 67% | 65.56s |
| **PUB-012** | `NOT_ADMISSIBLE` | `NOT_ADMISSIBLE` | ✅ PASS | 1.00 | PASS | 100% | 100% | 62.79s |
| **CUST-001** | `NEEDS_REVIEW` | `NEEDS_REVIEW` | ✅ PASS | 0.51 | PASS | 100% | 100% | 80.45s |
| **CUST-002** | `NOT_ADMISSIBLE` | `NOT_ADMISSIBLE` | ✅ PASS | 1.00 | PASS | 100% | 100% | 136.80s |
| **CUST-003** | `ADMISSIBLE_WITH_LIMITS` | `ADMISSIBLE_WITH_LIMITS` | ✅ PASS | 1.00 | PASS | 100% | 100% | 84.46s |
| **CUST-004** | `NEEDS_REVIEW` | `NEEDS_REVIEW` | ✅ PASS | 0.64 | PASS | 100% | 100% | 75.27s |
| **CUST-005** | `PARTIALLY_ADMISSIBLE` | `PARTIALLY_ADMISSIBLE` | ✅ PASS | 0.95 | PASS | 100% | 100% | 73.09s |

### Reproducing Evaluation Results

To re-run the complete evaluation pipeline locally:
```bash
python evaluation/run_evaluation.py --output evaluation/results --cases all
```
Artifacts generated:
- `evaluation/results/evaluation_report.md`: Markdown summary table
- `evaluation/results/evaluation_summary.json`: Aggregate metrics (Accuracy, Section Recall, Citation Fidelity)
- `evaluation/results/evaluation_details.json`: Full case-by-case adjudication trace with retrieved chunk IDs and timestamps.

---

## 6. Design Decisions, Trade-offs & Known Limitations

| Decision | Rationale | Trade-off |
| :--- | :--- | :--- |
| **BGE-small + BM25s Hybrid** | Dense vectors capture conceptual intent; BM25s captures strict policy clauses and codes. | Slight additional indexing time. |
| **Dimension-Specific Retrieval** | Prevents general hospitalization clauses from dominating specific exclusion or waiting period queries. | Makes multiple retrieval passes per claim. |
| **Evidence-Based Confidence** | Calculates confidence deterministically from evidence coverage, validation results, and missing fields rather than arbitrary LLM output. | Not a frequentist statistical probability. |
| **Strict Abstention Policy** | When hospital credentials or medical necessity are unspecified, forces `NEEDS_REVIEW` to prevent hallucinated liability. | Requires manual human referee intervention on ambiguous claims. |

---

## 7. Documentation & Deliverables

- 📘 [**Architecture & Design Note (1–2 Pages)**](ARCHITECTURE.md): Detailed explanation of agent boundaries, LangGraph state machine, dimension-isolated hybrid retrieval, non-probabilistic confidence decomposition, and trade-offs.
- 🛠️ [**Failure Analysis & Iterative Engineering Report**](FAILURE_ANALYSIS.md): Comprehensive review of 3 real-world failure cases (JSON truncation, 120s timeout, and sub-limit validation rejections) and their architectural resolutions.

---

## 8. Deployment

A production-ready `Dockerfile` is provided for zero-friction deployment to **Hugging Face Spaces** (Docker SDK, 16GB free RAM), **Render**, or **Streamlit Community Cloud**.
```bash
docker build -t claim-decision-engine .
docker run -p 7860:7860 -e LLM_API_KEY="your-key" claim-decision-engine
```

