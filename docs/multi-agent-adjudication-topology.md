# Multi-Agent Adjudication Topology

## Agent Roles
- **Intake Agent**: Parses insurance claim PDFs, receipts, and clinical notes into typed Pydantic models.
- **Policy Search Agent**: Hybrid RAG query retrieving relevant insurance policy exclusions and deductibles.
- **Fraud Detection Agent**: Analyzes billing code frequency anomalies and provider sanctions.
- **Adjudicator Agent**: Synthesizes findings, computes approved payout, and outputs verifiable citations.
