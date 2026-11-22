# Multi-Agent Architecture: LangGraph Cyclic State Machine

## 1. Convergence Guarantee
Arbitration cycles are capped at $N=3$ iterations; failure to reach consensus defaults deterministically to `NEEDS_REVIEW`.
