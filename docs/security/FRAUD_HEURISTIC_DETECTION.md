# Security & Integrity: Healthcare Claim Fraud Heuristic Rules

## 1. Fraud Detection Engine
The heuristic engine evaluates incoming claim line items against statutory National Correct Coding Initiative (NCCI) edit rules:
1. **Unbundling Detector**: Flags claims where component procedure codes are billed alongside comprehensive bundle codes for the same operative session.
2. **Upcoding Detector**: Flags diagnosis code severity that is statistically anomalous relative to documented length of hospital stay and laboratory markers.
