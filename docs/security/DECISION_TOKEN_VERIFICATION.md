# Compliance Specification: Tamper-Evident Decision Token Verification

## 1. Architectural Invariant
Every automated claim decision is cryptographically anchored by signing a canonical SHA-256 digest of the decision status, payout valuation, cited policy chunk citations, and timestamp.

$$\text{DecisionDigest} = \text{SHA-256}(\text{ClaimID} \parallel \text{Status} \parallel \text{PayableAmount} \parallel \text{CitedChunks} \parallel \text{Timestamp})$$

## 2. Verification Protocol
External insurance regulators and internal compliance auditors can independently verify decision integrity by invoking `/api/v1/decisions/{id}/verify` with the public audit key.
