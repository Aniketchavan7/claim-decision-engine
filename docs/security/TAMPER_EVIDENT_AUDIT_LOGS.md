# Compliance Architecture: Tamper-Evident Audit Trails

## 1. Hash Chaining
Each audit log entry contains `prev_hash = SHA256(entry[t-1])`, forming an immutable cryptographic audit ledger.
