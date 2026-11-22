# Cryptographic Audit Trail Architecture

## Integrity Guarantee
Every autonomous decision output by the Claim Decision Engine is hashed into an immutable Merkle tree:
- Leaf Node: `SHA256(claim_id + policy_version + prompt_hash + decision_json + timestamp)`
- Root Hash: Anchored periodically to a tamper-proof transparency log.
- Audit Verification: Independent auditors can verify that decision parameters were never altered retroactively.
