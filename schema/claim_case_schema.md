# Claim Case Input Schema

Required fields include `case_id`, `policy_id`, `policy_start_date`, `claim_date`, `sum_insured_inr`, continuity information, patient, hospital, treatment, expenses, documents, and task.

Common treatment fields: `type`, `admission_hours`, `diagnosis`, `procedure`, `pre_existing`, `experimental`.

Common optional fields include `prior_policy`, `evidence_context`, and `expense_timing`. The candidate system should tolerate unknown/non-critical fields.
