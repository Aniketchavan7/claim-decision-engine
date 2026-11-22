# Integration Specification: HL7 FHIR R4 ExplanationOfBenefit (EOB) Mapping

## 1. Purpose
Defines the bidirectional normalization bridge between international HL7 FHIR R4 standard healthcare claims and internal policy-aware adjudication models.

## 2. Resource Field Mapping
| FHIR EOB Path | ClaimDecision Model Field | Transformation Rule |
| :--- | :--- | :--- |
| `item[].service.coding[].code` | `procedure_code` | Normalized to standard CPT/ICD-10-PCS |
| `diagnosis[].diagnosisCodeableConcept` | `diagnosis_code` | Normalized to ICD-10-CM |
| `total[].amount.value` | `billed_amount` | Decimal monetary representation |
| `insurance[].coverage.reference` | `policy_id` | Canonical policy chunk index lookup |
