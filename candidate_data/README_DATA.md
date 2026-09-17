# Aptino Candidate Input Package

This package contains the policy source material and synthetic public claim cases for the AI Engineer take-home assignment.

## Contents
- `candidate_data/public_test_cases.json` — 12 synthetic public claim cases.
- `candidate_data/README_DATA.md` — data usage notes.
- `schema/claim_case_schema.md` — input field description.
- `policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf` — policy source supplied for the assignment.

## Important
All claimants, dates, hospitals, amounts, and claim facts are synthetic.

The supplied policy is the authoritative source for policy decisions. Do not invent policy rules from external insurance or medical knowledge. If the policy and supplied evidence do not support a safe conclusion, the system should abstain.

The public cases intentionally cover: waiting periods, pre-existing diseases, domiciliary treatment, day-care/less-than-24-hour treatment, insufficient evidence, category limits, exclusions, pre/post hospitalization windows, portability, hospital-definition evidence gaps, and experimental treatment.

Do not modify the supplied public cases. Add your own cases separately.
