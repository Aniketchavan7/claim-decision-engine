"""Pydantic models for claim case input validation."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, ConfigDict


class Patient(BaseModel):
    age: int


class Hospital(BaseModel):
    name: str
    network_provider: bool = False


class Treatment(BaseModel):
    type: str  # inpatient, day_care, domiciliary
    admission_hours: int = 0
    diagnosis: str = ""
    procedure: str = ""
    pre_existing: bool = False
    experimental: bool = False
    # Domiciliary-specific
    hospital_room_unavailable: Optional[bool] = None
    patient_cannot_be_moved: Optional[bool] = None


class Expenses(BaseModel):
    room: float = 0
    doctor_fees: float = 0
    medicines_diagnostics: float = 0
    pre_hospitalization: float = 0
    post_hospitalization: float = 0
    ambulance: float = 0

    @property
    def total(self) -> float:
        return (
            self.room
            + self.doctor_fees
            + self.medicines_diagnostics
            + self.pre_hospitalization
            + self.post_hospitalization
            + self.ambulance
        )


class PriorPolicy(BaseModel):
    insurer_type: str = ""
    continuous_years: int = 0
    database_and_claim_history_received: bool = False
    previous_sum_insured_inr: float = 0

    model_config = ConfigDict(extra="allow")


class ExpenseTiming(BaseModel):
    pre_hospitalization_days_before_admission: int = 0
    post_hospitalization_days_after_discharge: int = 0
    same_condition_confirmed: bool = False


class EvidenceContext(BaseModel):
    hospital_registered: Optional[bool] = None
    medical_necessity_confirmed: Optional[bool] = None
    hospital_minimum_criteria_documented: Optional[bool] = None

    model_config = ConfigDict(extra="allow")


class ClaimCase(BaseModel):
    """Top-level claim case model. Tolerates unknown fields."""

    case_id: str
    policy_id: str = ""
    policy_start_date: str = ""
    claim_date: str = ""
    sum_insured_inr: float = 0
    continuous_coverage_months: int = 0
    prior_insurer_continuous_years: int = 0

    patient: Patient
    hospital: Hospital
    treatment: Treatment
    expenses_inr: Expenses
    documents: list[str] = []
    task: str = ""

    # Optional fields
    prior_policy: Optional[PriorPolicy] = None
    expense_timing: Optional[ExpenseTiming] = None
    evidence_context: Optional[EvidenceContext] = None

    model_config = ConfigDict(extra="allow")
