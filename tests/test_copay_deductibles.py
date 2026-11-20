"""Unit tests for insurance copayment, deductibles, and proportionate deduction formulas."""

import pytest


def calculate_copay_deduction(payable_amount: float, copay_percentage: float) -> tuple[float, float]:
    """Calculate copayment deduction and remaining insurer liability.
    
    Returns (insurer_payable, copay_deducted).
    """
    if copay_percentage <= 0:
        return payable_amount, 0.0
    deduction = round(payable_amount * (copay_percentage / 100.0), 2)
    payable = round(payable_amount - deduction, 2)
    return payable, deduction


def apply_annual_deductible(claimed_amount: float, deductible_remaining: float) -> tuple[float, float]:
    """Apply annual aggregate deductible to an admissible claim.
    
    Returns (admissible_after_deductible, remaining_deductible_balance).
    """
    if claimed_amount <= deductible_remaining:
        return 0.0, deductible_remaining - claimed_amount
    payable = claimed_amount - deductible_remaining
    return payable, 0.0


def calculate_proportionate_deduction(
    admissible_charges: float,
    room_rent_eligible_per_day: float,
    room_rent_actual_per_day: float,
) -> float:
    """Calculate proportionate deduction on associated medical expenses when room rent exceeds eligible cap.
    
    Applicable only to associate medical expenses (nursing, ICU, doctor visits), excluding pharmacy and implants.
    """
    if room_rent_actual_per_day <= room_rent_eligible_per_day:
        return admissible_charges
    proportion = room_rent_eligible_per_day / room_rent_actual_per_day
    return round(admissible_charges * proportion, 2)


def test_copay_calculation_standard_20_percent():
    payable, deducted = calculate_copay_deduction(payable_amount=100000.0, copay_percentage=20.0)
    assert deducted == 20000.0
    assert payable == 80000.0


def test_copay_zero_percent():
    payable, deducted = calculate_copay_deduction(payable_amount=50000.0, copay_percentage=0.0)
    assert deducted == 0.0
    assert payable == 50000.0


def test_annual_deductible_exhaustion():
    # Claim 25,000 against a 50,000 deductible -> 0 payable, 25,000 deductible remaining
    payable1, rem1 = apply_annual_deductible(claimed_amount=25000.0, deductible_remaining=50000.0)
    assert payable1 == 0.0
    assert rem1 == 25000.0

    # Next claim of 60,000 against remaining 25,000 deductible -> 35,000 payable, deductible exhausted
    payable2, rem2 = apply_annual_deductible(claimed_amount=60000.0, deductible_remaining=rem1)
    assert payable2 == 35000.0
    assert rem2 == 0.0


def test_proportionate_room_rent_deduction():
    # Eligible: 5,000/day. Actual room: 10,000/day. Proportion: 50%
    # Associate expenses 40,000 -> payable 20,000
    payable = calculate_proportionate_deduction(
        admissible_charges=40000.0,
        room_rent_eligible_per_day=5000.0,
        room_rent_actual_per_day=10000.0,
    )
    assert payable == 20000.0


def test_proportionate_deduction_not_applied_when_within_cap():
    payable = calculate_proportionate_deduction(
        admissible_charges=30000.0,
        room_rent_eligible_per_day=5000.0,
        room_rent_actual_per_day=4500.0,
    )
    assert payable == 30000.0
