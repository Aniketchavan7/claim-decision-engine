"""Unit tests verifying insurance policy coverage exclusion rules and waiting period heuristics."""

import pytest


def calculate_ambulance_payable(sum_insured: float, claimed: float) -> float:
    """Ambulance charges: 1.0% of Sum Insured or Rs 1,000, whichever is less (Clause chunk_8_111)."""
    limit = min(sum_insured * 0.01, 1000.0)
    return min(claimed, limit)


def calculate_room_rent_cap(sum_insured: float, days: int, claimed_per_day: float) -> float:
    """Room rent sub-limit: 1.0% of Sum Insured per day (Clause chunk_7_100)."""
    cap_per_day = sum_insured * 0.01
    payable_per_day = min(claimed_per_day, cap_per_day)
    return payable_per_day * days


def is_ped_waiting_period_satisfied(policy_tenure_months: int) -> bool:
    """Pre-existing condition waiting period requires minimum 48 months (Clause chunk_8_108)."""
    return policy_tenure_months >= 48


def test_ambulance_sublimit_cap():
    """Verify ambulance allowance is capped at Rs 1,000 even for large sum insured."""
    # Sum Insured 5,00,000 -> 1% is 5,000 -> capped at 1,000
    assert calculate_ambulance_payable(sum_insured=500000.0, claimed=2500.0) == 1000.0
    # Claimed less than cap
    assert calculate_ambulance_payable(sum_insured=500000.0, claimed=800.0) == 800.0


def test_room_rent_daily_cap():
    """Verify room rent is capped at 1% per day."""
    # Sum Insured 3,00,000 -> 1% is 3,000/day. Claimed 4,500/day for 3 days -> 9,000 payable
    assert calculate_room_rent_cap(sum_insured=300000.0, days=3, claimed_per_day=4500.0) == 9000.0
    # Claimed within cap: 2,000/day for 3 days -> 6,000 payable
    assert calculate_room_rent_cap(sum_insured=300000.0, days=3, claimed_per_day=2000.0) == 6000.0


def test_ped_waiting_period_satisfaction():
    """Verify 48-month PED threshold."""
    assert is_ped_waiting_period_satisfied(47) is False
    assert is_ped_waiting_period_satisfied(48) is True
    assert is_ped_waiting_period_satisfied(60) is True
