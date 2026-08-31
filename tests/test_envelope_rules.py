"""Table-driven unit tests for envelope engine rules (R1 to R7).

R1_AFA_CEILING: amount > ceiling(category) and mandate not AFA-provisioned
R2_ATTEMPT_CAP: attempts_used >= policy.max_attempts
R3_MIN_GAP: now < last_attempt_at + policy.min_gap
R4_DEAD_MANDATE: failure_class == MANDATE_INVALID
R5_ISSUER_DOWN: issuer health DEGRADED or DOWN
R6_UNKNOWN_CAUSE: failure_class == UNKNOWN
R7_FAIL_CLOSED: permissible set empty
"""
import pytest


def test_envelope_rules_placeholder():
    """Placeholder for envelope rules table-driven tests."""
    pass
