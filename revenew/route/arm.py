"""ArmAssigner: `crc32(customer_id + salt) % 100 < control_pct -> control`.

Stable across runs by construction -- the same customer_id and salt always
hash to the same bucket, so re-running detection for a window that already
ran (a retry, a redeployment) reassigns nobody. That stability is what makes
"logged and never actioned" a real counterfactual rather than a coin flip that
could land differently on a retry.

The 80/20 split happens right after the arbiter in diagram 1 on purpose: arm
assignment is per (customer, window), the SAME granularity attribution was
just resolved at, not per opportunity_type. A customer cannot be control for
one opportunity type and treatment for another in the same window -- that
would let the same customer contaminate both arms at once.
"""

from __future__ import annotations

import zlib

from revenew.models import Arm

DEFAULT_CONTROL_PCT = 20


def assign_arm(customer_id: str, *, salt: str, control_pct: int = DEFAULT_CONTROL_PCT) -> Arm:
    bucket = zlib.crc32(f"{customer_id}{salt}".encode()) % 100
    return Arm.CONTROL if bucket < control_pct else Arm.TREATMENT
