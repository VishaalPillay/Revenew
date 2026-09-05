"""Revenew Agent Channel: makes the merchant sellable to external AI shopping agents.

External shopping agents query products, negotiate bounded discounts against real
merchant policy (EnvelopeValidator), and complete checkout via Razorpay Payment Links.
"""

from revenew.agent.negotiate import create_checkout, negotiate

__all__ = ["negotiate", "create_checkout"]
