"""Revenew: a decision layer above Razorpay.

Finds where a merchant's revenue is leaking, proposes one bounded commercial
action per opportunity, learns which actions work from an outcome ledger, and
reports incremental impact against a held-out control arm.

See SYSTEM_DESIGN.md for the full component inventory, and README.md for the
one-paragraph pitch. This package is the runtime: it opens `revenew.db` and
nothing else. `harness/` is a separate package that knows ground truth and
grades this package from the outside -- see harness/__init__.py.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
