"""`build_adapter()`: the execution-mode switch that, before it existed, did
not exist anywhere in the codebase. `RAZORPAY_KEY_ID`/`RAZORPAY_KEY_SECRET`
used to be read by zero files under `revenew/`, and `LiveAdapter` was
constructed only in tests -- `decide_one_opportunity` hardcoded
`FixtureAdapter()` unconditionally. These tests are the guarantee that
flipping `REVENEW_EXECUTION_MODE=live` actually does something, and that it
never does something by ACCIDENT.
"""

from __future__ import annotations

from revenew.execute import razorpay as razorpay_module
from revenew.execute.razorpay import FixtureAdapter, LiveAdapter, build_adapter


def test_default_mode_is_fixture_even_with_credentials_present(monkeypatch):
    """The conservative default: real credentials sitting in the environment
    must never be enough on their own to start sending real payment links."""
    monkeypatch.setattr(razorpay_module, "REVENEW_EXECUTION_MODE", "fixture")
    monkeypatch.setattr(razorpay_module, "RAZORPAY_KEY_ID", "rzp_test_real")
    monkeypatch.setattr(razorpay_module, "RAZORPAY_KEY_SECRET", "real_secret")

    assert isinstance(build_adapter(), FixtureAdapter)


def test_live_mode_without_credentials_degrades_to_fixture_not_a_crash(monkeypatch, capsys):
    """A misconfigured environment (mode=live, no credentials) must fail into
    'nothing sent' rather than a traceback mid-decision -- and it must say
    so, not degrade silently."""
    monkeypatch.setattr(razorpay_module, "REVENEW_EXECUTION_MODE", "live")
    monkeypatch.setattr(razorpay_module, "RAZORPAY_KEY_ID", "")
    monkeypatch.setattr(razorpay_module, "RAZORPAY_KEY_SECRET", "")

    assert isinstance(build_adapter(), FixtureAdapter)
    assert "falling back to FixtureAdapter" in capsys.readouterr().out


def test_live_mode_with_only_one_credential_still_degrades(monkeypatch):
    """Half a credential pair is not a credential pair."""
    monkeypatch.setattr(razorpay_module, "REVENEW_EXECUTION_MODE", "live")
    monkeypatch.setattr(razorpay_module, "RAZORPAY_KEY_ID", "rzp_test_real")
    monkeypatch.setattr(razorpay_module, "RAZORPAY_KEY_SECRET", "")

    assert isinstance(build_adapter(), FixtureAdapter)


def test_live_mode_with_both_credentials_returns_a_real_adapter(monkeypatch):
    assert LiveAdapter is not FixtureAdapter  # sanity: distinguishable types
    monkeypatch.setattr(razorpay_module, "REVENEW_EXECUTION_MODE", "live")
    monkeypatch.setattr(razorpay_module, "RAZORPAY_KEY_ID", "rzp_test_real")
    monkeypatch.setattr(razorpay_module, "RAZORPAY_KEY_SECRET", "real_secret")

    adapter = build_adapter()
    assert isinstance(adapter, LiveAdapter)
