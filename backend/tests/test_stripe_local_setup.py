"""The local setup helper must never enable live billing or disclose credentials."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("stripe_local", Path(__file__).parents[1] / "scripts/stripe_local.py")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def test_live_configuration_is_never_overwritten(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    before = "VERITAS_BILLING_MODE=live\nVERITAS_STRIPE_SECRET_KEY=sk_live_do_not_touch\n"
    env.write_text(before)
    monkeypatch.setattr(helper, "ENV_FILE", env)
    with pytest.raises(RuntimeError, match="refuses to replace"):
        helper.configure()
    assert env.read_text() == before


def test_redaction_covers_cli_test_live_restricted_and_webhook_keys():
    line = "keys sk_test_example rk_test_example pk_live_example whsec_example"
    assert helper.SECRET.sub("[redacted]", line) == "keys [redacted] [redacted] [redacted] [redacted]"


def test_listener_refuses_a_changed_webhook_secret_before_starting(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("VERITAS_BILLING_MODE=test\nVERITAS_STRIPE_WEBHOOK_SECRET=whsec_old\n")
    monkeypatch.setattr(helper, "ENV_FILE", env)
    monkeypatch.setattr(helper, "signing_secret", lambda: "whsec_new")
    with pytest.raises(RuntimeError, match="secret changed"):
        helper.listen()
