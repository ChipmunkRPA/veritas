import json

import pytest

from scripts import writing_two_sided_transfer as transfer


def test_holdout_uses_the_exact_prior_fit(monkeypatch):
    original = transfer.fit()
    published = transfer.read(transfer.PRIOR_REPORT)
    assert original == published["fitted"]
    published["fitted"]["arms"]["two_sided"]["point"] += 0.01
    monkeypatch.setattr(transfer, "read", lambda _: published)
    with pytest.raises(ValueError, match="Published fit"):
        transfer.fit()


def test_new_holdout_refuses_prior_author(tmp_path):
    prior = transfer.read(transfer.PRIOR_REPORT)
    author = prior["source_roster"][0]["author"]
    (tmp_path / "source.json").write_text(json.dumps({"groups": [{"authors": [author]}]}))
    with pytest.raises(ValueError, match="author groups must be disjoint"):
        transfer.lock(tmp_path)
