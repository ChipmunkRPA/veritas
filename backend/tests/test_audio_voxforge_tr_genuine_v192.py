import io
import tarfile

import pytest

from scripts import audio_voxforge_tr_genuine_v192 as study

STOCK = (
    b"GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n"
    b"This alternative is allowed only occasionally and noncommercially, and only if you received the object code.\n"
)
NOTICE = (
    b"These files are free software: you can redistribute them and/or modify "
    b"them under the terms of the GNU General Public License as published by "
    b"the Free Software Foundation, either version 3 of the License, or "
    b"(at your option) any later version."
)


def metadata(monkeypatch, *, reference=STOCK, notice=NOTICE, readme=b"User Name: Example\nLanguage: TR\n"):
    source = study.source
    monkeypatch.setattr(study, "STOCK_LICENSES", frozenset({source.digest(STOCK)}))
    monkeypatch.setattr(study, "STOCK_NORMALIZED", source.digest(source.norm(STOCK.decode()).encode()))
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:") as archive:
        files = {
            "session/etc/README": readme,
            "session/etc/gpl_license.txt": reference,
            "session/etc/PROMPTS": b"tr-0001 Ornek metin.\n",
        }
        if notice is not None:
            files["session/etc/LICENSE"] = notice
        for path, payload in files.items():
            member = tarfile.TarInfo(path)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    with tarfile.open(fileobj=io.BytesIO(data.getvalue()), mode="r:") as archive:
        return study.session_metadata(archive, source.regular_members(archive), "example")


def test_exact_reference_does_not_block_explicit_session_grant(monkeypatch):
    prompts, evidence = metadata(monkeypatch)
    assert set(prompts) == {"tr-0001"}
    assert evidence["license"] == "GPL-3.0-or-later"
    assert evidence["license_receipt"]["sha256"] == study.source.digest(NOTICE)
    assert evidence["stock_gpl_reference"]["sha256"] == study.source.digest(STOCK)


@pytest.mark.parametrize(
    ("overrides", "failure"),
    [
        ({"reference": STOCK + b"This dataset is for noncommercial research only.\n"}, "Exact audited stock"),
        ({"notice": None}, "Separate exact GPL reference"),
        ({"notice": STOCK}, "Conflicting per-session"),
        ({"notice": NOTICE + b" Data for noncommercial research only."}, "Conflicting per-session"),
        ({"readme": b"User Name: Example\nLanguage: TR\nResearch use only.\n"}, "Conflicting per-session"),
        ({"readme": b"User Name: Example\nLanguage: EN\n"}, "Exact declared Turkish language"),
    ],
)
def test_unknown_stock_and_explicit_session_restrictions_still_fail(monkeypatch, overrides, failure):
    with pytest.raises(ValueError, match=failure):
        metadata(monkeypatch, **overrides)


def test_unavailable_slot_is_preserved_and_does_not_claim_license():
    slots = [{"slot": "genuine-00"}]
    rows = [{"slot": "genuine-00", "status": "unavailable", "language": "tr", "label": 0}]
    attempts = [{"status": "session_unavailable"}]
    result = study.aggregation(rows, attempts, slots, [])
    assert result["planned_cases"] == 1
    assert result["case_outcomes"] == {"unavailable": 1}
    assert result["qualified_licenses"] == {}
