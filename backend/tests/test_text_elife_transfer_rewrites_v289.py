from collections import Counter

import pytest

from scripts import text_elife_transfer_rewrites_v289 as transfer


def source(group):
    prefix = "Prévious words.\n\n"
    target = " ".join(f"original{i}" for i in range(30))
    suffix = "\n\nNext words."
    text = prefix + target + suffix
    start, end = len(prefix), len(prefix) + len(target)
    return {
        "group": group,
        "body_slots": [0, 3],
        "text": text,
        "text_sha256": transfer.previous.text_sha(text),
        "paragraph_codepoints": [[0, start - 2], [start, end], [end + 2, len(text)]],
        "edit_target_codepoints": [start, end],
    }


def test_balanced_hash_split_is_input_order_independent_and_has_no_leakage():
    rows = [source(f"g-{i}") for i in range(26)]
    result = transfer.partition(rows)
    assert result == transfer.partition(rows[::-1])
    dev, confirmation = ([row["group"] for row in result[role]] for role in transfer.ROLES)
    assert len(dev) == len(confirmation) == 13 and not set(dev) & set(confirmation)
    assert sorted(
        dev + confirmation, key=lambda group: transfer.previous.text_sha(transfer.SALT + group)
    ) == (dev + confirmation)


def test_bound_source_offsets_and_excluded_denominator():
    rows = [source(f"g-{i}") for i in range(26)]
    works = [
        {key: value for key, value in row.items() if key != "text"} | {"status": "admitted"} for row in rows
    ]
    works += [{"status": "no_eligible_excerpt"}] * 8 + [{"status": "license_not_cc_by_4"}]
    receipt = {
        "schema": "chipmunk.text-elife-transfer-intake/286",
        "status": "complete_intake",
        "roster_sha256": transfer.ROSTER_SHA,
        "planned_works": 35,
        "admitted_source_groups": 26,
        "dispositions": dict(Counter(work["status"] for work in works)),
        "works": works,
    }
    assert transfer.validate(receipt, rows) == rows
    rows[0]["edit_target_codepoints"] = [1, 3]
    with pytest.raises(ValueError, match="source target changed"):
        transfer.validate(receipt, rows)


def test_scoped_worker_receives_only_its_role_without_changing_v273_rules(monkeypatch):
    rows = [source(f"g-{i}") for i in range(26)]
    split = transfer.partition(rows)
    seen = []

    def fake_worker(batch, output):
        scoped = globals()
        seen.append(
            (tuple(scoped["GENERATORS"])[batch], [row["group"] for row in scoped["verify"]()], output)
        )

    # The frozen worker resolves these two values from its isolated module globals.
    monkeypatch.setattr(transfer.previous, "worker", fake_worker)
    monkeypatch.setattr(transfer, "verify", lambda: split)
    transfer.worker(3, "fake-output")
    assert seen == [("confirmation:smol2", [row["group"] for row in split["confirmation"]], "fake-output")]

    row = split["confirmation"][0]
    raw = " ".join(f"rédaction{i}" for i in range(30))
    assert transfer.rewrite(row, "confirmation:smol2", raw) == {
        **transfer.previous.rewrite(row, "smol2", raw),
        "role": "confirmation",
    }
    unavailable = transfer.rewrite(row, "confirmation:smol2", "")
    assert unavailable["text"] is unavailable["ai_spans"] is None
    assert unavailable["reason"] == "empty_response"
