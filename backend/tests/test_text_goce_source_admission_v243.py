import pytest

from scripts import text_goce_source_admission_v243 as admission

TEXT_A = "the first private sentence contains enough words for a twelve word match today"
TEXT_B = "this unrelated source supplies a distinct set of exact comparison words for indexing"


def window(pair_id):
    return {
        "before": {"excerpt": f"Earlier unique archival excerpt {pair_id}"},
        "after": {"excerpt": f"Revised unique archival excerpt {pair_id}"},
    }


def pair(page_id, *windows):
    return {
        "page_id": page_id,
        "before_revision": page_id * 10,
        "after_revision": page_id * 10 + 1,
        "windows": list(windows),
    }


def roster():
    return {
        10: [pair(10, window(10)), pair(10, window(11))],
        20: [pair(20, window(20))],
        30: [pair(30, window(30)), pair(30, window(31), window(32))],
        40: [pair(40, window(40))],
    }


def test_exposure_receipt_preserves_source_counts_and_union_order_independent():
    a, b = ("packet-a", "a", [TEXT_A, TEXT_B]), ("packet-b", "b", [TEXT_A])
    receipt, union = admission.build_exposure((a, b))
    reverse, reverse_union = admission.build_exposure((b, a))
    assert receipt["text_entries"] == 3
    assert receipt["unique_fragments"] == len(union)
    assert receipt["combined_fragments_sha256"] == reverse["combined_fragments_sha256"]
    assert union == reverse_union == admission.text.fingerprints(TEXT_A) | admission.text.fingerprints(TEXT_B)
    assert TEXT_A not in str(receipt) and TEXT_B not in str(receipt)
    with pytest.raises(ValueError, match="supplied no fragments"):
        admission.build_exposure(
            (
                (
                    "missing",
                    "hash",
                    ["short"],
                ),
            )
        )


def test_only_the_two_fixed_approvals_are_passed_to_existing_rights_gate(monkeypatch):
    monkeypatch.setattr(
        admission,
        "source_citation",
        lambda pair, location, packets, introduction, evidence: {
            **location,
            "title": "private fixture source",
        },
    )
    review, attribution, planned = admission.review_records(roster(), [], {}, set())
    assert review["review_method"] == admission.REVIEW_METHOD and not review["external_human_review"]
    assert len(attribution) == len(review["approved"]) == 2
    assert [item["page_id"] for item in review["approved"]] == [20, 30]
    assert all(item["decision"] == "rights_unresolved" for item in review["held"])
    assert len(review["held"]) == 5
    assert admission.apply_review(planned, review, set()) == {
        "reviewed_windows": 2,
        "pending_windows": 5,
        "rights_admitted_pages": 2,
        "rights_exclusions": {},
    }


def test_overlap_in_either_version_prevents_source_approval(monkeypatch):
    groups = roster()
    groups[20][0]["windows"][0]["after"]["excerpt"] = TEXT_A
    monkeypatch.setattr(admission, "source_citation", lambda *_: {})
    with pytest.raises(ValueError, match="known project fragment match"):
        admission.review_records(groups, [], {}, admission.text.fingerprints(TEXT_A))


def test_generation_journal_checks_source_binding_even_for_excluded_rows():
    def make(original_hash):
        return b"\n".join(
            admission.prior.json_bytes(row).rstrip()
            for row in (
                {"originals_sha256": "source-manifest"},
                {"id": "source", "input_text_sha256": original_hash, "text": None},
            )
        )

    assert list(admission.email_generation(make("bound"), {"source": "bound"}, "source-manifest")) == []
    with pytest.raises(ValueError, match="source hash differs"):
        list(admission.email_generation(make("changed"), {"source": "bound"}, "source-manifest"))
