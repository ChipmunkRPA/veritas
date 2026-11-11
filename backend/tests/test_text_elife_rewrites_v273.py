import pytest

from scripts import text_elife_rewrites_v273 as generation


def sample(group="demo"):
    before = "Prévious context.\n\n"
    target = " ".join(f"original{i}" for i in range(40))
    after = "\n\nNéxt context."
    a, b = len(before), len(before) + len(target)
    source = before + target + after
    return {
        "group": group,
        "text": source,
        "text_sha256": generation.text_sha(source),
        "body_slots": [0, 3],
        "edit_target_codepoints": [a, b],
        "paragraph_codepoints": [[0, a - 2], [a, b], [b + 2, len(source)]],
    }


def replacement(n=40):
    return " ".join(f"révision{i}" for i in range(n))


def test_two_exact_coordinates_preserve_original_context():
    row = sample()
    result = generation.rewrite(row, "smol2", replacement())
    a, b = row["edit_target_codepoints"]
    start, end = result["ai_spans"][0]
    assert result["status"] == "constructed" and start == a
    assert result["text"][start:end] == replacement()
    assert result["text"][:start] + row["text"][a:b] + result["text"][end:] == row["text"]
    assert result["text_sha256"] == generation.text_sha(result["text"])


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        ("", "empty_response"),
        ("Sure, " + replacement(), "wrapper_or_multiple_paragraphs"),
        (replacement() + "\n\nUnrequested continuation.", "wrapper_or_multiple_paragraphs"),
        (replacement() + " https://example.org/", "link_or_contact"),
        (replacement(9), "outside_word_bound"),
        (" ".join(f"original{i}" for i in range(40)), "unchanged"),
        (" ".join(f"original{i}" for i in range(20)) + " " + replacement(20), "source_copy"),
    ],
)
def test_rejections_preserve_identity_without_an_inference_input(output, reason):
    result = generation.rewrite(sample(), "qwen", output)
    assert result["status"] == "unavailable" and result["reason"] == reason
    assert result["raw_output_sha256"] == generation.text_sha(output)
    assert result["text"] is result["text_sha256"] is result["ai_spans"] is None


def test_authority_binding_rejects_shifted_target():
    rows = [sample(group) for group in generation.GROUPS]
    receipt = {
        "schema": "chipmunk.text-elife-source-intake/272",
        "status": "complete_intake",
        "planned_works": 9,
        "admitted_works": 5,
        "works": [{**row, "status": "admitted"} for row in rows],
    }
    assert generation.targets(receipt, rows) == rows
    rows[0] = {**rows[0], "edit_target_codepoints": [1, 2]}
    with pytest.raises(ValueError, match="differs from v272 receipt"):
        generation.targets(receipt, rows)
