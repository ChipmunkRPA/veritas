"""Formatting counterfactuals cannot change content, split words or invent source positions."""

import pytest

from scripts.text_format_views_v93 import correspondence, ingress_views, project_labels, rewrap


def test_unicode_punctuation_combining_marks_and_invisible_nonspace_are_retained():
    source = "  Café\u00a0naïve 👩🏽‍💻 e\u0301.\n\nSecond\tpart\u200b! "
    expected = "".join(c for c in source if not c.isspace())
    for view in ingress_views(source):
        assert "".join(c for c in view.text if not c.isspace()) == expected
        mapped = [p for p in view.source_positions if p is not None]
        assert mapped == [i for i, c in enumerate(source) if not c.isspace()]
        assert all(source[p] == view.text[i] for i, p in enumerate(view.source_positions) if p is not None)


@pytest.mark.parametrize(
    "target", ["AIgenerated text.", "A I generated text.", "AI generated text!", "Ai generated text."]
)
def test_equal_or_almost_equal_character_sequences_do_not_authorize_content_or_word_changes(target):
    with pytest.raises(ValueError):
        correspondence("AI generated text.", target)


def test_wrap_keeps_long_tokens_whole_and_changes_only_existing_whitespace():
    source = "one " + ("x" * 120) + " two\tthree"
    rendered = rewrap(source, 10)
    assert "x" * 120 in rendered and rendered.count("\n") == 2
    assert correspondence(source, rendered)
    for width in (0, -1, True, 3.5):
        with pytest.raises(ValueError):
            rewrap(source, width)


def test_ingress_normalization_identifies_the_horizontal_space_placebo():
    source = "First  paragraph\twith words.\n\nSecond paragraph has more words."
    views = {v.name: v for v in ingress_views(source)}
    assert views["horizontal_space_control"].input_text != source
    assert views["horizontal_space_control"].text == views["original"].text
    assert views["single_spaces"].text == views["wrap_48"].text
    assert views["single_spaces"].text != views["original"].text


def test_noncontiguous_source_indices_and_whitespace_only_spans_are_not_filled_in():
    source = "human\t\t\n\nAI words"
    view = next(v for v in ingress_views(source) if v.name == "single_spaces")
    start = view.text.index("AI")
    assert view.project(start, start + 2) == {source.index("A"), source.index("I")}
    assert view.project(start - 1, start) == set()
    assert view.project(0, len(view.text)) == {i for i, c in enumerate(source) if not c.isspace()}
    for a, b in [(True, 2), (-1, 2), (1, 1), (0, len(view.text) + 1)]:
        with pytest.raises(ValueError):
            view.project(a, b)


def test_projected_provenance_preserves_exact_human_and_ai_characters():
    source = "human\n\nAI\tpassage"
    labels = [0 if i < source.index("A") else 1 for i in range(len(source))]
    for view in ingress_views(source):
        mapped = project_labels(source, labels, view)
        assert sum(v == 1 for v in mapped) == sum(
            labels[i] == 1 and not c.isspace() for i, c in enumerate(source)
        )
        assert sum(v == 0 for v in mapped) == 5
    labels[source.index("A")] = -1
    with pytest.raises(ValueError, match="explicit provenance"):
        project_labels(source, labels, ingress_views(source)[0])


def test_empty_or_whitespace_only_input_has_no_authorship_positions():
    for source in ("", " \t\n\n"):
        for view in ingress_views(source):
            assert view.text == "" and view.source_positions == ()


def test_packet_retains_every_requested_view_but_deduplicates_only_identical_effective_inputs():
    from scripts.prepare_text_format_audit_v93 import build_case
    from scripts.text_format_views_v93 import text_sha

    source = "Human passage.\n\nAI passage."
    labels = [0 if i < source.index("AI") else 1 for i in range(len(source))]
    case = {
        "id": "authored-fixture",
        "group": "fixture-group",
        "role": "diagnostic",
        "recipe": "mixed",
        "text": source,
        "text_sha256": text_sha(source),
        "labels": labels,
    }
    packet = build_case(case)
    assert len(packet["requested_views"]) == 5 and len(packet["effective"]) == 2
    assert packet["effective"][0]["views"] == ["original", "horizontal_space_control"]
    assert set(packet["effective"][1]["views"]) == {"single_spaces", "wrap_48", "paragraph_40_words"}
    for effective in packet["effective"]:
        assert sum(v == 1 for v in effective["labels"]) == 10


def test_packet_refuses_changed_source_identity_and_incomplete_character_truth():
    from scripts.prepare_text_format_audit_v93 import build_case
    from scripts.text_format_views_v93 import text_sha

    case = {
        "id": "fixture",
        "group": "fixture",
        "role": "diagnostic",
        "recipe": "human",
        "text": "Real text.",
        "text_sha256": text_sha("Real text."),
        "labels": [0] * 10,
    }
    with pytest.raises(ValueError, match="identity"):
        build_case({**case, "text": "Fake text."})
    with pytest.raises(ValueError, match="source labels"):
        build_case({**case, "labels": [0] * 9})
