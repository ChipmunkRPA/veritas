from copy import deepcopy

import pytest

from scripts.review_speech_paragraph_transfer import masks, text_sha


def fixture():
    paragraphs = [f"{word} " * 17 + f"{word}." for word in ("First", "Second", "Third", "Fourth")]
    text = "\n\n".join(paragraphs)
    spans = []
    start = 0
    for paragraph in paragraphs:
        spans.append([start, start + len(paragraph)])
        start += len(paragraph) + 2
    case = {"text": text, "paragraph_codepoints": spans, "ai_spans": [spans[2]]}
    answer = {"current": [{"start": a, "end": b, "raw": score}
                          for (a, b), score in zip(spans, (.4, .9, .7, .4), strict=True)], "paragraphs": []}
    for (a, b), score in zip(spans[1:-1], (.49, .5), strict=True):
        answer["paragraphs"].append({"spans": [{"start": a, "end": b}], "input_sha256": text_sha(text[a:b]),
                                      "score": score, "score_exact_hex": score.hex(), "token_coverage": {
                                          "tokens": 23, "covered_tokens": 23, "token_windows": 1, "omitted_tokens": 0,
                                      }})
    return case, answer


def test_isolated_replay_uses_the_matching_whole_paragraph():
    case, answer = fixture()
    marks, truth, count = masks(case, answer)
    assert count == 2 and truth == marks["isolated"]
    assert marks["paragraph"] & truth == 0
    assert marks["raw"] & truth == truth


@pytest.mark.parametrize("field,value", [("input_sha256", "wrong"), ("score", .75),
                                         ("token_coverage", {"tokens": 23, "covered_tokens": 22})])
def test_isolated_replay_rejects_tampered_input_or_incomplete_response(field, value):
    case, answer = fixture()
    altered = deepcopy(answer)
    altered["paragraphs"][0][field] = value
    with pytest.raises(ValueError, match="recorded whole paragraph"):
        masks(case, altered)


def test_current_scopes_cannot_omit_a_source_character():
    case, answer = fixture()
    answer["current"][0]["start"] += 1
    with pytest.raises(ValueError, match="omit measured source characters"):
        masks(case, answer)
