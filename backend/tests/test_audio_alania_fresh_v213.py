from scripts import audio_alania_fresh_v213 as study


def metadata(index):
    return {
        "id": f"id-{index}",
        "voice_id": f"voice-{index}",
        "text_doc_id": f"doc-{index}",
        "row_group": 0,
        "row_offset": index,
        "duration": 3.5,
        "text_source": "domain",
        "license": "CC-BY-4.0",
        "voice_type": "designed-identity",
        "render_mode": "reference",
        "generator": study.roster.GENERATOR,
        "postfilter": "resclip",
    }


def test_previous_identity_or_location_excludes_candidate():
    previous = metadata(0)
    candidates = [metadata(i) for i in range(1, 8)]
    for candidate, key in zip(candidates, study.IDENTITIES, strict=False):
        candidate[key] = previous[key]
    candidates[3]["row_offset"] = previous["row_offset"]
    candidates[4]["duration"] = 2.99
    candidates[5]["license"] = "CC-BY-NC-4.0"
    assert study.eligible_rows(candidates, [previous]) == [candidates[-1]]


def test_short_or_repeated_audio_is_unavailable():
    row = {"status": "qualified", "samples": 48_000, **{key: "a" for key in study.FINGERPRINTS}}
    empty = {key: set() for key in study.FINGERPRINTS}
    assert study.qualification_reason(row, empty) is None
    assert (
        study.qualification_reason({**row, "samples": 47_999}, empty)
        == "shorter_than_three_seconds_after_decode"
    )
    assert (
        study.qualification_reason(row, {**empty, "pcm_sha256": {"a"}})
        == "overlap_with_prior_or_selected_audio"
    )
    assert (
        study.qualification_reason({"status": "unavailable", "reason": "decode_failure"}, empty)
        == "decode_failure"
    )
