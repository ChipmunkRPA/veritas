import pytest

from scripts import audio_voxforge_tr_genuine_v212 as study


def publisher_listing(aliases):
    paths = [f"{alias}-2026010{day}-abc.tgz" for alias in aliases for day in range(1, 4)]
    return "\n".join(f'<a href="{path}">{path}</a>' for path in paths).encode()


def test_roster_excludes_both_previous_plans_and_does_not_fill_from_anonymous():
    excluded = {f"old{i}" for i in range(36)} | {"historical"}
    candidates = [f"new{i}" for i in range(23)]
    aliases = [*sorted(excluded), *candidates, "anonymous1"]
    plan, available = study.source_plan(publisher_listing(aliases), excluded)
    assert study.source_plan(publisher_listing(reversed(aliases)), excluded) == (plan, available)
    assert available == 23
    assert len(plan) == len({row["contributor"] for row in plan}) == 16
    assert all(row["contributor"] in candidates and len(row["sessions"]) == 2 for row in plan)
    assert [row["slot"] for row in plan] == [f"genuine-{index:02d}" for index in range(16)]


def test_prior_v192_full_and_v199_full_and_cropped_identities_are_excluded():
    controls = [
        {"prompt_sha256": "old-prompt", "source_sha256": "old-original", "pcm_sha256": "old-pcm"},
        {
            "prompt_sha256": "newer-prompt",
            "source_sha256": "newer-original",
            "pcm_sha256": "newer-crop",
            "full_pcm_sha256": "newer-full",
        },
    ]
    prompts, fingerprints = study.exclusion_keys(controls, {"text:historical", "pcm:historical"})
    assert prompts == {"old-prompt", "newer-prompt"}
    assert fingerprints == {
        "bytes:old-original",
        "bytes:newer-original",
        "pcm:old-pcm",
        "pcm:newer-crop",
        "pcm:newer-full",
        "text:historical",
        "pcm:historical",
    }


@pytest.mark.parametrize("overlap", ["prompt_sha256", "source_sha256", "pcm_sha256", "full_pcm_sha256"])
def test_aggregation_fails_a_qualified_repeat_and_keeps_unavailable_slots(monkeypatch, overlap):
    key = "identity-overlap"
    control = {"prompt_sha256": "prior-text", "source_sha256": "prior-bytes", "pcm_sha256": "prior-crop"}
    # Check either previously observed PCM identity even when the new overlap is on the other extent.
    control["pcm_sha256" if overlap == "full_pcm_sha256" else overlap] = key
    monkeypatch.setattr(study, "prior_controls", lambda: ([], [control], {}))
    monkeypatch.setattr(study, "read", lambda _: {"keys": []})
    plan = [{"slot": f"genuine-{index:02d}"} for index in range(study.GROUPS)]
    rows = [{**row, "status": "unavailable", "language": "tr", "label": 0} for row in plan]
    assert study.aggregation(rows, [], plan, [])["case_outcomes"] == {"unavailable": 16}
    archive = {"slot": "genuine-00", "archive": "test.tgz", "sha256": "archive", "status": "downloaded"}
    rows[0] |= {
        "status": "qualified",
        "archive": archive["archive"],
        "archive_sha256": archive["sha256"],
        "license": "GPL-3.0-or-later",
        "stock_gpl_reference": {"sha256": next(iter(study.prior.rights.STOCK_LICENSES))},
        "selection": "pause_bounded_crop",
        "prompt_sha256": "fresh-text",
        "source_sha256": "fresh-original",
        "pcm_sha256": "fresh-crop",
        "full_pcm_sha256": "fresh-full",
    }
    assert study.aggregation(rows, [], plan, [archive])["case_outcomes"] == {
        "qualified": 1,
        "unavailable": 15,
    }
    rows[0][overlap] = key
    with pytest.raises(ValueError, match="Qualified source repeats prior identity"):
        study.aggregation(rows, [], plan, [archive])
