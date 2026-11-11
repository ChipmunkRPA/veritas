import numpy as np

from scripts import audio_short_edit_cases_v172 as study


def input_row(slot, voice, lane="genuine", status="qualified"):
    return {"slot": slot, "voice": voice, "group_id": slot + "-group", "lane": lane, "status": status}


def manifest():
    return {
        "role": "development",
        "foregrounds": [
            input_row("genuine-0", "voice-a"),
            input_row("genuine-1", "voice-b"),
            input_row("clone-0", "voice-c", "clone"),
        ],
        "noise": [{"slot": "noise-00", "status": "qualified"}],
    }


def samples(seed):
    generator = np.random.default_rng(seed)
    return generator.integers(-4000, 4000, size=study.SAMPLES, dtype=np.int16).astype("<i2").tobytes()


def test_plan_is_deterministic_role_local_and_donor_independent():
    bundle = manifest()
    groups = study.comparisons(bundle)
    assert groups == study.comparisons({**bundle, "foregrounds": bundle["foregrounds"][::-1]})
    assert sum(group["planned_cases"] for group in groups) == 45
    ai = next(group for group in groups if group["family"] == "ai")
    assert ai["role"] == "development"
    assert ai["genuine_host_slot"] != ai["genuine_donor_slot"]
    assert ai["genuine_host_group"] != ai["genuine_donor_group"]
    assert ai["noise_slot"] == "noise-00"


def test_actual_authored_wavs_keep_all_paired_noisy_sample_residuals(tmp_path):
    bundle = manifest()
    groups = study.comparisons(bundle)
    ai = next(group for group in groups if group["family"] == "ai")
    raw = {
        row["slot"]: samples(index) for index, row in enumerate([*bundle["foregrounds"], *bundle["noise"]])
    }
    output = {}
    for index, (row, payload) in enumerate(study.construct(ai, raw)):
        assert row["status"] == "constructed"
        path = tmp_path / f"case-{index}.wav"
        study.write_case(path, payload)
        output[(row["arm"], row["condition"])] = (row, study.read_pcm(path))
    assert len(output) == 21
    assert len({row["gain"] for row, _ in output.values()}) == 1
    for condition, level in study.LEVELS[1:]:
        residuals, hashes = set(), set()
        for arm, _ in study.arms("ai"):
            clean = np.frombuffer(output[(arm, "clean")][1], dtype="<i2")
            row, noisy = output[(arm, condition)]
            assert row["nominal_snr_db"] == level
            residual = np.frombuffer(noisy, dtype="<i2").astype(np.int32) - clean
            residuals.add(residual.astype("<i2").tobytes())
            hashes.add(row["shared_noise_pcm_sha256"])
        assert len(residuals) == len(hashes) == 1
        assert study.sources.digest(residuals.pop()) == hashes.pop()
    ordinary, ordinary_pcm = output[("genuine_to_genuine_fade20", "clean")]
    mixed, mixed_pcm = output[("genuine_to_ai_fade20", "clean")]
    start, end = study.controls.TARGET_SAMPLES
    assert ordinary_pcm[: 2 * start] == mixed_pcm[: 2 * start]
    assert ordinary_pcm[2 * end :] == mixed_pcm[2 * end :]
    assert ordinary["matched_host_pcm_sha256"] == mixed["matched_host_pcm_sha256"]
    assert ordinary["label"]["ai_support_samples"] is None
    assert mixed["label"]["ai_support_samples"] == [start + 1, end - 1]
    assert mixed["label"]["pure_ai_samples"] == [start + 319, end - 319]
    assert all(start <= left < right <= end for left, right in mixed["observed_changed_intervals"])


def test_genuine_plan_keeps_four_edit_arms_with_no_ai_truth():
    bundle = manifest()
    group = next(group for group in study.comparisons(bundle) if group["family"] == "genuine")
    rows = study.construct(group, {group["source_slot"]: samples(1), "noise-00": samples(2)})
    assert len(rows) == 12
    assert all(row["label"]["speech_origin"] == "genuine_source_only" for row, _ in rows)
    assert all(row["status"] == "constructed" for row, _ in rows)


def test_missing_donor_and_noise_preserve_planned_null_denominator():
    ai = next(group for group in study.comparisons(manifest()) if group["family"] == "ai")
    rows = study.construct(ai, {ai["source_slot"]: samples(1), ai["genuine_host_slot"]: samples(2)})
    assert len(rows) == 21
    complete = [row for row, payload in rows if payload is not None]
    assert len(complete) == 1 and complete[0]["arm"] == "fully_synthetic"
    assert complete[0]["condition"] == "clean" and complete[0]["label"]["ai_support_samples"] == [0, 48_000]
    assert {row["reason"] for row, _ in rows if row["status"] == "unavailable"} == {
        "noise_parent_unavailable",
        "independent_genuine_host_or_donor_unavailable",
    }
