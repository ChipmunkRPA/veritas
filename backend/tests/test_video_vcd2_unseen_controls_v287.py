from scripts import video_vcd2_unseen_controls_v287 as study


def test_selects_fixed_strata_without_reusing_prior_recordings_or_cameras():
    raw, benchmark, old = {}, {}, []
    for n, stratum in enumerate(study.catalog.STRATA):
        for position in range(4):
            stem = f"{n * 4 + position:032x}"
            camera = f"vid_{n:04x}&pid_{position:04x}"
            raw[stem] = {
                "log_recording_camera_code": camera,
                "video_resolution": "1280x720",
                "si_ti_group": stratum,
            }
            benchmark[stem] = {**raw[stem], "group": "TH", "relative_path": f"benchmarking/mkv/TH/{stem}.mkv"}
            if position == 0:
                old.append({"recording_stem": stem, "camera_type": camera})
            if position == 1:
                raw[stem]["log_recording_camera_code"] = old[-1]["camera_type"]

    selected = study.select(raw, benchmark, old)
    assert len(selected) == len({row["camera_type"] for row in selected}) == 8
    assert [row["stratum"] for row in selected] == [item for item in study.catalog.STRATA for _ in range(2)]
    assert all(row["recording_stem"] not in {prior["recording_stem"] for prior in old} for row in selected)
    assert all(row["camera_type"] not in {prior["camera_type"] for prior in old} for row in selected)
    assert selected == study.select(raw, dict(reversed(list(benchmark.items()))), old)
