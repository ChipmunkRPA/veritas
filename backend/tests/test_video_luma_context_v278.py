import numpy as np

from scripts import video_luma_context_v278 as study


def roles(background, face):
    rows = {}
    for role, background_level, face_level in zip(study.ROLES, background, face, strict=True):
        rows[role] = [
            {
                "luma": np.full(256, background_level),
                "context": np.ones(256, dtype=bool),
                "face": face_level,
                "face_sites": 4,
            }
            for _ in range(3)
        ]
    return rows


def test_local_broad_and_monotone_shifts():
    local = study.measure(roles([55, 55, 55], [55, 80, 55]))
    broad = study.measure(roles([55, 80, 55], [55, 80, 55]))
    monotone = study.measure(roles([55, 80, 105], [55, 80, 105]))
    assert local["face_shift"] == broad["face_shift"] == 25
    assert local["context_affected_fraction"] == 0
    assert broad["context_affected_in_face_direction_fraction"] == 1
    assert monotone["context_affected_fraction"] == 0
    assert monotone["context_affected_in_face_direction_fraction"] is None


def test_common_support_excludes_every_moving_face_box():
    sampled = roles([55, 80, 55], [55, 80, 55])
    sampled["before"][0]["context"][:16] = False
    sampled["after"][0]["context"][16:32] = False
    assert study.measure(sampled)["context_sites"] == 224
    sampled["during"].clear()
    assert study.measure(sampled)["status"] == "unavailable"


def test_native_grid_color_and_half_box_context_margin():
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[64:128, 64:128, 0] = 200
    row = study.sample_frame(rgb, {"x": 64, "y": 64, "w": 64, "h": 64})
    assert row["luma"].shape == (16, 16)
    assert np.isclose(row["face"], 59.8)
    assert (~row["context"]).sum() == 64
