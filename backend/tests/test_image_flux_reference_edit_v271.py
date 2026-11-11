import hashlib

import numpy as np
import pytest

from scripts import image_flux_reference_edit_v271 as smoke


def test_freeze_plan_binds_exact_assets_fixtures_and_guard():
    frozen = smoke.plan()
    assert len(frozen["assets"]) == 11
    assert sum(row["bytes"] for row in frozen["assets"].values()) == 4_619_700_117
    assert frozen["acquisition_receipt_sha256"] == smoke.ACQUISITION_SHA
    assert frozen["size"] == [256, 256]
    assert frozen["max_generator_calls"] == 3 and frozen["steps"] == 4
    assert frozen["max_sampled_tree_rss_bytes"] == frozen["initial_ram_bytes"] == 6 * 1024**3
    assert (
        frozen["fixtures"]["interior"]["source_rgb_sha256"] != frozen["fixtures"]["edge"]["source_rgb_sha256"]
    )


@pytest.mark.parametrize("case", ["interior", "edge"])
def test_composition_discards_raw_changes_outside_declared_mask(case):
    source, mask = smoke.fixture(case)
    assert bool(mask[:, 0].any()) == (case == "edge")
    raw = source.copy()
    raw[0, 0] ^= 17
    y, x = np.argwhere(mask == 255)[0]
    raw[y, x] ^= 1
    edited, changed, result = smoke.compose(source, raw, mask)
    assert np.array_equal(edited[mask == 0], source[mask == 0])
    assert np.array_equal(edited[mask == 255], raw[mask == 255])
    assert changed[y, x] == 255 and np.count_nonzero(changed) == 1
    assert result["raw_changes_inside"] == result["raw_changes_outside"] == 1
    assert result["outside_composite_rgb_sha256"] == result["outside_source_rgb_sha256"]
    assert result["declared_mask_sha256"] != result["truth_mask_sha256"]
    assert set(result["ordinary_controls"]) == {"tone", "blur"}


def test_wrong_geometry_mask_and_noop_refuse_qualification():
    source, mask = smoke.fixture("interior")
    with pytest.raises(ValueError, match="geometry"):
        smoke.compose(source, source[:-1], mask)
    broken = mask.copy()
    broken[0, 0] = 10
    with pytest.raises(ValueError, match="binary mask"):
        smoke.compose(source, source, broken)
    outside_only = source.copy()
    outside_only[0, 0] ^= 1
    with pytest.raises(ValueError, match="Nonempty intended and observed edit"):
        smoke.compose(source, outside_only, mask)


def test_pinned_bytes_and_paths_are_verified(tmp_path):
    root = tmp_path.resolve()
    path = root / "fixture.bin"
    path.write_bytes(b"good")
    item = {"bytes": 4, "lfs_sha256": hashlib.sha256(b"good").hexdigest(), "git_blob": None}
    smoke.check_asset(root, path.name, item)
    path.write_bytes(b"bad!")
    with pytest.raises(ValueError, match="LFS digest"):
        smoke.check_asset(root, path.name, item)
    item["lfs_sha256"] = None
    item["git_blob"] = hashlib.sha1(b"blob 4\0bad!").hexdigest()
    smoke.check_asset(root, path.name, item)
    relocated = root / "link.bin"
    relocated.symlink_to(path)
    with pytest.raises(ValueError, match="Direct pinned asset"):
        smoke.check_asset(root, relocated.name, item)


def test_png_roundtrips_authored_rgb_without_implicit_color_conversion(tmp_path):
    source, _ = smoke.fixture("edge")
    target = tmp_path / "fixture.png"
    assert smoke.save_png(target, source) == smoke.sha(target)
    with pytest.raises(FileExistsError):
        smoke.save_png(target, source)
