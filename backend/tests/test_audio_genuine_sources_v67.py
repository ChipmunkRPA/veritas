import copy

import pytest

from scripts.audio_genuine_sources_v67 import REVISION, assets


def metadata():
    return {"sha": REVISION, "cardData": {"license": ["cc-by-4.0"]},
            "siblings": [{"rfilename": f"clean/train.360/{index:04d}.parquet", "size": 100,
                          "lfs": {"size": 100, "sha256": str(index) * 64}} for index in range(4)]}


def test_acquisition_uses_fixed_revision_complete_files_and_owned_names():
    result = assets(metadata())
    assert len(result) == 4
    assert all(REVISION in row["url"] and "/" not in row["name"] for row in result)


def test_different_revision_or_permission_cannot_be_substituted():
    body = metadata()
    body["sha"] = "other"
    with pytest.raises(ValueError):
        assets(body)
    body = metadata()
    body["cardData"]["license"] = ["cc-by-nc-4.0"]
    with pytest.raises(ValueError):
        assets(body)


def test_inconsistent_publisher_sizes_fail_before_download():
    body = copy.deepcopy(metadata())
    body["siblings"][0]["lfs"]["size"] += 1
    with pytest.raises(ValueError, match="size"):
        assets(body)
