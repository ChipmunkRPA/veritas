import pytest

from scripts.audio_matched_assets_v78 import REVISION, planned


def fixture():
    rows = [{"rfilename": f"voices/af_profile{n}.pt", "size": 100,
             "lfs": {"size": 100, "sha256": f"{n:064x}"}} for n in range(16)]
    prompts = [{"voice_asset": row["rfilename"], "voice_asset_sha256": row["lfs"]["sha256"]}
               for row in rows for _ in range(4)]
    return {"pending_matched_generation": prompts}, {"sha": REVISION, "cardData": {"license": "apache-2.0"}, "siblings": rows}


def test_fixed_prompts_deduplicate_to_exact_profile_assets():
    manifest, metadata = fixture()
    assert len(planned(manifest, metadata)) == 16


def test_changed_permission_or_profile_hash_cannot_be_used():
    manifest, metadata = fixture()
    metadata["cardData"]["license"] = "non-commercial"
    with pytest.raises(ValueError, match="Apache"):
        planned(manifest, metadata)
    manifest, metadata = fixture()
    manifest["pending_matched_generation"][0]["voice_asset_sha256"] = "0" * 64
    metadata["siblings"][0]["lfs"]["sha256"] = "1" * 64
    with pytest.raises(ValueError, match="binding"):
        planned(manifest, metadata)


def test_profile_path_cannot_escape_the_owned_asset_directory():
    manifest, metadata = fixture()
    manifest["pending_matched_generation"][0]["voice_asset"] = "../private.pt"
    with pytest.raises(ValueError, match="Safe"):
        planned(manifest, metadata)
