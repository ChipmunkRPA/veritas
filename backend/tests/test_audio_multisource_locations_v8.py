import pytest

from scripts.export_audio_multisource_locations_v8 import checked_url, direct_asset


def test_location_supplement_cannot_silently_change_original_asset():
    with pytest.raises(ValueError, match="differs from frozen original"):
        direct_asset({"sha256": "a", "bytes": 1, "asset": {"sha256": "b", "bytes": 1}})


def test_source_locations_refuse_private_or_credentialed_urls():
    for url in (
        "https://example.com/private",
        "https://huggingface.co/a?token=example",
        "https://user:password@huggingface.co/a",
        "file:///private/data",
    ):
        with pytest.raises(ValueError, match="public publisher"):
            checked_url(url)
    assert checked_url("https://huggingface.co/datasets/public/resolve/revision/file.wav").endswith(
        "file.wav"
    )
