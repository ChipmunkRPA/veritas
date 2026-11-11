import hashlib
import io
import time

import numpy as np
import pytest
from PIL import Image
from requests.structures import CaseInsensitiveDict

from scripts import image_bg_real_acquisition_v221 as subject


def png(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


class Response:
    def __init__(self, status, headers, payload=b""):
        self.status_code = status
        self.headers = CaseInsensitiveDict(headers)
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def iter_content(self, _chunk):
        yield self.payload


class Session:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.urls = []

    def get(self, url, **kwargs):
        assert kwargs["allow_redirects"] is False and kwargs["stream"] is True
        self.urls.append(url)
        return next(self.responses)


def test_pinned_resolver_and_lfs_digest_and_restricted_redirects():
    body = b"authored response"
    asset = "v2/masks/edit/a/example.png"
    etag = subject.digest(body)
    target = "https://cas-bridge.xethub.hf.co/key?signature=test"
    session = Session(
        Response(
            302,
            {
                "X-Repo-Commit": subject.roster.REVISION,
                "X-Linked-Etag": f'"{etag}"',
                "X-Linked-Size": str(len(body)),
                "Location": target,
            },
        ),
        Response(200, {"Content-Length": str(len(body))}, body),
    )
    got, proof = subject.download(session, asset, None, time.monotonic() + 5)
    assert got == body and proof["publisher_asset_commitment"] == etag
    assert session.urls == [subject.download_url(asset), target]
    assert "signature" not in str(proof)
    for redirect in (
        "https://localhost/private",
        "https://cas-bridge.xethub.hf.co:invalid/a",
        "https://huggingface.co/api/resolve-cache/datasets/unrelated/main/a",
    ):
        with pytest.raises(subject.Rejected, match="redirect"):
            subject.redirect_to(subject.download_url(asset), redirect, asset)
    wrong = Session(Response(200, {"X-Repo-Commit": "wrong"}, body))
    with pytest.raises(subject.Rejected, match="revision"):
        subject.download(wrong, asset, None, time.monotonic() + 5)


def test_fallback_is_exact_bytes_and_git_commitment_checks_blob_identity():
    body = b"mask payload"
    git = hashlib.sha1(b"blob 12\0" + body, usedforsecurity=False).hexdigest()
    assert subject.verify_asset_bytes(body, None, git) == subject.digest(body)
    assert subject.verify_asset_bytes(body, None, None) == subject.digest(body)
    with pytest.raises(subject.Rejected, match="publisher_image_sha256"):
        subject.verify_asset_bytes(body, "a" * 64, None)
    with pytest.raises(subject.Rejected, match="publisher_asset_commitment"):
        subject.verify_asset_bytes(body, None, "a" * 40)
    assert subject.commitment({"ETag": "a" * 40}) is None


def test_response_size_and_shape_are_checked():
    session = Session(
        Response(
            200, {"X-Repo-Commit": subject.roster.REVISION, "Content-Length": str(subject.MAX_BYTES + 1)}, b""
        )
    )
    with pytest.raises(subject.Rejected, match="asset_size_limit"):
        subject.download(session, "v2/images/a/example.png", None, time.monotonic() + 5)
    body = png(np.zeros((64, 70, 3), np.uint8))
    assert subject.png_pixels(body, (70, 64)).shape == (64, 70, 3)
    with pytest.raises(subject.Rejected, match="geometry"):
        subject.png_pixels(body, (64, 70))
    alpha = png(np.zeros((64, 70, 4), np.uint8))
    with pytest.raises(subject.Rejected, match="transparent"):
        subject.png_pixels(alpha, (70, 64))


def test_binary_support_does_not_replace_exact_truth():
    original = np.zeros((64, 64, 3), np.uint8)
    edited, mask = original.copy(), original.copy()
    mask[10:20, 10:20] = 255
    edited[11, 11, 1] = 1
    exact, support = subject.edit_truth(original, edited, mask, 100 / 4096)
    assert int(exact.sum()) == 1 and exact[11, 11] and support == 100
    edited[10, 9] = 1
    with pytest.raises(subject.Rejected, match="outside"):
        subject.edit_truth(original, edited, mask, 100 / 4096)
    mask[12, 11, 0] = 128
    with pytest.raises(subject.Rejected, match="nonbinary"):
        subject.edit_truth(original, original, mask, 100 / 4096)


def test_transitive_exposure_keeps_pair_failure_and_excludes_linked_candidates():
    base = {"fingerprints": {}, "pair_failure": None, "prior_matches": []}
    first = {**base, "ordinal": 0, "identity_keys": ["photo:1"]}
    bridge = {
        **base,
        "ordinal": 1,
        "identity_keys": ["photo:1", "author:a"],
        "pair_failure": "empty_exact_edit",
    }
    prior = {**base, "ordinal": 2, "identity_keys": ["author:a"], "prior_matches": ["prior_exact_rgb"]}
    independent = {**base, "ordinal": 3, "identity_keys": ["photo:4"]}
    assert subject.ordered_decisions([first, bridge, prior, independent]) == {
        0: "prior_exposure_component",
        1: "empty_exact_edit",
        2: "prior_exposure_component",
        3: "qualified",
    }


def test_complete_offline_readback_retains_rejected_primary_and_ordered_backup(tmp_path, monkeypatch):
    trial, cache = tmp_path / "trial", tmp_path / "media"
    trial.mkdir()
    monkeypatch.setattr(subject.roster, "PRIMARY", 1)
    monkeypatch.setattr(subject.roster, "BACKUP", 1)
    cases, payloads = [], {}
    for ordinal in (0, 1):
        source = "oi_" + str(ordinal + 1) * 16
        folder = source[3]
        image = np.random.default_rng(ordinal).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        edited = image.copy()
        y, x = (0, 0) if ordinal == 0 else (11, 11)
        edited[y, x, 0] ^= 1
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        blobs = dict(zip(subject.ASSETS, map(png, (image, edited, mask)), strict=True))
        paths = {
            "original": f"v2/images/{folder}/{source}_A0.png",
            "edited": f"v2/images/{folder}/{source}_F5.png",
            "semantic_mask": f"v2/masks/edit/{folder}/{source}_F5_edit.png",
        }
        payloads.update({paths[role]: data for role, data in blobs.items()})
        cases.append(
            {
                "ordinal": ordinal,
                "tier": "primary" if ordinal == 0 else "backup",
                "recipient": source,
                "publisher_split": "test_id",
                "size": [64, 64],
                "semantic_edit_area": 100 / 4096,
                "flickr_owner": f"author-{ordinal}",
                "flickr_photo_id": f"200{ordinal}",
                **{
                    role: {
                        "path": paths[role],
                        "sha256": subject.digest(data) if role != "semantic_mask" else None,
                    }
                    for role, data in blobs.items()
                },
            }
        )

    monkeypatch.setattr(subject, "lease", lambda *_args: (cache, cases, set(), ([], set(), set())))
    fetched = []

    def fake_download(_session, path, expected, _deadline):
        data = payloads[path]
        actual = subject.verify_asset_bytes(data, expected, None)
        fetched.append(path)
        return data, {
            "sha256": actual,
            "bytes": len(data),
            "publisher_asset_commitment": None,
            "resolver_revision": subject.roster.REVISION,
            "final_host": subject.HUB,
        }

    monkeypatch.setattr(subject, "download", fake_download)
    subject.acquire(trial, "authored plan")
    subject.seal(trial, "authored plan")
    summary = subject.read(trial / "summary.json")
    private = subject.read(trial / "qualified-private.json")
    assert len(fetched) == len(set(fetched)) == 6
    assert summary["qualified"] == summary["backup_attempted"] == 1
    assert summary["status_counts"] == {"exact_changes_outside_semantic_mask": 1, "qualified": 1}
    assert summary["accepted_mask_bindings"] == {"exact_acquired_bytes": 1}
    assert "oi_" not in str(summary) and "images/" not in str(summary)
    final = private["qualified"][0]
    assert final["source"]["ordinal"] == 1
    assert subject.sha(cache / final["truth_file"]) == final["truth_png_sha256"]
