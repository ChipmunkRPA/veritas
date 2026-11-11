import time

import pytest
from requests.structures import CaseInsensitiveDict

from scripts import image_bg_real_transport_v225 as subject

PATH = "/xet-bridge-us/" + "a" * 24 + "/" + "b" * 64
URL = f"https://{subject.AWS}{PATH}?signature=private-test"
ASSET = "v2/masks/edit/a/test.png"


class Response:
    def __init__(self, status, headers, body=b""):
        self.status_code = status
        self.headers = CaseInsensitiveDict(headers)
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def iter_content(self, _size):
        yield self.body


class Session:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        assert kwargs["timeout"] == (5, 10) and kwargs["allow_redirects"] is False
        self.calls.append(url)
        return next(self.responses)


def headers(body, *, url=URL):
    return {
        "Location": url,
        "X-Repo-Commit": subject.original.roster.REVISION,
        "X-Linked-Etag": f'"{subject.original.digest(body)}"',
        "X-Linked-Size": str(len(body)),
    }


def test_exact_publisher_route_and_body_digest_for_all_asset_roles():
    body = b"authored PNG bytes"
    for asset in (ASSET, "v2/images/a/original.png", "v2/images/a/edited.png"):
        session = Session(
            Response(302, headers(body)), Response(200, {"Content-Length": str(len(body))}, body)
        )
        got, receipt = subject.download(session, asset, subject.original.digest(body), time.monotonic() + 2)
        assert got == body and receipt["final_host"] == subject.AWS
        assert session.calls == [subject.original.download_url(asset), URL]
        assert receipt["aws_transport"]["path"] == PATH
        assert receipt["aws_transport"]["linked_sha256"] == receipt["sha256"]
        assert "signature" not in str(receipt)


@pytest.mark.parametrize(
    "url",
    [
        f"http://{subject.AWS}{PATH}?signature=test",
        f"https://{subject.AWS}{PATH}",
        f"https://{subject.AWS}{PATH}/extra?signature=test",
        f"https://{subject.AWS}{PATH.replace('xet-bridge-us', 'xet-bridge-eu')}?signature=test",
        f"https://{subject.AWS}{PATH.replace('a' * 24, 'a' * 23)}?signature=test",
        f"https://{subject.AWS}{PATH.replace('b' * 64, '%2f' + 'b' * 64)}?signature=test",
        f"https://{subject.AWS}{PATH.upper()}?signature=test",
        f"https://{subject.AWS}:443{PATH}?signature=test",
        f"https://user@{subject.AWS}{PATH}?signature=test",
        f"https://{subject.AWS}.invalid{PATH}?signature=test",
    ],
)
def test_nearby_or_untrusted_paths_are_never_requested(url):
    session = Session(Response(302, headers(b"authored", url=url)))
    with pytest.raises(subject.original.Rejected, match="untrusted"):
        subject.download(session, ASSET, None, time.monotonic() + 2)
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    ("name", "value", "reason"),
    [
        ("X-Linked-Etag", None, "sha256"),
        ("X-Linked-Etag", "a" * 40, "sha256"),
        ("X-Linked-Size", None, "size"),
        ("X-Repo-Commit", "other", "revision"),
    ],
)
def test_resolver_commitment_is_required_before_aws_get(name, value, reason):
    fields = headers(b"authored")
    if value is None:
        del fields[name]
    else:
        fields[name] = value
    session = Session(Response(302, fields))
    with pytest.raises(subject.original.Rejected, match=reason):
        subject.download(session, ASSET, None, time.monotonic() + 2)
    assert len(session.calls) == 1


def test_aws_is_terminal_and_received_bytes_are_verified():
    body = b"authored"
    session = Session(Response(302, headers(body)), Response(302, {"Location": URL}))
    with pytest.raises(subject.original.Rejected, match="terminal"):
        subject.download(session, ASSET, None, time.monotonic() + 2)
    session = Session(Response(302, headers(body)), Response(200, {}, b"badbytes"))
    with pytest.raises(subject.original.Rejected, match="commitment_mismatch"):
        subject.download(session, ASSET, None, time.monotonic() + 2)


def test_offline_readback_checks_query_free_transport_proof(tmp_path):
    source = "oi_a" + "0" * 15
    asset = f"v2/masks/edit/a/{source}_F5_edit.png"
    body = b"authored"
    session = Session(Response(302, headers(body)), Response(200, {}, body))
    _, receipt = subject.download(session, asset, None, time.monotonic() + 2)
    folder = tmp_path / "00"
    folder.mkdir()
    (folder / "semantic_mask.png").write_bytes(body)
    case = {"recipient": source, "ordinal": 0, "semantic_mask": {"path": asset, "sha256": None}}
    record = {**receipt, "status": "saved", "repo_path": asset, "file": "00/semantic_mask.png"}
    assert subject.verify_file(tmp_path, case, "semantic_mask", record) == body
    record["aws_transport"]["path"] += "/extra"
    with pytest.raises(ValueError, match="transport proof"):
        subject.verify_file(tmp_path, case, "semantic_mask", record)


def test_other_v221_routes_still_use_original_transport():
    body = b"authored"
    target = "https://cas-bridge.xethub.hf.co/key"
    session = Session(Response(302, headers(body, url=target)), Response(200, {}, body))
    got, receipt = subject.download(session, ASSET, None, time.monotonic() + 2)
    assert got == body and receipt["final_host"] == "cas-bridge.xethub.hf.co"
    assert "aws_transport" not in receipt
