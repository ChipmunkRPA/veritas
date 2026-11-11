import base64
import hashlib

import httpx
import pytest

from scripts import video_vcd2_genuine_v197 as source

TAG = "0x8DEC57EEA21469F"
QUOTED = f'"{TAG}"'
URL = "https://vcdpublic.blob.core.windows.net/vcd2/fixture"


@pytest.mark.parametrize("tag", [TAG, QUOTED, "0xabcdef", '"0xABCDEF"'])
def test_azure_etag_accepts_hex_value_and_exact_quotes(tag):
    assert source.azure_etag(tag) == tag.strip('"')


@pytest.mark.parametrize(
    "tag",
    [
        None,
        "",
        '"arbitrary"',
        "*",
        '"*"',
        f"W/{QUOTED}",
        f"{TAG}, 0x2",
        f" {TAG}",
        f'"{TAG}',
        f'{TAG}"',
        "0x",
        "0xZZ",
        "0Xabc",
        "0xF\n",
    ],
)
def test_azure_etag_rejects_non_specific_or_malformed_conditions(tag):
    with pytest.raises(ValueError, match="Invalid Azure object ETag"):
        source.azure_etag(tag)


def transfer(monkeypatch, head_tag=TAG, get_tag=TAG, *, defect=None):
    body = b"Fixture body"
    body_md5 = base64.b64encode(hashlib.md5(body, usedforsecurity=False).digest()).decode("ascii")
    head = {"ETag": head_tag, "Content-Type": "video/x-matroska", "Content-Length": str(len(body))}
    get = {**head, "ETag": get_tag, "Content-MD5": body_md5}
    status = 200
    if defect == "status":
        status = 412
    elif defect == "partial":
        status = 206
        get["Content-Range"] = f"bytes 0-{len(body) - 1}/{len(body)}"
    elif defect == "range-header":
        get["Content-Range"] = f"bytes 0-{len(body) - 1}/{len(body)}"
    elif defect == "length":
        get["Content-Length"] = str(len(body) + 1)
    elif defect == "content-encoding":
        get["Content-Encoding"] = "gzip"
    elif defect == "md5":
        get["Content-MD5"] = "wrong"

    def handler(request):
        assert str(request.url) == URL and request.headers["Accept-Encoding"] == "identity"
        if request.method == "HEAD":
            return httpx.Response(200, headers=head)
        assert request.method == "GET" and "Range" not in request.headers
        assert request.headers["If-Match"] == QUOTED
        return httpx.Response(status, headers=get, stream=httpx.ByteStream(body))

    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(handler), **kw))
    return body


@pytest.mark.parametrize("head_tag,get_tag", [(TAG, TAG), (QUOTED, TAG), (TAG, QUOTED), (QUOTED, QUOTED)])
def test_transfer_compares_identical_objects_across_api_forms(tmp_path, monkeypatch, head_tag, get_tag):
    body = transfer(monkeypatch, head_tag, get_tag)
    output = tmp_path / "fixture.partial"
    receipt = source.download(URL, output)
    assert output.read_bytes() == body
    assert receipt["bytes"] == len(body) and receipt["sha256"] == hashlib.sha256(body).hexdigest()
    assert (receipt["etag"], receipt["head_etag"], receipt["get_etag"], receipt["if_match"]) == (
        TAG,
        head_tag,
        get_tag,
        QUOTED,
    )


@pytest.mark.parametrize(
    "get_tag,defect",
    [
        ("0xABCDEF", None),
        (TAG.lower(), None),
        ("*", None),
        (TAG, "status"),
        (TAG, "partial"),
        (TAG, "range-header"),
        (TAG, "length"),
        (TAG, "content-encoding"),
    ],
)
def test_transfer_rejects_unbound_body_before_writing(tmp_path, monkeypatch, get_tag, defect):
    transfer(monkeypatch, get_tag=get_tag, defect=defect)
    output = tmp_path / "fixture.partial"
    with pytest.raises((ValueError, httpx.HTTPStatusError)):
        source.download(URL, output)
    assert not output.exists()


def test_transfer_rejects_bad_official_md5(tmp_path, monkeypatch):
    transfer(monkeypatch, defect="md5")
    with pytest.raises(ValueError, match="Content-MD5"):
        source.download(URL, tmp_path / "fixture.partial")
