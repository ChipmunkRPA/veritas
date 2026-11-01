"""Acoustic assets obey the same frozen-set and case authority as voice clips."""
import hashlib

from sqlalchemy.orm import Session
from test_image_evidence_access import evidence_http  # noqa: F401
from test_multimedia_evidence_access import _add_audio

from app.evidence import service


def test_acoustic_charts_and_intervals_are_public_only_with_their_parent_case(evidence_http):  # noqa: F811
    f = evidence_http
    with Session(f.engine) as db:
        _, evidence = _add_audio(db, f.store, "quiet-acoustics", "visible-case", 0)
        db.commit()
    acoustic = evidence["acoustic_context"]
    assert evidence["version"] == "audio-evidence-2"
    assert acoustic["events"][0]["category"] == "near_silence"
    assert acoustic["events"][0]["interval"] == {"start_sample": 0, "end_sample": 64000}
    collection = f.client.get("/api/v2/analyses/quiet-acoustics/evidence-packets", headers=f.viewer)
    assert collection.status_code == 200 and collection.json() == evidence
    declared = service.asset_refs(evidence)
    assets = [*acoustic["overview"].values(), *acoustic["events"][0]["artifacts"].values()]
    for asset in assets:
        assert declared[asset["sha256"]] == asset
        url = asset["url"]
        assert f.client.get(url).status_code == 401
        assert f.client.get(url, headers=f.outsider).status_code == 404
        response = f.client.get(url, headers=f.viewer)
        assert response.status_code == 200 and hashlib.sha256(response.content).hexdigest() == asset["sha256"]
        assert f.client.get(url.replace(evidence["id"], "wrong-set"), headers=f.viewer).status_code == 404
        if asset["media_type"] == "audio/wav":
            partial = f.client.get(url, headers={**f.viewer, "Range": "bytes=0-63"})
            assert partial.status_code == 206 and partial.content == response.content[:64]
    # The input mapping is a commitment, not an automatic asset download grant.
    assert evidence["source"]["mapping_sha256"] not in declared
