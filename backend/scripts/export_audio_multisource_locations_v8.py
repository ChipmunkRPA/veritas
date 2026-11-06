"""Supplement omitted upstream file locations without changing frozen inputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def checked_url(url):
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.hostname not in ("huggingface.co", "openslr.trmal.net")
    ):
        raise ValueError("Only public publisher locations without credentials/query are allowed")
    return url


def direct_asset(row):
    asset = row["asset"]
    if asset["sha256"] != row["sha256"] or asset["bytes"] != row["bytes"]:
        raise ValueError("Direct source asset differs from frozen original")
    return {
        "url": checked_url(asset["url"]),
        "sha256": asset["sha256"],
        "bytes": asset["bytes"],
        "kind": "direct-source",
    }


def export(root, capsule, output):
    body = json.loads(gzip.decompress(capsule.read_bytes()))
    prepared_path = root / "prepared-sources.json"
    prepared = json.loads(prepared_path.read_text())
    if body["prepared_sources_sha256"] != digest(prepared_path):
        raise ValueError("Public capsule and frozen originals differ")
    public = {r["sha256"]: r for r in body["sources"] + body["backgrounds"]}
    planning = root.parent / "audio-v7/fresh-planning"
    checksums = {
        line.split()[1]: line.split()[0]
        for line in (planning / "librispeech-md5sum.txt").read_text().splitlines()
    }
    tree = {
        Path(r["path"]).name: r
        for r in json.loads((planning / "duplex-pln-tree.json").read_text())
        if r["path"].endswith(".tar")
    }
    locations, archives = [], {}
    for row in prepared["sources"] + prepared["backgrounds"]:
        reference = public[row["sha256"]]
        if reference.get("url"):
            continue
        if row.get("asset"):
            asset = direct_asset(row)
        else:
            archive = Path(row["archive"])
            if str(archive) not in archives:
                name = archive.name
                if row["dataset"] == "OpenSLR12/LibriSpeech":
                    expected = checksums[name]
                    if digest(archive, "md5") != expected:
                        raise ValueError("Actual original LibriSpeech archive differs from publisher MD5")
                    asset = {
                        "url": checked_url("https://openslr.trmal.net/resources/12/" + name),
                        "archive_name": name,
                        "publisher_md5": expected,
                        "sha256": digest(archive),
                        "bytes": archive.stat().st_size,
                        "kind": "archive",
                        "publisher_md5_verified_locally": True,
                    }
                elif row["dataset"] == "DuplexGen/duplexgen-spoken":
                    record = tree[name]
                    expected = record["lfs"]["oid"]
                    if digest(archive) != expected:
                        raise ValueError("Actual original Chatterbox archive differs from publisher SHA256")
                    asset = {
                        "url": checked_url(
                            f"https://huggingface.co/datasets/{row['dataset']}/resolve/{row['revision']}/{record['path']}"
                        ),
                        "archive_name": name,
                        "sha256": expected,
                        "bytes": record["size"],
                        "kind": "archive",
                        "publisher_sha256_verified_locally": True,
                    }
                else:
                    raise ValueError("No justified publisher location for source without URL")
                archives[str(archive)] = asset
            asset = archives[str(archive)]
        locations.append({"source_sha256": row["sha256"], "publisher_file": asset})
    result = {
        "schema": "chipmunk.audio-v8-source-location-supplement/1",
        "capsule_sha256": digest(capsule),
        "prepared_sources_sha256": digest(prepared_path),
        "code_sha256": digest(Path(__file__)),
        "locations": locations,
        "source_selection_changed": False,
        "method_or_inputs_changed": False,
        "scope": "Adds missing direct/archived publisher-file locations for already frozen source-byte hashes only. Selected archive members remain identified by their name digest and original-byte digest in the capsule. No transcripts, provider IDs, local paths, model outputs or new sources.",
        "data_terms": body["data_terms"],
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "source_location_refs": len(locations),
                "verified_archives": len(archives),
                "supplement_sha256": digest(output),
                "source_selection_changed": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("capsule", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.root, args.capsule, args.output)
