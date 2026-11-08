"""Acquire pinned public research assets without package installation or source execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

LOCK = Path(__file__).with_name("public_audioseal_assets_v2.json")
RESERVE = 1536 * 1024**2
HOSTS = {"codeload.github.com", "huggingface.co", "files.pythonhosted.org", "raw.githubusercontent.com"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def relative(value):
    path = PurePosixPath(value)
    require(
        isinstance(value, str)
        and value
        and not path.is_absolute()
        and "\\" not in value
        and all(p not in (".", "..") for p in value.split("/")),
        "Canonical relative member required",
    )
    return path


def checked(path, row):
    require(path.is_file() and not path.is_symlink(), "A regular asset file is required")
    require(
        path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
        "Pinned file identity mismatch: " + row["path"],
    )


def url_allowed(url, *, redirect=False):
    parsed = urllib.parse.urlsplit(url)
    require(
        parsed.scheme == "https"
        and parsed.username is None
        and parsed.password is None
        and not parsed.fragment,
        "Public HTTPS asset URL required",
    )
    host = parsed.hostname or ""
    require(
        host in HOSTS or (redirect and (host.endswith(".hf.co") or host.endswith(".huggingface.co"))),
        "Unexpected asset host",
    )


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        url_allowed(newurl, redirect=True)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(row, destination):
    url_allowed(row["source"])
    request = urllib.request.Request(
        row["source"], headers={"User-Agent": "Chipmunk-Verify-public-research/2"}
    )
    received = 0
    with (
        urllib.request.build_opener(PublicRedirect()).open(request, timeout=60) as response,
        destination.open("xb") as target,
    ):
        require(response.status == 200, "Complete public asset response required")
        while chunk := response.read(min(1024**2, row["bytes"] + 1 - received)):
            received += len(chunk)
            require(received <= row["bytes"], "Asset exceeds pinned byte count")
            target.write(chunk)
    checked(destination, row)


def extract(archive, rows, target):
    """Read only explicitly reviewed regular members; never invoke extractall/setup."""
    members = [str(relative(r["member"])) for r in rows]
    destinations = [str(relative(r["path"])) for r in rows]
    require(
        len(set(members)) == len(members) and len({s.casefold() for s in destinations}) == len(destinations),
        "Unique archive and destination members required",
    )
    if archive.name.endswith(".tar.gz"):
        handle = tarfile.open(archive, "r:gz")
        names = [m.name for m in handle.getmembers()]

        def read(row):
            member = handle.getmember(row["member"])
            require(member.isfile() and member.size == row["bytes"], "Pinned regular tar member required")
            with handle.extractfile(member) as stream:
                return stream.read(row["bytes"] + 1)
    else:
        handle = zipfile.ZipFile(archive)
        names = handle.namelist()

        def read(row):
            member = handle.getinfo(row["member"])
            require(
                not member.is_dir()
                and not stat.S_ISLNK(member.external_attr >> 16)
                and member.file_size == row["bytes"],
                "Pinned regular zip member required",
            )
            with handle.open(member) as stream:
                return stream.read(row["bytes"] + 1)

    with handle:
        require(
            all(names.count(member) == 1 for member in members),
            "Each reviewed member must occur exactly once",
        )
        for row in rows:
            data = read(row)
            require(
                len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"],
                "Reviewed archive member identity mismatch",
            )
            path = target / row["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify(root, lock):
    for row in lock["assets"]:
        checked(root / row["path"], row)
    for key, directory in [("source_files", "reviewed-source"), ("dependency_files", "isolated-python")]:
        for row in lock[key]:
            checked(root / directory / row["path"], row)
    require(
        b"license: mit" in (root / "publisher-model-README.md").read_bytes(),
        "Pinned publisher MIT model card required",
    )
    return {
        "assets": len(lock["assets"]),
        "source_files": len(lock["source_files"]),
        "dependency_files": len(lock["dependency_files"]),
        "package_installations": 0,
        "source_imports": 0,
        "checkpoint_loads": 0,
        "neural_forwards": 0,
    }


def prepare(root, caches, allow_download):
    lock = json.loads(LOCK.read_text())
    require(lock["schema"] == "chipmunk.public-audioseal-assets/v2", "Pinned asset lock required")
    # Cache contents are copied, never hard-linked or edited: original studies stay independent.
    required = sum(r["bytes"] for r in lock["assets"]) + sum(
        r["bytes"] for r in lock["source_files"] + lock["dependency_files"]
    )
    require(
        shutil.disk_usage(root.parent).free >= RESERVE + required,
        "Insufficient reserved disk for isolated acquisition",
    )
    root.mkdir(exist_ok=False)
    downloads = []
    for row in lock["assets"]:
        relative(row["path"])
        destination = root / row["path"]
        cached = next((p / row["path"] for p in caches if (p / row["path"]).exists()), None)
        if cached is not None:
            checked(cached, row)
            shutil.copyfile(cached, destination)
            checked(destination, row)
            mode = "verified_cache_copy"
        else:
            require(
                allow_download, "Missing public asset in cache; pass --download for public HTTPS acquisition"
            )
            download(row, destination)
            mode = "public_https_download"
        downloads.append({**row, "acquisition": mode})
    for key, directory in [("source_files", "reviewed-source"), ("dependency_files", "isolated-python")]:
        archives = sorted({r["archive"] for r in lock[key]})
        for name in archives:
            extract(root / name, [r for r in lock[key] if r["archive"] == name], root / directory)
    source = {
        "code_revision": lock["code_revision"],
        "model_revision": lock["model_revision"],
        "files": [{k: r[k] for k in ("path", "bytes", "sha256")} for r in lock["source_files"]],
        "executed_source": False,
    }
    dependencies = {"files": [{k: r[k] for k in ("path", "sha256")} for r in lock["dependency_files"]]}
    write(root / "source-review.json", source)
    write(root / "isolated-python-files.json", dependencies)
    write(root / "downloads.json", {"files": downloads, "package_installations": 0, "checkpoint_loads": 0})
    result = {
        "schema": "chipmunk.public-audioseal-acquisition/2",
        "lock_sha256": sha(LOCK),
        **verify(root, lock),
        "public_http_downloads": sum(r["acquisition"] == "public_https_download" for r in downloads),
        "customer_inputs": 0,
        "serving_environment_modified": False,
    }
    write(root / "acquisition.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--cache", action="append", type=Path, default=[])
    parser.add_argument("--download", action="store_true", help="Allow only pinned public asset downloads")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.root, json.loads(LOCK.read_text()))
            if args.verify_only
            else prepare(args.root, args.cache, args.download),
            indent=2,
        )
    )
