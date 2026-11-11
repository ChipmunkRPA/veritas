"""Resume the frozen v224 webcam diagnostic with read-only SQLite WAL headers."""

from __future__ import annotations

import argparse
import json
import sqlite3
from types import SimpleNamespace

from scripts import video_webcam_confound_v224 as prior
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = prior.ROOT, prior.BASE, prior.PANEL
TRIAL = BASE / "video-webcam-confound-resume-v227"
PUBLIC = PANEL / "video-webcam-confound-resume-freeze-v227.json"
FAILED = PANEL / "video-webcam-confound-unavailable-v224.json"
CODE = (
    "backend/scripts/video_webcam_confound_resume_v227.py",
    "backend/tests/test_video_webcam_confound_resume_v227.py",
    "docs/video-webcam-confound-resume-v227-method.md",
)
read, sha, require = prior.read, prior.sha, prior.require


def normalize_header(body):
    """Return a memory-only rollback-header copy; never touch the archived bytes."""
    require(
        isinstance(body, bytes) and len(body) >= 100 and body[:16] == b"SQLite format 3\x00",
        "Exact SQLite database header required",
    )
    require(body[18:20] in (b"\x01\x01", b"\x02\x02"), "Unsupported SQLite journal header")
    return body[:18] + b"\x01\x01" + body[20:]


class MemoryConnection:
    def __init__(self):
        self._connection = sqlite3.connect(":memory:")

    def deserialize(self, body):
        self._connection.deserialize(normalize_header(body))

    def execute(self, *args, **kwargs):
        return self._connection.execute(*args, **kwargs)

    def close(self):
        self._connection.close()


def memory_connect(target):
    require(target == ":memory:", "Only an in-memory connection is allowed")
    return MemoryConnection()


# select_record checks that archived WAL and journal sidecars are absent or empty
# before invoking connect/deserialize. The scoped override preserves that order.
def select_record(stream, condition, row):
    sqlite = SimpleNamespace(connect=memory_connect)
    return scoped(prior.select_record, {"sqlite3": sqlite})(stream, condition, row)


def identity():
    original = prior.identity()
    require(
        read(prior.PUBLIC) == read(prior.TRIAL / "freeze.json") == original,
        "Exact frozen v224 diagnostic required",
    )
    failed = read(FAILED)
    require(
        failed["schema"] == "chipmunk.video-webcam-confound-result/224"
        and failed["status"] == "complete"
        and failed["freeze_sha256"] == sha(prior.PUBLIC)
        and failed["source_groups"] == 1
        and failed["model_calls"] == 0
        and failed["serving_defaults_changed"] is False
        and set(failed["roles"]) == set(prior.prior.ROLES)
        and all(
            item == {"status": "unavailable", "reason": "source_or_archive_binding_failed"}
            for item in failed["roles"].values()
        ),
        "Exact failed v224 readout required",
    )
    return {
        "schema": "chipmunk.video-webcam-confound-resume-freeze/227",
        "v224_freeze_sha256": sha(prior.PUBLIC),
        "v224_failed_readout_sha256": sha(FAILED),
        "code_sha256": {path: sha(ROOT / path) for path in CODE},
        "database_header_bytes": [18, 19],
        "database_header_value": 1,
        "model_calls_permitted": False,
        "serving_change_permitted": False,
    }


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve prior attempts")
    frozen = identity()
    TRIAL.mkdir(mode=0o700)
    for path in (TRIAL / "freeze.json", PUBLIC):
        with path.open("x") as file:
            json.dump(frozen, file, sort_keys=True, indent=2, allow_nan=False)
            file.write("\n")
    return frozen


def readout():
    require(
        read(PUBLIC) == read(TRIAL / "freeze.json") == identity(), "Exact frozen v227 continuation required"
    )
    # Preserve v224's identity, archive verification, guarded frame checks,
    # measurements, unavailable statuses and read-only/no-network enforcement.
    result = scoped(prior.readout, {"select_record": select_record})()
    require(
        result["schema"] == "chipmunk.video-webcam-confound-result/224"
        and result["freeze_sha256"] == sha(prior.PUBLIC),
        "Unexpected original diagnostic result",
    )
    return {
        **result,
        "schema": "chipmunk.video-webcam-confound-resume-result/227",
        "freeze_sha256": sha(PUBLIC),
        "v224_freeze_sha256": sha(prior.PUBLIC),
        "v224_failed_readout_sha256": sha(FAILED),
        "scope": "One consumed matched webcam source; memory-only SQLite header normalization. "
        "Descriptive measurements are not edit or origin decisions.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "readout"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "readout": readout}[args.action](), indent=2, allow_nan=False))
