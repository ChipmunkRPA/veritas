"""Bounded remote ZIP directories and fixed source/utterance planning, no inference."""

from __future__ import annotations

import argparse
import json
import subprocess
import types
import zipfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

from scripts import grid_acquisition_continuation_v135 as acquisition
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = acquisition.ROOT, acquisition.BASE, acquisition.PANEL
TRIAL = BASE / "grid-clip-plan-v136"
PUBLIC = PANEL / "grid-clip-plan-freeze-v136.json"
CODE = (
    "backend/scripts/grid_clip_plan_v136.py",
    "backend/tests/test_grid_clip_plan_v136.py",
    "docs/grid-clip-plan-v136-method.md",
    "backend/scripts/grid_media_preflight_v134.py",
)
read, write, sha, require = acquisition.read, acquisition.write, acquisition.sha, acquisition.require


class DirectoryReader:
    """Read-only seekable publisher byte ranges; never open an archive member."""

    def __init__(self, row):
        self.row, self.position = row, 0
        self.requests, self.network_bytes, self.cache = 0, 0, []

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        bases = {0: 0, 1: self.position, 2: self.row["bytes"]}
        require(whence in bases, "Known seek mode required")
        position = bases[whence] + offset
        if not 0 <= position <= self.row["bytes"]:
            raise OSError("Directory seek outside publisher object")
        self.position = position
        return position

    def read(self, amount=-1):
        remaining = self.row["bytes"] - self.position
        amount = remaining if amount in (None, -1) else min(amount, remaining)
        require(type(amount) is int and 0 <= amount <= 2 * 1024**2, "Bounded directory read required")
        if not amount:
            return b""
        start, end = self.position, self.position + amount
        for a, b, data in self.cache:
            if a <= start and end <= b:
                self.position = end
                return data[start - a : end - a]
        require(
            self.requests < 8 and self.network_bytes + amount <= 5 * 1024**2,
            "Bounded directory requests/bytes required",
        )
        request = Request(
            self.row["url"],
            headers={
                "User-Agent": "Chipmunk-Verify-transfer-evaluation",
                "Accept-Encoding": "identity",
                "Range": f"bytes={start}-{end - 1}",
            },
        )
        self.requests += 1
        with urlopen(request, timeout=30) as response:
            require(
                response.status == 206
                and response.headers.get("Content-Range") == f"bytes {start}-{end - 1}/{self.row['bytes']}"
                and response.headers.get("Content-Encoding", "identity") == "identity",
                "Exact unencoded publisher directory range required",
            )
            data = response.read(amount + 1)
        require(len(data) == amount, "Exact complete bounded directory bytes required")
        self.network_bytes += len(data)
        self.cache.append((start, end, data))
        self.position = end
        return data


def directory(row):
    reader = DirectoryReader(row)
    with zipfile.ZipFile(reader) as archive:
        entries = archive.infolist()
        require(
            len(entries) <= 5000 and len({r.filename for r in entries}) == len(entries),
            "Bounded unique ZIP directory required",
        )
        media = []
        for entry in entries:
            path = PurePosixPath(entry.filename)
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and "\\" not in entry.filename
                and "\x00" not in entry.filename,
                "Safe ZIP member path required",
            )
            if "__MACOSX" in path.parts or path.name.startswith("._"):
                continue
            if path.suffix.lower() == ".mpg":
                require(
                    not entry.flag_bits & 1
                    and entry.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    and 1000 <= entry.file_size <= 2 * 1024**2,
                    "Bounded real video member required",
                )
                media.append({"name": entry.filename, "bytes": entry.file_size, "crc32": entry.CRC})
    require(len(media) == 1000, "One thousand real media directory entries required")
    return media, reader.requests, reader.network_bytes


def completed(batch):
    folder = TRIAL / f"bind-{batch:02d}"
    if not all(p.is_file() for p in (folder / "output/result.json", folder / "profile.json")):
        return False
    result, profile = read(folder / "output/result.json"), read(folder / "profile.json")
    require(
        result["status"] == "complete"
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["worker_receipt_sha256"] == sha(folder / "output/result.json"),
        "Actual complete prior data stage required",
    )
    return True


def identity():
    acquisition.verify()
    return {
        "schema": "chipmunk.grid-clip-plan-freeze/136",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "acquisition_freeze_sha256": sha(acquisition.PUBLIC),
        "private_source_plan_sha256": sha(acquisition.TRIAL / "plan.json"),
        "source_archives": 16,
        "directory_batches": 2,
        "archives_per_directory_batch": 8,
        "selected_clips_per_archive": 2,
        "planned_clips": 32,
        "selection": "Original source order. For each source, v134 fixed salted SHA order of utterance stems, excluding globally used stems; no content, face, audio or detector response.",
        "scope": "Directory metadata only; full archive checksums, video/audio clocks, geometry, generation and detector transfer remain separate. All source/utterance identities and per-media CRC/fingerprints private. No raw media entry is opened.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print('{"status":"frozen","source_archives":16,"planned_clips":32}', flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Unchanged directory/clip planning procedure required",
    )


def metadata_batch(batch, output):
    selected = read(acquisition.TRIAL / "plan.json")["selected"]
    rows, requests, network_bytes = [], 0, 0
    with (output / "directories.jsonl").open("x") as journal:
        for record in selected[batch * 8 : (batch + 1) * 8]:
            media, calls, transferred = directory(record)
            row = {"source": record, "media": media, "full_archive_checksum_verified_by_this_stage": False}
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
            rows.append(row)
            requests += calls
            network_bytes += transferred
            print(
                json.dumps({"directory_completed_count": len(rows), "real_video_members": len(media)}),
                flush=True,
            )
    result = {
        "schema": "chipmunk.grid-directory-result/136",
        "status": "complete",
        "batch": batch,
        "source_archives": len(rows),
        "real_video_members_inventoried": sum(len(r["media"]) for r in rows),
        "range_requests": requests,
        "metadata_range_network_bytes": network_bytes,
        "private_directory_journal_sha256": sha(output / "directories.jsonl"),
        "freeze_sha256": sha(PUBLIC),
        "new_neural_calls": 0,
        "defaults_changed": False,
        "scope": "Publisher ZIP directory-range metadata only, not full-body checksum, media decoding, input qualification or detector performance. No archive member opened. All directory details remain private.",
    }
    write(output / "result.json", result)


def clip_plan(rows):
    require(
        len(rows) == 16 and [r["source"]["ordinal"] for r in rows] == list(range(16)),
        "All original ordered sources required",
    )
    used, selected = set(), []
    for row in rows:
        members = acquisition.preflight.select_members(row["media"], used)
        used.update(PurePosixPath(m["name"]).stem for m in members)
        for offset, member in enumerate(members):
            selected.append(
                {
                    "ordinal": len(selected),
                    "source_ordinal": row["source"]["ordinal"],
                    "within_source_ordinal": offset,
                    "member": member,
                    "source": row["source"],
                    "study_role": "external_transfer",
                    "media_qualified": False,
                }
            )
    require(len(selected) == len(used) == 32, "Thirty-two unique exact utterance stems required")
    return selected


def lock_clips(output):
    require(all(completed(b) for b in (0, 1)), "Both actual metadata batches required")
    rows = [
        json.loads(line)
        for b in (0, 1)
        for line in (TRIAL / f"bind-{b:02d}/output/directories.jsonl").read_text().splitlines()
    ]
    require(
        [r["source"] for r in rows] == read(acquisition.TRIAL / "plan.json")["selected"],
        "Exact original source plan required",
    )
    selected = clip_plan(rows)
    require(
        [r["member"] for r in selected[:2]] == read(acquisition.preflight.TRIAL / "plan.json")["clips"],
        "Previously fixed first-source choices must not change",
    )
    write(output / "selected-clips.json", selected)
    result = {
        "schema": "chipmunk.grid-selected-clips-result/136",
        "status": "complete",
        "source_archives": len(rows),
        "selected_clips": len(selected),
        "unique_utterance_stems": 32,
        "study_role": "external_transfer",
        "previous_first_source_choices_unchanged": True,
        "private_selected_clip_plan_sha256": sha(output / "selected-clips.json"),
        "private_directory_journal_sha256": [
            sha(TRIAL / f"bind-{b:02d}/output/directories.jsonl") for b in (0, 1)
        ],
        "freeze_sha256": sha(PUBLIC),
        "new_neural_calls": 0,
        "defaults_changed": False,
        "scope": "All original source/utterance choices fixed without media content or detector responses. Not full archive acquisition, independent people/ancestry proof, video/audio/face qualification or detector performance. Existing heads/calibrations remain fixed for external transfer.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


def routed_popen(argv, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"],
        "Only data-only metadata/plan stages permitted",
    )
    args = list(argv)
    args[3] = "scripts.grid_clip_plan_v136"
    return subprocess.Popen(args, **kwargs)


def run(batch):
    require(type(batch) is int and batch in (0, 1, 2), "One explicit metadata/plan stage required")
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        acquisition.core.runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda c, b: b == 0 or (completed(0) and (b == 1 or completed(1))),
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("bind", batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "bind"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "No extra freeze arguments")
        freeze()
    elif args.command == "run":
        require(len(args.arguments) == 1, "One fixed batch required")
        run(int(args.arguments[0]))
    else:
        verify()
        require(
            len(args.arguments) == 2 and args.arguments[1] in ("0", "1", "2"), "Fixed output/batch required"
        )
        batch = int(args.arguments[1])
        output = Path(args.arguments[0])
        require(
            output.resolve() == (TRIAL / f"bind-{batch:02d}/output").resolve(),
            "Exact guarded output required",
        )
        output.mkdir(exist_ok=False)
        (lock_clips(output) if batch == 2 else metadata_batch(batch, output))
