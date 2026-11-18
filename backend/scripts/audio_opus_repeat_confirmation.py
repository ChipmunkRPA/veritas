# SPDX-License-Identifier: Apache-2.0
"""Source-new comparison of frozen sparse stereo repeat positions."""
from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
from scipy.signal import resample_poly

from app.detectors.audio.decode import _exact_zero_based_clock, decode_audio, decode_native_stereo
from app.detectors.audio.ogg_opus_reference import native_clock
from app.evidence import audio_aac_alignment, audio_opus_alignment
from scripts import audio_stereo_opus_repetition as opus
from scripts import audio_stereo_sparse_lag as sparse
from scripts.audio_harmonic_overlap_transfer import room_activity
from scripts.audio_native_alignment_panel import deliveries, read, save, save_wav, segment

R = 48000
SCHEMA = "chipmunk.dcase-opus-repeat-confirmation/1"
SCENES = ("busystreet", "office", "openairmarket", "park", "quietstreet", "supermarket")
SIZES = (3 * R, 7 * R // 2, 4 * R)
MEETINGS = {
    "ES2006d": "audio-stream-presence-ami-native-2026-11-11/ES2006d.wav",
    "ES2009d": "audio-stream-presence-ami-native-2026-11-11/ES2009d.wav",
    "ES2015d": "audio-stream-presence-ami-native-2026-11-11/ES2015d.wav",
    "IS1009a": "audio-ami-native-stereo-v291/raw/IS1009a.Array1-01.wav",
}
EXCLUDED_MEETINGS = ("ES2010a", "IS1005a", "ES2011a", "IS1004a", "ES2004a", "TS3004a", "ES2011b", "IS1006b")
DELAYS = (8, -11, 27, -21, 19, -32)
POSITIVES = ("repeat_10db", "repeat_18db")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def rank(value):
    return sha(f"{SCHEMA}/{value}".encode())


def file_sha(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def choose_foreground(annotations, source_root):
    result = {}
    with ZipFile(annotations) as archive:
        root = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        participants = {node.get("observation"): {p.get("global_name") for p in node if p.get("global_name")}
                        for node in root.iter() if node.get("observation")}
        excluded = set().union(*(participants[name] for name in EXCLUDED_MEETINGS))
        for room, relative in MEETINGS.items():
            if participants[room] & excluded:
                raise ValueError("The source groups share meeting participants")
            excluded.update(participants[room])
            path = source_root / relative
            occupancy = room_activity(archive, room)
            with wave.open(str(path)) as wav:
                available = min(wav.getnframes(), (path.stat().st_size - 1024)//2, len(occupancy)*160)
            positions = []
            for sec in range(303, available // 16000 - 5, 7):
                reach = occupancy[sec * 100:sec * 100 + 400]
                if (reach >= 1).sum() >= 310 and (reach >= 2).sum() <= 60:
                    positions.append(sec)
            active = sorted(positions, key=lambda sec:rank(f"{room}/{sec}"))[:6]
            if len(active) != 6:
                raise ValueError("Too few transcript-qualified distinct foregrounds")
            result[room] = {"sha256":file_sha(path), "sections":active, "path":relative}
    return result


def frame_case(media, begin, count):
    with wave.open(io.BytesIO(media)) as wav:
        if wav.getparams()[:4] != (2,2,44100,1323000):
            raise ValueError("Unexpected stereo PCM in DCASE")
        wav.setpos(begin * 44100)
        pcm = np.frombuffer(wav.readframes(count*147//160), dtype="<i2").reshape(-1,2)
    values = resample_poly(pcm.astype(float),160,147,axis=0)
    return np.rint(values).clip(-32768,32767).astype("<i2")


def subjects(zip_file):
    names = [name for name in zip_file.namelist() if name.endswith(".wav")]
    for family in SCENES:
        pool = sorted((name for name in names if Path(name).stem.rstrip("0123456789")==family), key=rank)
        if len(pool) != 10:
            raise ValueError("DCASE is incomplete")
        yield from pool[:4]


def implementation():
    modules = (sparse, audio_aac_alignment, audio_opus_alignment)
    files = (Path(__file__), *(Path(module.__file__) for module in modules))
    return {path.name: file_sha(path) for path in files}


def build(archive, annotations, sources):
    if file_sha(archive,"sha1") != sparse.ARCHIVE_SHA1:
        raise ValueError("Unexpected publisher archive")
    foreground = choose_foreground(annotations,sources)
    with ZipFile(archive) as zipped:
        groups = []
        for index, member in enumerate(subjects(zipped)):
            digest = sha(zipped.read(member))
            room = tuple(MEETINGS)[index % len(MEETINGS)]
            size = SIZES[index % len(SIZES)]
            groups.append({"id":Path(member).stem,"scene":Path(member).stem.rstrip("0123456789"),"member":member,
                           "source_sha256":digest, "background_start":3+int(rank(member)[0],16)%8,
                           "voice_room":room, "voice_start":foreground[room]["sections"][index//len(MEETINGS)],
                           "samples":size,"variant":(index//3 + index%3)%3, "delay":DELAYS[index%len(DELAYS)]})
    return {"schema":SCHEMA,"archive_sha1":sparse.ARCHIVE_SHA1,"annotations_sha256":file_sha(annotations),
            "code":implementation(),
            "foreground":foreground,"groups":groups,"scope":"160 kb/s CBR Ogg Opus with 20 ms packets, two channels",
            "roles":["recorded","ordinary_panned","different_background","moving_voice",*POSITIVES,
                     "repeated_foreground","same_channel_twice"]}


def case_sources(plan, archive, root):
    for source in plan["foreground"].values():
        if file_sha(root / source["path"]) != source["sha256"]:
            raise ValueError("Original foreground bytes changed")
    with ZipFile(archive) as zip_file:
        for group in plan["groups"]:
            raw = zip_file.read(group["member"])
            if sha(raw) != group["source_sha256"]:
                raise ValueError("Original background bytes changed")
            room = frame_case(raw, group["background_start"], group["samples"])
            neighbor = frame_case(raw, group["background_start"]+12, 30720)
            path = root / plan["foreground"][group["voice_room"]]["path"]
            voice = resample_poly(segment(path,group["voice_start"],group["samples"]//3,16000).astype(float),3,1)
            settings = (room, neighbor, voice, group["variant"])
            takes = list(deliveries(*settings,offset=group["delay"]))
            first, second = takes[-1][2]
            revoiced = voice.copy()
            revoiced[slice(*second)] = voice[slice(*first)]
            repeated = next(pcm for kind,pcm,_ in deliveries(room, neighbor, revoiced, group["variant"], offset=group["delay"])
                            if kind == "ordinary_panned")
            all_roles = (*takes,("repeated_foreground",repeated,None),("same_channel_twice",np.repeat(room[:,:1],2,axis=1),None))
            for role, pcm, truth in all_roles:
                if pcm.shape!=(group["samples"],2):
                    raise ValueError("Source does not cover its complete clock")
                yield group, role, pcm, truth


def decoded(path, n):
    args = ["ffmpeg","-nostdin","-v","info","-protocol_whitelist","file","-copyts","-i",str(path),
            "-map","0:a:0","-af","ashowinfo","-f","s16le","-c:a","pcm_s16le","pipe:1"]
    result = subprocess.run(args,check=True,capture_output=True,timeout=15)
    if len(result.stdout)!=4*n or not _exact_zero_based_clock(result.stderr.decode(),n):
        raise ValueError("Independently decoded frames do not match the exact source clock")
    return result.stdout


def save_fixture(fixtures, group, role, wav, ogg, spans, metadata):
    if fixtures is None or group["id"] != "office04" or role not in {"repeat_10db", "ordinary_panned"}:
        return
    case = "gain" if role == "repeat_10db" else "ordinary"
    delivery = fixtures / f"{case}.ogg"
    shutil.copyfile(ogg, delivery)
    metadata[case] = {"sha256": file_sha(delivery), "ranges": spans}
    if case == "gain":
        variable = fixtures / "variable.ogg"
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(wav), "-c:a", "libopus",
                        "-b:a", "160k", "-vbr", "on", "-frame_duration", "20", str(variable)],
                       check=True, capture_output=True, timeout=15)
        metadata["variable"] = {"sha256": file_sha(variable), "ranges": None}


def measure(plan_path, archive, root, output, fixtures):
    plan = read(plan_path)
    if plan["code"] != implementation() or file_sha(Path(audio_aac_alignment.__file__)) != opus.DETECTOR_SHA:
        raise ValueError("Frozen implementation changed")
    rows, metadata = [], {}
    if fixtures is not None:
        fixtures.mkdir(mode=0o700)
    with tempfile.TemporaryDirectory(prefix="opus-dcase-") as workspace:
        wav,ogg=(Path(workspace)/name for name in ("parent.wav","exact.ogg"))
        for group, role, pcm, truth in case_sources(plan,archive,root):
            save_wav(wav,pcm)
            opus.encode(wav,ogg)
            clock=native_clock(ogg,packet_bytes=400,channels=2,presented=group["samples"])
            body=decoded(ogg,group["samples"])
            baseline, extra = sparse.assess(np.frombuffer(body, dtype="<i2").reshape(-1, 2), fallback=True)
            file_digest = file_sha(ogg)
            mapping = decode_audio(ogg, source_sha256=file_digest).mapping
            native = decode_native_stereo(ogg, mapping, opus=True)
            if native != body:
                raise ValueError("The ordinary runtime does not recover the frozen PCM")
            fitted, reason, spans = audio_opus_alignment.search(np.frombuffer(native, dtype="<i2").reshape(-1, 2))
            expected = extra or baseline
            if ((fitted.right_offset_samples if fitted else None) != baseline["alignment"]
                    or (reason, spans) != (expected["status"], expected["spans"])):
                raise ValueError("The ordinary runtime differs from the frozen interval verifier")
            save_fixture(fixtures, group, role, wav, ogg, spans, metadata)
            rows.append({"group":group["id"],"scene":group["scene"],"role":role,"truth":truth,
                         "parent_sha256":file_sha(wav),"opus_sha256":file_sha(ogg),"pcm_sha256":sha(body),"clock":clock,
                         "baseline":baseline,"sparse":extra})
            if len(rows)%8==0:
                print("held",len(rows),192,group["id"],flush=True)
    body={"freeze_sha256":file_sha(plan_path),"rows":rows}
    save(output, body)
    if fixtures is not None:
        save(fixtures / "metadata.json", metadata)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("freeze","score"))
    for name in ("archive","annotations","root","freeze"):
        p.add_argument(name,type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--fixtures", type=Path)
    a=p.parse_args()
    if a.mode=="freeze":
        plan=build(a.archive,a.annotations,a.root)
        save(a.freeze,plan)
        print("Source groups",len(plan["groups"]),flush=True)
    else:
        if a.out is None:
            p.error("scoring requires --out")
        measure(a.freeze, a.archive, a.root, a.out, a.fixtures)


if __name__=="__main__":
    main()
