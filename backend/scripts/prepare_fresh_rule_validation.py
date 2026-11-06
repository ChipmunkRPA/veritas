"""Prepare source-excluded tests for the frozen customer 80% rule."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from prepare_video_upgrade_eval import download, sha

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "data/independent-evaluation/rule80-v1"
SEED = "chipmunk-rule80-confirmation-20261105-v1"
IMAGE_REV = "ca9ea06c8f926c3a11ca4b657074cc7cbb99e5c7"
DFADD_REV = "dfc1eeab3cb0068db8e87a2b89a1ebd103665b1f"
VIDEO_REV = "476932c3369424ca59dc1f8fb758cb3c13fe63fe"
RTFS_REV = "bca865e675852666a426368b2f5614abf1c1926a"
AUDIO_FAMILIES = ("GradTTS", "MatchaTTS", "PflowTTS", "StyleTTS2")


def order(value):
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen preparation record changed: {path}")
        return
    with path.open("x") as out:
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)


def get(url):
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    return response


def walk(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def exclusions():
    paths = sorted(set(
        list((ROOT / "data/independent-evaluation").rglob("*manifest*.json"))
        + list((ROOT / "data/localization-eval").glob("*/manifest.json"))
        + list((ROOT / "data/text-upgrade").glob("*manifest*.json"))
    ))
    prior, docs, coco, used_hashes = [], [], set(), set()
    for path in paths:
        if TRIAL in path.parents:
            continue
        document = json.loads(path.read_text())
        prior.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path)})
        for row in walk(document):
            used_hashes.update(v for k, v in row.items() if "sha256" in k and isinstance(v, str))
            if isinstance(row.get("text"), str) and row.get("domain") in {"article", "mixed", "email"}:
                docs.append(row)
            if row.get("source_id") and row.get("family") in {"firefly", "sd2", "flux-fill"}:
                coco.add(str(row["source_id"]))
            path_name = row.get("source_path", "")
            if "/0_real/" in path_name:
                coco.add(Path(path_name).stem.lstrip("0") or "0")
    save(TRIAL / "exclusions.json", {"manifests": prior, "sha256": sorted(used_hashes), "coco": sorted(coco)})
    save(TRIAL / "previous-articles.json", {"cases": docs})
    return paths, used_hashes, coco


def acquire_file(item):
    target = TRIAL / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        download(item["download_url"], target)
        return {**item, "status": "ready", "sha256": sha(target), "bytes": target.stat().st_size}
    except Exception as error:
        return {**item, "status": "unavailable", "reason": f"{type(error).__name__}: {error}"}


def finish(kind, items):
    used = set(json.loads((TRIAL / "exclusions.json").read_text())["sha256"])
    seen = set()
    for row in items:
        if row.get("sha256") in used | seen:
            raise ValueError(f"Repeated previously scored source bytes: {row['id']}")
        if row.get("status") == "ready":
            seen.add(row["sha256"])
    save(TRIAL / f"{kind}-manifest.json", {
        "seed": SEED, "kind": kind, "plan_sha256": sha(TRIAL / f"{kind}-plan.json"), "items": items,
    })
    print(kind, Counter((row["label"], row["status"]) for row in items), flush=True)


def images(coco):
    repo = "Jouesmak/DiTFake"
    info = get(f"https://huggingface.co/api/datasets/{repo}/revision/{IMAGE_REV}").json()
    assert info["sha"] == IMAGE_REV
    paths = [r["rfilename"] for r in info["siblings"]]
    rows = []
    real = sorted([p for p in paths if "/FLUX.1-schnell/0_real/" in p and p.endswith(".jpg")
                   and (Path(p).stem.lstrip("0") or "0") not in coco], key=order)[:12]
    for i, path in enumerate(real):
        rows.append({"id": f"dit-real-{i:02}", "kind": "image", "label": 0, "family": "camera",
                     "group": f"COCO:{Path(path).stem}", "source_path": path})
    for family in ("FLUX.1-schnell", "PixArt-Sigma-XL-2-1024-MS", "stable-diffusion-3-medium-diffusers"):
        candidates = sorted([p for p in paths if f"/{family}/1_fake/" in p and p.endswith(".png")], key=order)
        for path in candidates[:4]:
            rows.append({"id": f"dit-ai-{len(rows):02}", "kind": "image", "label": 1, "family": family,
                         "group": f"DiTFake:{family}:{Path(path).stem}", "source_path": path})
    for row in rows:
        path = row["source_path"]
        row.update(path=f"image-native/{row['id']}{Path(path).suffix}",
                   source_url=f"https://huggingface.co/datasets/{repo}/blob/{IMAGE_REV}/README.md",
                   download_url=f"https://huggingface.co/datasets/{repo}/resolve/{IMAGE_REV}/{quote(path)}",
                   source_terms="Publisher Apache-2.0; underlying COCO photographs retain source rights")
    save(TRIAL / "image-plan.json", {"seed": SEED, "items": rows, "splice_pairs": 12,
                                    "native_source_bytes": True})
    with ThreadPoolExecutor(4) as pool:
        acquired = list(pool.map(acquire_file, rows))
    save(TRIAL / "image-native-acquired.json", acquired)
    # The separately fixed TGIF source plan adds twelve authentic/edit pairs.
    import prepare_tgif_localization as tgif
    sub = TRIAL / "tgif"
    sub.mkdir(exist_ok=True)
    tgif.plan(sub, 4, coco | {Path(p).stem.lstrip("0") or "0" for p in real})
    tgif.acquire(sub)
    tg = json.loads((sub / "manifest.json").read_text())["items"]
    for row in tg:
        acquired.append({"id": "tgif-" + row["id"], "kind": "image", "label": row["label"],
                         "family": row["family"] if row["label"] else "camera",
                         "group": f"TGIF:COCO:{row['source_id']}", "status": "ready",
                         "path": "tgif/" + row["image"], "sha256": row["image_sha256"],
                         "source_url": "https://github.com/IDLabMedia/tgif-dataset",
                         "source_terms": "CC BY-SA 4.0 dataset; upstream COCO photograph rights separate"})
    finish("image", acquired)


def videos(paths):
    from prepare_face_swap_upgrade_eval import plan as face_plan
    used_actions, old_faces = set(), []
    for path in paths:
        if TRIAL in path.parents:
            continue
        data = json.loads(path.read_text())
        for row in walk(data):
            group = str(row.get("group") or row.get("source_group") or "")
            if group.startswith("DeepAction:action-"):
                used_actions.add(int(group.rsplit("-", 1)[1]))
            if group.startswith("RTFS:") and "donor_filename" in row:
                old_faces.append(row)
    used_actions |= {29, 40, 54, 85}
    # Native clips retain their original duration, size, encoding and provenance.
    actions = sorted(set(range(1, 101)) - used_actions, key=lambda x: order(str(x)))[:12]
    families = ("CogVideoX5B", "RunwayML", "Veo", "BDAnimateDiffLightning", "StableDiffusion", "VideoPoet")
    rows = []
    for i, action in enumerate(actions):
        for label, family in ((0, "Pexels"), (1, families[i % len(families)])):
            source = f"{family}/{action}/a.mp4"
            ident = f"deepaction-{action}-{family}"
            rows.append({"id": ident, "kind": "video", "label": label, "family": family,
                         "group": f"DeepAction:action-{action}", "source_path": source,
                         "path": f"video-native/{ident}.mp4",
                         "source_url": "https://huggingface.co/datasets/faridlab/deepaction_v1",
                         "download_url": f"https://huggingface.co/datasets/faridlab/deepaction_v1/resolve/{VIDEO_REV}/{source}",
                         "source_terms": "CC BY 4.0 generated clips; Pexels source license for camera clips"})
    metadata = ROOT / "data/independent-evaluation/video-v5/sources/metadata"
    faces = face_plan(metadata, 12, tuple(old_faces))
    for i in range(12):
        group = faces[3*i:3*i+3]
        for row in (group[0], group[1 + i % 2]):
            ident = row["id"]
            rows.append({"id": ident, "kind": "video", "label": row["label"], "family": row["family"],
                         "group": row["group"], "donor_filename": row["donor_filename"],
                         "path": f"video-native/{ident}.mp4", "download_url": row["download_url"],
                         "source_url": row["source_url"], "source_terms": row["license"]})
    save(TRIAL / "video-plan.json", {"seed": SEED, "excluded_actions": sorted(used_actions), "items": rows})
    with ThreadPoolExecutor(4) as pool:
        acquired = list(pool.map(acquire_file, rows))
    finish("video", acquired)


def articles():
    from prepare_article_confirmation import freeze
    from prepare_text_upgrade import article_pairs, digest, normalized_hash
    prior = json.loads((TRIAL / "previous-articles.json").read_text())["cases"]
    groups = {row["group"] for row in prior if row.get("group")}
    texts = {normalized_hash(row["text"]) for row in prior}
    pool = article_pairs(ROOT / "data/text-upgrade/sources/llmtrace-test.jsonl")
    chosen, counts = [], Counter()
    for topic in sorted(pool, key=order):
        if digest(f"article:{topic}") in groups:
            continue
        available = [r for r in pool[topic] if normalized_hash(r[3]) not in texts]
        humans = [r for r in available if r[1] == "human"]
        machines = [r for r in available if r[1] == "ai"]
        if not humans or not machines:
            continue
        ai = min(machines, key=lambda r: (counts[r[2]], order(r[3])))
        human = min(humans, key=lambda r: (abs(len(r[3].split())-len(ai[3].split())), order(r[3])))
        if abs(len(human[3].split())-len(ai[3].split())) > 80:
            continue
        counts[ai[2]] += 1
        for r in (human, ai):
            chosen.append({"id": f"llmtrace:{r[0]}", "text": r[3], "label": int(r[1] == "ai"),
                           "generator": r[2], "group": digest(f"article:{topic}"),
                           "source_url": "https://huggingface.co/datasets/iitolstykh/LLMTrace_classification"})
            texts.add(normalized_hash(r[3]))
        if len(chosen) == 24:
            break
    assert len(chosen) == 24
    confirmation = freeze(ROOT / "data/text-upgrade/sources/news.parquet", TRIAL / "previous-articles.json", 2)
    assert len(confirmation) == 24
    chosen.extend({**r, "label": int(r["label"] == "ai"),
                   "source_url": "https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Text_Dataset"} for r in confirmation)
    plan = []
    for row in chosen:
        content = row.pop("text").encode()
        ident = order(row["id"])[:16]
        path = TRIAL / f"articles/{ident}.txt"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        plan.append({**row, "id": f"article-{ident}", "kind": "document", "family": row["generator"],
                     "path": str(path.relative_to(TRIAL)), "sha256": sha(path), "status": "ready",
                     "text_sha256": hashlib.sha256(content).hexdigest(),
                     "source_terms": "Publisher dataset declarations retained in the source study"})
    save(TRIAL / "document-plan.json", {"seed": SEED, "items": plan})
    finish("document", plan)


def audio(paths):
    import fsspec
    import soundfile as sf
    from evaluate_audio_candidates import transcript_key
    old = [row for path in paths if TRIAL not in path.parents
           for row in json.loads(path.read_text()).get("base_files", [])]
    used_sources = {row.get("source_group") for row in old}
    used_prompts = {row["prompt_index"] for row in old if row.get("prompt_index") is not None}
    used_texts = {row.get("transcript_sha256") for row in old}
    used_speakers = {row.get("speaker") for row in old}
    meta = TRIAL / "audio-metadata"
    meta.mkdir(exist_ok=True)
    handles, indices = {}, {}
    for family in (*AUDIO_FAMILIES, "VCTK_BONAFIDE"):
        url = f"https://huggingface.co/datasets/isjwdu/DFADD/resolve/{DFADD_REV}/DATASET_{family}.zip"
        stream = fsspec.open(url, "rb", block_size=65536).open()
        archive = zipfile.ZipFile(stream)
        handles[family] = (stream, archive, url)
        indices[family] = archive.namelist()
        save(meta / (family + ".json"), indices[family])
    vctk = json.loads((ROOT / "data/independent-evaluation/audio-v2/metadata/vctk-index.json").read_text())
    speakers = set.intersection(*[
        {m[1] for name in indices[family] if (m := re.search(r"/(p\d+|s\d+)_(\d+)_", name))}
        for family in AUDIO_FAMILIES])
    selected_voices = sorted(speakers, key=order)[:24]
    assert len(selected_voices) == 24
    plan = []
    for i, speaker in enumerate(selected_voices):
        family = AUDIO_FAMILIES[i % 4]
        candidates = [name for name in indices[family] if f"/{speaker}_" in name
                      and name.endswith((".wav", ".flac")) and
                      int(re.search(r"/(?:p\d+|s\d+)_(\d+)_", name)[1]) not in used_prompts]
        fake = min(candidates, key=order)
        used_prompts.add(int(re.search(r"/(?:p\d+|s\d+)_(\d+)_", fake)[1]))
        authentic = [r for r in vctk if r["speaker_id"] == speaker and r["mic_id"] == "mic1"
                     and "vctk:"+r["id"].rsplit("_mic", 1)[0] not in used_sources
                     and transcript_key(r["text"]) not in used_texts]
        genuine = min(authentic, key=lambda r: order(r["id"]))
        rid = genuine["id"].rsplit("_mic",1)[0]
        real = next(name for name in indices["VCTK_BONAFIDE"] if Path(name).stem == rid)
        for label, fam, member in ((0, "VCTK_BONAFIDE", real), (1, family, fake)):
            plan.append({"id": f"audio-{i:02}-{label}", "kind": "audio", "label": label, "family": fam,
                         "group": "vctk:" + speaker, "member": member, "speaker_seen_previously": "vctk:"+speaker in used_speakers,
                         "source_group": "vctk:"+rid if label == 0 else f"dfadd:{fam}:{Path(member).stem}",
                         "source_url": "https://huggingface.co/datasets/isjwdu/DFADD",
                         "source_terms": "MIT compilation; VCTK CC BY 4.0; LJ Speech public-domain text"})
    # Background works were downloaded for source inspection but never selected in a prior speech evaluation.
    used_bg = {r.get("source_id") for name in ("audio", "audio-v2", "audio-v3")
               for r in json.loads((ROOT / f"data/independent-evaluation/{name}/manifest.json").read_text()).get("backgrounds", [])}
    bg = {"noise": [], "music": []}
    for file in (ROOT / "data/independent-evaluation/audio-v3/metadata").glob("parquet-audio-group-*.json"):
        for row in json.loads(file.read_text()).get("rows", []):
            if (row.get("source") not in {"free-sound", "rfm"} or row.get("path") in used_bg
                    or row["ref"]["decoded_samples"] < 64000):
                continue
            kind = "noise" if row["source"] == "free-sound" else "music"
            bg[kind].append(row)
    bg = {kind: sorted(rows, key=lambda r: order(r["path"]))[:4 if kind == "noise" else 2] for kind, rows in bg.items()}
    assert len(bg["noise"]) == 4 and len(bg["music"]) == 2
    save(TRIAL / "audio-plan.json", {"seed": SEED, "items": plan, "backgrounds": bg,
                                    "conditions": {"clean": None, "noise_0": 0, "music_5": 5},
                                    "voice_overlap": "Fresh recordings and unused synthetic prompt indices. Speaker identities overlap earlier studies."})
    acquired = []
    for i, row in enumerate(plan):
        _, archive, url = handles[row["family"]]
        raw = archive.read(row["member"])
        path = TRIAL / f"audio-native/{order(row['id'])[:16]}.audio"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
        # Use the same production decoder before making the declared source prefix.
        from app.detectors.audio.decode import decode_audio
        decoded = decode_audio(path, source_sha256=hashlib.sha256(raw).hexdigest())
        pcm = decoded.pcm[:64000]
        assert len(pcm) > 0
        for condition, snr in (("clean", None), ("noise_0", 0), ("music_5", 5)):
            output = pcm.copy()
            background = None
            if snr is not None:
                kind = condition.split("_")[0]
                # Each paired real/AI voice group gets the same work and offset.
                background = bg[kind][(i//2) % len(bg[kind])]
                bgpath = ROOT / "data/independent-evaluation/audio-v3" / background["ref"]["path"]
                noise = decode_audio(bgpath, source_sha256=background["ref"]["sha256"]).pcm
                assert len(noise) >= len(pcm)
                noise = noise[:len(pcm)]
                scale = np.sqrt(np.mean(pcm**2)) / max(np.sqrt(np.mean(noise**2)), 1e-8) * 10**(-snr/20)
                output = output + scale*noise
            output = output / max(1., float(np.abs(output).max()) / .98)
            target = TRIAL / f"audio-inputs/{order(row['id']+condition)[:16]}.wav"
            target.parent.mkdir(exist_ok=True)
            sf.write(target, output, 16000, subtype="PCM_16")
            acquired.append({**row, "id": row["id"]+"-"+condition, "condition": condition,
                             "path": str(target.relative_to(TRIAL)), "sha256": sha(target), "status": "ready",
                             "source_sha256": hashlib.sha256(raw).hexdigest(), "source_archive": url,
                             "seconds": len(output)/16000, "background_work": background["path"] if background else None,
                             "background_sha256": background["ref"]["sha256"] if background else None, "snr_db": snr})
        print(row["id"], len(pcm)/16000, "prepared", flush=True)
    for stream, archive, _ in handles.values():
        archive.close()
        stream.close()
    finish("audio", acquired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("exclusions", "image", "video", "document", "audio"))
    args = parser.parse_args()
    TRIAL.mkdir(parents=True, exist_ok=True)
    if args.kind == "exclusions":
        exclusions()
        return
    excluded = json.loads((TRIAL/"exclusions.json").read_text())
    paths = [ROOT / r["path"] for r in excluded["manifests"]]
    if args.kind == "image":
        images(set(excluded["coco"]))
    elif args.kind == "video":
        videos(paths)
    elif args.kind == "document":
        articles()
    else:
        audio(paths)


if __name__ == "__main__":
    main()
