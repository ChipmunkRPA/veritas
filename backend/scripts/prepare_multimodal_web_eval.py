"""Prepare a pinned, labelled external smoke set without querying any detector.

This is NOT an independent accuracy benchmark: detector training overlap is
unknown, publishers differ in encoding, and source samples are correlated.
Original image/video bytes are retained; text is the exact dataset body field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/independent-evaluation/multimodal"
SEED = "chipmunk-verify-external-smoke-2026-11-01-v1"
REPOS = {
    "image": ("Jouesmak/DiTFake", "ca9ea06c8f926c3a11ca4b657074cc7cbb99e5c7"),
    "video": ("faridlab/deepaction_v1", "476932c3369424ca59dc1f8fb758cb3c13fe63fe"),
    "document": ("liamdugan/raid", "865cac74188466cb0c3b7574a10204007b57a459"),
}
LIMITS = [
    "This labelled external smoke test is not a prevalence-weighted accuracy estimate or calibration set.",
    "Overlap with detector or backbone training data cannot be ruled out; RAID samples are from its labelled train split.",
    "Dataset labels are publisher assertions, not an independent forensic authentication of each real source.",
    "Original formats, resolution and compression differ between sources/classes; encoding shortcuts are possible.",
    "Only local evaluation is intended. Original source rights are retained; do not redistribute this sample collection.",
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ranked(values: list[str], key: str) -> list[str]:
    return sorted(values, key=lambda v: sha(f"{SEED}:{key}:{v}".encode()))


def save(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"Refusing to change frozen preparation output: {path}")
    path.write_bytes(data)


def get(url: str) -> bytes:
    data = bytearray()
    with requests.get(url, timeout=90, stream=True) as response:
        response.raise_for_status()
        for chunk in response.iter_content(1_048_576):
            if len(data) + len(chunk) > 150_000_000:
                raise ValueError(f"Source file exceeds bounded download size: {url}")
            data.extend(chunk)
    return bytes(data)


def source(kind: str) -> tuple[str, str, list[str]]:
    repo, revision = REPOS[kind]
    info_url = f"https://huggingface.co/api/datasets/{repo}/revision/{revision}"
    info = json.loads(get(info_url))
    assert info["sha"] == revision and not info.get("gated"), info_url
    card_url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/README.md"
    save(DEST / "sources" / f"{kind}-README.md", get(card_url))
    return repo, revision, [row["rfilename"] for row in info["siblings"]]


def media_item(kind: str, path: str, label: str, index: int, *, generator: str | None,
               source_group: str, license_name: str, label_evidence: str) -> dict:
    repo, revision = REPOS[kind]
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(path)}"
    item_id = f"{kind}-{label}-{index:02d}"
    local = DEST / "originals" / f"{item_id}{Path(path).suffix.lower()}"
    data = local.read_bytes() if local.exists() else get(url)
    save(local, data)
    dimensions = {}
    if kind == "image":
        with Image.open(local) as image:
            dimensions = {"width": image.width, "height": image.height, "image_format": image.format}
    if kind == "video":
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(local),
        ]))
        dimensions = {"duration_s": float(probe["format"]["duration"]),
                      "streams": [{k: row.get(k) for k in ["codec_type", "codec_name", "width", "height", "r_frame_rate"]}
                                  for row in probe["streams"]]}
    return {
        "id": item_id, "kind": kind, "label": label, "path": str(local.relative_to(ROOT)),
        "source_url": f"https://huggingface.co/datasets/{repo}/blob/{revision}/README.md",
        "download_url": url, "repository": repo, "revision": revision, "source_path": path,
        "source_group": source_group, "generator": generator, "license": license_name,
        "evaluation_only": True, "source_label_evidence": label_evidence,
        "provenance_note": "Publisher-labelled external sample; training overlap unknown. Downloaded bytes are unaltered.",
        "sha256": sha(data), "bytes": len(data), "media": dimensions,
    }


def images() -> list[dict]:
    _, _, paths = source("image")
    output = []
    real_paths = [p for p in paths if "/FLUX.1-schnell/0_real/" in p and p.endswith(".jpg")]
    for index, path in enumerate(ranked(real_paths, "image-real")[:8], 1):
        output.append(media_item("image", path, "real", index, generator=None,
            source_group=f"COCO:{Path(path).stem}", license_name="Dataset card Apache-2.0; underlying COCO image rights remain separate",
            label_evidence="Official DiTFake test/0_real directory; README identifies the real images as COCO photographs."))
    index = 0
    for generator, count in [("FLUX.1-schnell", 3), ("PixArt-Sigma-XL-2-1024-MS", 3), ("stable-diffusion-3-medium-diffusers", 2)]:
        candidates = [p for p in paths if f"/{generator}/1_fake/" in p and p.endswith(".png")]
        for path in ranked(candidates, f"image-{generator}")[:count]:
            index += 1
            output.append(media_item("image", path, "ai", index, generator=generator,
                source_group=f"DiTFake:{generator}:{Path(path).stem}", license_name="Dataset card Apache-2.0",
                label_evidence=f"Official DiTFake test/{generator}/1_fake directory; README states these are generated images."))
    return output


def videos() -> list[dict]:
    repo, revision, paths = source("video")
    captions = get(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/captions.csv")
    save(DEST / "sources/video-captions.csv", captions)
    generators = ["CogVideoX5B", "RunwayML", "Veo", "VideoPoet"]
    groups = [str(i) for i in range(100) if all(f"{g}/{i}/a.mp4" in paths for g in ["Pexels", *generators])]
    selected = ranked(groups, "video-actions")[:4]
    output = []
    for index, (group, generator) in enumerate(zip(selected, generators, strict=True), 1):
        for label, folder in [("real", "Pexels"), ("ai", generator)]:
            item = media_item("video", f"{folder}/{group}/a.mp4", label, index,
                generator=generator if label == "ai" else None, source_group=f"DeepAction:action-{group}",
                license_name="CC-BY-4.0" if label == "ai" else "Pexels License",
                label_evidence=f"Official DeepAction {folder} directory. README identifies Pexels as real and the other six named generator directories as synthetic; action {group} is shared by this real/AI pair.")
            item["action_id"] = int(group)
            output.append(item)
    return output


def documents() -> list[dict]:
    """Read bounded row groups through HTTP ranges, preserving publisher labels.

    RAID's test labels are hidden. Use explicitly labelled train rows for this
    smoke check only, with six human source articles paired to six generated
    continuations. The held-out audio calibration does not use these records.
    """
    import fsspec
    import pyarrow.parquet as pq

    source("document")
    repo, original_revision = REPOS["document"]
    conversion_revision = "5008b2d167f74bfd4ceff6b3b0512b7097d7cbb1"
    parquet_path = "raid/partial-train/0006.parquet"
    url = f"https://huggingface.co/datasets/{repo}/resolve/{conversion_revision}/{parquet_path}"
    pool_groups = list(range(9, 41))
    metadata = []
    with fsspec.open(url, "rb", block_size=65536) as remote:
        parquet = pq.ParquetFile(remote)
        for group in pool_groups:
            rows = parquet.read_row_group(group, columns=["id", "source_id", "model", "domain", "attack"]).to_pylist()
            for offset, row in enumerate(rows):
                row.update(row_group=group, row_group_offset=offset, row_index=1_480_000 + group * 1000 + offset)
                metadata.append(row)
        clean = [r for r in metadata if r["domain"] == "news" and r["attack"] == "none"]
        generators = ["mistral", "mistral-chat", "llama-chat"]
        groups = {r["source_id"] for r in clean if r["model"] == "human"}
        for generator in generators:
            groups &= {r["source_id"] for r in clean if r["model"] == generator}
        selected_groups = ranked(list(groups), "document-news-sources")[:6]
        wanted = []
        for index, group in enumerate(selected_groups, 1):
            generator = generators[(index - 1) % len(generators)]
            for label, model in [("real", "human"), ("ai", generator)]:
                candidates = [r for r in clean if r["source_id"] == group and r["model"] == model]
                row_id = ranked([r["id"] for r in candidates], f"document-{group}-{model}")[0]
                chosen = next(r for r in candidates if r["id"] == row_id)
                wanted.append({**chosen, "label": label, "index": index})
        row_groups = {group: parquet.read_row_group(group).to_pylist() for group in {r["row_group"] for r in wanted}}
    output = []
    for selected in wanted:
        row = row_groups[selected["row_group"]][selected["row_group_offset"]]
        assert row["id"] == selected["id"] and row["attack"] == "none" and row["domain"] == "news"
        item_id = f"document-{selected['label']}-{selected['index']:02d}"
        local = DEST / "originals" / f"{item_id}.txt"
        data = row["generation"].encode("utf-8")
        save(local, data)
        save(DEST / "sources" / f"{item_id}-row.json", (json.dumps(row, ensure_ascii=False, indent=2) + "\n").encode())
        output.append({
            "id": item_id, "kind": "document", "label": selected["label"], "path": str(local.relative_to(ROOT)),
            "source_url": f"https://huggingface.co/datasets/{repo}/blob/{original_revision}/README.md",
            "download_url": url, "repository": repo, "revision": original_revision,
            "conversion_revision": conversion_revision, "source_path": f"train.csv row id {row['id']}",
            "row_index": selected["row_index"], "row_id": row["id"], "source_group": f"RAID:news:{row['source_id']}",
            "generator": row["model"] if selected["label"] == "ai" else None,
            "license": "Dataset MIT declaration; underlying source-article rights retained",
            "evaluation_only": True,
            "source_label_evidence": f"Author dataset row model={row['model']}, attack=none, domain=news; original row retained in sources/.",
            "provenance_note": "Labelled RAID train split, not hidden-label test split. Training overlap unknown. Only exact generation body submitted, with no label, title or prompt prefix. Each real/AI pair shares a source article.",
            "sha256": sha(data), "bytes": len(data), "characters": len(row["generation"]),
        })
    return output


def challenges() -> list[dict]:
    """Two fixed publisher demos, separate from the paired main smoke set."""
    revision = "60783e704160023913bee78f0b47036d393d4dfa"
    sources = [
        {
            "generator": "HunyuanVideo-1.5",
            "source_url": f"https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/{revision}/README.md",
            "snapshot_url": f"https://raw.githubusercontent.com/Tencent-Hunyuan/HunyuanVideo-1.5/{revision}/README.md",
            "download_url": "https://github.com/user-attachments/assets/fdc3c27b-69f5-46a1-b707-0b57510fa32f?download=1",
            "published_media_url": "https://github.com/user-attachments/assets/fdc3c27b-69f5-46a1-b707-0b57510fa32f",
            "source_group": "HunyuanVideo-1.5:official-instruction-following-demo-1",
            "source_label_evidence": "Official README generation showcase, Strong Instruction Following row, first video: a sad dark-haired Chinese woman looks toward the sky. Input and rewritten generation prompts are supplied beside this output.",
            "revision": revision,
            "snapshot_suffix": ".md",
        },
        {
            "generator": "LTX-2.5 (official product demo; exact checkpoint variant undisclosed)",
            "source_url": "https://ltx.io/model/ltx-2",
            "snapshot_url": "https://ltx.io/model/ltx-2",
            "download_url": "https://videos.ltx.io/LTX-2/LTX-2-5/4cards/02-autoDuration.mp4",
            "published_media_url": "https://videos.ltx.io/LTX-2/LTX-2-5/4cards/02-autoDuration.mp4",
            "source_group": "LTX-2.5:official-auto-duration-demo",
            "source_label_evidence": "Official model page, LTX-2.5 Auto Duration generation capability card; poster alt describes a man diving off rocks into the sea. The linked MP4 is that card's generated example, not a depth/pose control input.",
            "revision": None,
            "snapshot_suffix": ".html",
        },
    ]
    output = []
    for index, entry in enumerate(sources, 1):
        item_id = f"challenge-video-ai-{index:02d}"
        snapshot = DEST / "sources" / f"{item_id}-source{entry['snapshot_suffix']}"
        snapshot_data = snapshot.read_bytes() if snapshot.exists() else get(entry["snapshot_url"])
        save(snapshot, snapshot_data)
        assert entry["published_media_url"].encode() in snapshot_data
        local = DEST / "originals" / f"{item_id}.mp4"
        data = local.read_bytes() if local.exists() else get(entry["download_url"])
        save(local, data)
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(local),
        ]))
        output.append({
            **{key: value for key, value in entry.items() if key not in {"snapshot_url", "snapshot_suffix"}},
            "id": item_id, "kind": "video", "label": "ai", "path": str(local.relative_to(ROOT)),
            "source_snapshot": str(snapshot.relative_to(ROOT)), "source_snapshot_sha256": sha(snapshot_data),
            "sha256": sha(data), "bytes": len(data), "evaluation_only": True,
            "license": "Public author demonstration, local evaluation only; no redistribution permission established. Model/code license is not treated as a license to demo footage.",
            "provenance_note": "Publisher-labelled showcase output, not a representative random sample. Exact original published bytes, no trimming or transcoding. Does not establish that all audio, titles, or editing were generated by the named model. Detector training overlap unknown.",
            "media": {"duration_s": float(probe["format"]["duration"]),
                      "streams": [{k: row.get(k) for k in ["codec_type", "codec_name", "width", "height", "r_frame_rate"]}
                                  for row in probe["streams"]]},
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["image", "video", "document", "challenge"], required=True)
    args = parser.parse_args()
    items = {"image": images, "video": videos, "document": documents, "challenge": challenges}[args.kind]()
    manifest = {"schema": "chipmunk-external-smoke/v1", "selection_seed": SEED,
                "selection_policy": "SHA256 rank of fixed source paths/action groups; fixed class/generator quotas; no detector result was consulted.",
                "limitations": LIMITS, "items": items}
    if args.kind == "challenge":
        manifest["selection_policy"] = "Two fixed official generation demos chosen before any inference: first HunyuanVideo-1.5 instruction-following example and LTX-2.5 Auto Duration card. Separate unpaired challenge set; never merge into the frozen 36-sample paired smoke summary."
        manifest["limitations"] = LIMITS + [
            "Publisher showcase selection bias; two generated clips and no matched real controls. No accuracy or calibration claim.",
            "These public demo assets are authorized local analysis inputs, not licensed production assets or redistributable training data.",
            "Exact checkpoint, postprocessing, and generated-versus-added audio composition are not fully documented.",
        ]
    save(DEST / f"{args.kind}-manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps({"kind": args.kind, "items": len(items), "bytes": sum(i["bytes"] for i in items),
                      "manifest": str(DEST / f"{args.kind}-manifest.json")}))


if __name__ == "__main__":
    main()
