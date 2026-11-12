# SPDX-License-Identifier: Apache-2.0
"""Lock a fresh source-only palette-matched challenge for the paired cut guard."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import cv2
import networkx as nx
import numpy as np

from scripts import video_native_background_panel as earlier

common, require, old = earlier.common, earlier.require, earlier.old
ARCHIVE = earlier.ARCHIVE
PRIOR_PANEL = earlier.previous.prior.DATA / "video-native-background-panel-2026-11-12"
N = earlier.N
PAIR_COUNT = 6
PROBES = (0, 55, 64, 73, 82, 96, 118)
SPANS = ((59, 69), (65, 77), (71, 81))
LONG = ((50, 68), (56, 78))
ORDINARY = (65, 77)
CANDIDATE_SHA256 = "ab7908fbaafe35a5e879190ada426401d7b248e8d37069db733b330a5df2eca6"
SPEC = {
    "schema": "chipmunk.video-native-background-challenge/1",
    "sources": "Use only native archived real target and publisher swap groups outside all 65 known previous short-review/scene study source URLs and donor keys. Preserve source rate, original delivery identity, and native-frame hashes privately. Known linkage excludes aliases documented in the publisher metadata; general project exposure and unknown identity linkage remain possible.",
    "matching": "At seven fixed genuine sample times, average the masked outer grayscale distance and normalized HSV palette intersection distance. Weight them .35 and .65. Compute maximum-cardinality minimum-weight disjoint matching on all eligible sources and keep its six least-distant pairs. No face model, incumbent alert, guard feature, or label enters pair selection.",
    "deliveries": "Each direction of every pair supplies exact whole-picture cuts at all three fixed spans, both direct and with an ordinary constant donor color grade. Other views include an original, one publisher swap, and one lighting or camera-punch control per source. Exact 640x360/25 120-picture H.264 yuv420p MP4, CRF 23, silent, with earlier-nearest native source pictures and no interpolation.",
    "color_grade": "For each source compute per-channel RGB mean and standard deviation over the outer 40 percent of the frame, pooled over seven fixed genuine source pictures. For a graded cut apply target/source standard-deviation ratio clipped to .80–1.20 and a mean-matching offset clipped to ±24 RGB levels. Apply it uniformly to the entire authentic donor frame, with rounding and clipping. No compositing or generated pixels are used; the arithmetic color grade is an ordinary video edit.",
    "preview_ticks": list(PROBES), "cut_spans": list(map(list,SPANS)), "long_swap_spans": list(map(list,LONG)),
    "ordinary_span": list(ORDINARY), "candidate_sha256": CANDIDATE_SHA256,
    "conditional_challenge": "Finish and seal all genuine cut and collateral incumbent outputs before the fixed candidate sees any of these deliveries. Use every selected real coarse range. All genuine cuts without one and all unaffected controls remain in the ledger. Do not estimate false-alarm rate from the alert-conditioned subset.",
    "gate": {"minimum_cut_removals":2,"minimum_independent_pairs":2,"minimum_useful_swap_coarse":1,
             "maximum_useful_swap_losses":0,"maximum_false_matched_on_controls":0,"exact_native_edge_tolerance_pictures":1},
}


def pool(archive):
    before, paths = earlier.eligible(archive)
    panel = common.read(PRIOR_PANEL / "freeze.json")
    require(len(panel["source_roster"]) == 16, "Previous exact source roster required")
    metadata = old.trial.prior.comparison.metadata()

    def keys(index):
        receipt = common.read(common.DATA / f"group-{index:03d}-complete.json")
        return metadata[receipt["group"]]

    prior = [keys(row["index"]) for row in panel["source_roster"]]
    links, donors = {row["url"] for row in prior}, set.union(*(row["donors"] for row in prior))
    fresh = []
    for item in before:
        meta = keys(item["index"])
        if meta["url"] in links or meta["donors"] & donors:
            continue
        fresh.append(item)
        links.add(meta["url"])
        donors.update(meta["donors"])
    return fresh, [*paths, PRIOR_PANEL / "freeze.json"]


def preview_stats(frames):
    gray = earlier.previous.background(frames)
    histograms, outer = [], []
    for frame in frames.values():
        image = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0,1,2], None, [8,4,4], [0,180,0,256,0,256]).ravel().astype(np.float64)
        histograms.append(hist / hist.sum())
        outer.append(np.concatenate((image[:, :13].reshape(-1,3),image[:, -13:].reshape(-1,3))))
    pixels = np.concatenate(outer).astype(np.float64)
    return {"gray": gray, "histograms": np.stack(histograms), "mean":pixels.mean(axis=0), "std":pixels.std(axis=0)}


def compare(a,b):
    gray = float(np.mean(np.abs(a["gray"]-b["gray"])))
    palette = float(1-np.minimum(a["histograms"],b["histograms"]).sum(axis=1).mean())
    return .35*gray + .65*palette, gray, palette


def coefficients(a,b):
    ratio = np.clip(a["std"]/np.maximum(b["std"],1.),.8,1.2)
    offset = np.clip(a["mean"]-ratio*b["mean"],-24.,24.)
    return {"scale":ratio.tolist(),"offset":offset.tolist()}


def freeze(folder, archive):
    fresh, prior = pool(archive)
    require(len(fresh) >= 2*PAIR_COUNT, "Too few source-disjoint fresh candidates")
    stats = {}
    for pin in fresh:
        index = pin["index"]
        stats[index] = preview_stats(earlier.source_pictures(archive,pin,"original",PROBES,verify_file=False))
        print("genuine palette sample",len(stats),"of",len(fresh),flush=True)
    edges = []
    for n, a in enumerate(stats):
        for b in list(stats)[n+1:]:
            score,gray,palette = compare(stats[a],stats[b])
            edges.append((score,gray,palette,a,b))
    graph = nx.Graph()
    graph.add_nodes_from(stats)
    for tie,(score,gray,palette,a,b) in enumerate(sorted(edges)):
        graph.add_edge(a,b,weight=round(score*10**9)*10000+tie,measure=(score,gray,palette))
    matches = sorted((*graph[a][b]["measure"],*sorted((a,b))) for a,b in nx.min_weight_matching(graph))
    matches = matches[:PAIR_COUNT]
    require(len(matches)==PAIR_COUNT,"Six source-disjoint genuine pairs are required")
    sources = {i for *_,a,b in matches for i in (a,b)}
    pairs = [{"score":score,"gray":gray,"palette":palette,"groups":[a,b],
              "grades":[coefficients(stats[a],stats[b]),coefficients(stats[b],stats[a])]} for score,gray,palette,a,b in matches]
    folder.mkdir(parents=True,mode=0o700)
    common.write(folder/"freeze.json",{"spec":SPEC,"available_source_groups":len(fresh),"excluded_prior_groups":65,
          "roster":[pin for pin in fresh if pin["index"] in sources],"pairs":pairs,
          "eligible_pool_sha256":hashlib.sha256(repr(fresh).encode()).hexdigest(),
          "prior_freezes":[common.digest(path) for path in prior],"worker_sha256":common.digest(Path(__file__))})
    print("frozen",len(fresh),"eligible and",len(pairs),"paired",flush=True)


def frozen(folder, archive):
    lock = common.read(folder/"freeze.json")
    fresh, prior = pool(archive)
    require(lock["spec"]==SPEC and lock["worker_sha256"]==common.digest(Path(__file__))
            and lock["eligible_pool_sha256"]==hashlib.sha256(repr(fresh).encode()).hexdigest()
            and lock["prior_freezes"]==[common.digest(path) for path in prior],"Source or method freeze differs")
    return lock


def recipe(direction):
    cases = [(f"{style}_{i}",span,None) for i,span in enumerate(SPANS) for style in ("direct","graded")]
    cases += [("long_swap",LONG[direction],("swap_a","swap_b")[direction]),
              ("camera_punch" if direction==0 else "light",ORDINARY,None),("original",None,None)]
    return cases


def grade(rgb, settings):
    scale = np.asarray(settings["scale"],np.float32)
    offset = np.asarray(settings["offset"],np.float32)
    return np.clip(np.rint(rgb.astype(np.float32)*scale+offset),0,255).astype(np.uint8)


def encode(path, host, guest, condition, span, altered, color):
    if condition.startswith(("direct_","graded_")):
        a,b = span
        insert = {i:(guest[i] if condition.startswith("direct_") else grade(guest[i],color)) for i in range(a,b)}
        require(all(not np.array_equal(host[i],rgb) for i,rgb in insert.items()),"Whole-picture cut must change each authored frame")
        old.encoded.encode(path,frames=(insert.get(i,rgb) for i,rgb in host.items()),crf=23)
    else:
        earlier.encode_case(path,host,condition,span,altered,guest)


def prepare(folder,archive):
    lock = frozen(folder,archive)
    out = folder/"media"
    out.mkdir(mode=0o700)
    roster = {pin["index"]:pin for pin in lock["roster"]}
    rows = []
    for pair,match in enumerate(lock["pairs"]):
        a,b = match["groups"]
        base = {i:earlier.source_pictures(archive,roster[i],"original",range(N)) for i in (a,b)}
        for direction,(host,guest) in enumerate(((a,b),(b,a))):
            for condition,span,swap in recipe(direction):
                altered = earlier.source_pictures(archive,roster[host],swap,range(*span)) if swap else None
                path = out/f"pair-{pair:02d}-{direction}-{condition}.mp4"
                encode(path,base[host],base[guest],condition,span,altered,match["grades"][direction])
                row = old.describe(path,2*pair+direction,"held",condition,span if swap else None)
                rows.append({**row,"source_pair":pair,"direction":direction,"interval":span,
                             "genuine_cut":condition.startswith(("direct_","graded_"))})
            print("authored challenge source",pair,direction,flush=True)
    require(len(rows)==len({row["sha256"] for row in rows})==108,"Complete distinct challenge deliveries required")
    common.write(folder/"media.json",{"freeze_sha256":common.digest(folder/"freeze.json"),"media":rows,
                  "conditions":dict(Counter(row["condition"] for row in rows))})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("freeze","prepare"))
    parser.add_argument("folder",type=Path)
    parser.add_argument("--archive",type=Path,default=ARCHIVE)
    args=parser.parse_args()
    cv2.setNumThreads(1)
    (freeze if args.phase=="freeze" else prepare)(args.folder,args.archive)


if __name__=="__main__":
    main()
