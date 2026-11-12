# SPDX-License-Identifier: Apache-2.0
"""Compare two speech enhancers on separately sourced loud backgrounds."""

from __future__ import annotations

import argparse
import importlib.metadata
import tarfile
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

from scripts import audio_noise_views_transfer as prior
from scripts import audio_partial_publisher_transfer as publisher

common, require, save = prior.common, prior.require, prior.save
SCHEMA = "chipmunk.nemotron-noise-transfer/1"
ASSETS = {
    "model.safetensors": "06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1",
    "config.json": "e69986483774f01893d2727503addc49ef201d19e6e1a266b8d4e971e28c478d",
    "LICENSE": "a3d882953f026d3e2fdd57842d9c522c8320d4c0388af2278bcf4d438d255a2c",
}
REVISION = "31f82f2efad1095034492f236e8dac9239ba713b"
ARCHIVE = "a6d76436a371643920d1fc00fcc640d98964bd16e5075bab24d46694f77cc3d7"
BACKGROUNDS = {
    "restaurant.ogg": ("0aeb1fa7a3d9647d3284e97d9880b6375b54db2eda6a517c64e3bb02ac8a51cc", (6, 18, 36, 54)),
    "vienna.wav": ("a4b1807f396592351d7ffb17873391cec9b9144f9216af3d463b1f2009e90e36", (2, 8, 14, 20)),
}
RUNTIME = {"mlx": "0.32.4", "mlx-metal": "0.32.4", "mlx-audio": "0.6.5", "onnxruntime": "1.31.1"}
GATE = {"min_noisy_synthetic_groups": 4, "min_families": 3, "min_scenes": 2,
        "max_noisy_genuine_groups": 2, "max_clean_or_isolated_flags": 0}
PANEL = common.PUBLIC / "audio-partial-publisher-transfer-2026-11-12.json"
HEADS = ("old_enhancer", "new_enhancer")


def held_sources(archive_path, labels_path):
    require(archive_path.stat().st_size == 192 * 1024**2 and common.sha(archive_path) == ARCHIVE
            and common.sha(labels_path) == publisher.PINS['part-a.labels.txt'], "Pinned benchmark data differs")
    excluded = {row['source']['speaker'] for row in common.read(PANEL)['roster'] if row['source']}
    labels = {name: values for name, *values in map(str.split, labels_path.read_text().splitlines())}
    candidates, seen = defaultdict(list), set()
    try:
        with tarfile.open(archive_path, 'r|') as archive:
            for member in archive:
                if not member.isfile():
                    continue
                match = publisher.MEMBER.fullmatch(member.name)
                require(match is not None and 44 < member.size < 2 * 1024**2, "Unexpected tar member")
                name, group = match.groups()
                require(name in labels and name not in seen, "Publisher member has no unique label")
                with archive.extractfile(member) as entry:
                    data = entry.read()
                seen.add(name)
                duration, label, *segments = labels[name]
                if group in excluded or len(segments) != 1 or not 3 <= Decimal(duration) <= 14:
                    continue
                raw = publisher.native(data)
                samples = len(raw) // 2
                clock = publisher.labels_for(labels[name], samples)
                if samples < 48_000 or clock != [[0, samples, label]]:
                    continue
                family = None if label == "bonafide" else name.split('-')[1]
                candidates[family].append(({"id": name, "label": label, "group": group, "family": family,
                                            "wav_sha256": common.digest(data), "samples": samples}, raw))
    except (tarfile.ReadError, EOFError) as error:
        require(isinstance(error, EOFError) or 'end of data' in str(error), "Unexpected prefix read error")
    require(len(seen) == 992 and set(candidates) == {None, *publisher.GENERATORS}, "Publisher roster differs")
    used = set()
    for family in sorted(candidates, key=lambda k: (len({r['group'] for r, _ in candidates[k]}), k or '')):
        count = 0
        for row, raw in sorted(candidates[family], key=lambda item: common.digest(f"{SCHEMA}/{item[0]['id']}".encode())):
            if row['group'] in used:
                continue
            count += 1
            used.add(row['group'])
            yield row, raw
            if count == (10 if family is None else 2):
                break
        require(count == (10 if family is None else 2), "Not enough separate source groups")


def prepare(run, inputs, previous, labels):
    selected = list(held_sources(inputs / "part-a-192m.prefix.tar", labels))
    scenes = {}
    for name, (pin, _) in BACKGROUNDS.items():
        require(common.sha(inputs / name) == pin, "Noise source differs")
        scenes[name] = prior.enhancement.float_pcm(common.decode(inputs / name))
    old_roster, old_freeze = (common.read(previous / name) for name in ('roster.json', 'freeze.json'))
    require(common.sha(previous / 'roster.json') == old_freeze['roster_sha256'], "Prior calibration roster differs")
    run.mkdir(mode=0o700)
    (run / 'pcm').mkdir()
    rows = []

    def keep(raw, role, scene, label, group, family, origin):
        require(len(raw) == 96_000, "Every comparison uses exactly three seconds")
        row = {"id": len(rows), "role": role, "scene": scene, "label": label, "group": group, "family": family,
               "origin": origin, "pcm_sha256": common.digest(raw)}
        (run / 'pcm' / f"{row['id']:03}.pcm").write_bytes(raw)
        rows.append(row)

    for row in old_roster:
        if row['role'] == 'calibration':
            raw = (previous / 'pcm' / f"{row['id']:03}.pcm").read_bytes()
            require(common.digest(raw) == row['pcm_sha256'], "Prior calibration PCM differs")
            keep(raw, 'selection', row['scene'], row['label'], row['group'], None, row['id'])
    role_rank = Counter()
    for row, raw in selected:
        start = (row['samples'] - 48_000) // 2
        rank = role_rank[row['label']]
        role_rank[row['label']] += 1
        offsets = {name: starts[rank % 4] * 16_000 for name, (_, starts) in BACKGROUNDS.items()}
        noise = {name: scenes[name][offset:offset + 48_000] for name, offset in offsets.items()}
        matched = prior.match_views(prior.enhancement.float_pcm(raw[2 * start:2 * (start + 48_000)]), noise, -5)
        for scene, values in matched.items():
            keep(prior.enhancement.pcm(values), 'held', scene, row['label'], row['group'], row['family'],
                 {**row, 'center_start_sample': start, 'background_start_sample': offsets.get(scene)})
    for scene, (_, offsets) in BACKGROUNDS.items():
        for i, seconds in enumerate(offsets):
            offset = seconds * 16_000
            keep(prior.enhancement.pcm(scenes[scene][offset:offset + 48_000]), 'held', scene, 'isolated',
                 f'{scene}-{i}', None, {'background_start_sample': offset})
    require(Counter(row['role'] for row in rows) == {'selection': 56, 'held': 74}, "Input panel has a different size")
    save(run / 'roster.json', rows)
    save(run / 'freeze.json', {"schema": SCHEMA, "runner_sha256": common.sha(Path(__file__)),
         "reviewer_sha256": common.sha(Path(__file__).with_name('review_audio_nemotron_noise_transfer.py')),
         "roster_sha256": common.sha(run / 'roster.json'), "model": {"revision": REVISION, "files": ASSETS},
         "runtime": RUNTIME, "voice_assets": common.assets(), "prior_enhancer_sha256": prior.enhancement.MODEL_SHA,
         "prior_enhancer_license_sha256": prior.enhancement.LICENSE_SHA, "publisher_archive_sha256": ARCHIVE,
         "previous_selection_freeze_sha256": common.sha(previous / 'freeze.json'), "excluded_panel_sha256": common.sha(PANEL),
         "backgrounds": BACKGROUNDS, "decision_rule": GATE, "panel_authenticity_forwards_before_freeze": 0})
    print(f"Frozen 56 selection views and 74 held views. {common.sha(run / 'freeze.json')}", flush=True)


def frozen(run):
    lock = common.read(run / 'freeze.json')
    require(lock['runner_sha256'] == common.sha(Path(__file__)) and lock['roster_sha256'] == common.sha(run / 'roster.json'),
            "Frozen run differs")
    return lock, common.read(run / 'roster.json')


def enhance(run, model, old_model):
    import mlx.core as mx
    import onnxruntime as ort
    from mlx_audio.sts.utils import load_model

    _, rows = frozen(run)
    require({package: importlib.metadata.version(package) for package in RUNTIME} == RUNTIME, "Runtime differs")
    require({name: common.sha(model / name) for name in ASSETS} == ASSETS, "Publisher model files differ")
    graph = old_model / prior.enhancement.MODEL
    require(common.sha(graph) == prior.enhancement.MODEL_SHA
            and common.sha(old_model / 'LICENSE') == prior.enhancement.LICENSE_SHA, "Prior enhancement model differs")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    old = ort.InferenceSession(str(graph), sess_options=options, providers=['CPUExecutionProvider'])
    new = load_model(str(model))
    mx.set_cache_limit(256 * 1024**2)
    output = run / 'enhanced'
    output.mkdir(mode=0o700)
    receipts, repeat = [], None
    for row in rows:
        raw = (run / 'pcm' / f"{row['id']:03}.pcm").read_bytes()
        require(common.digest(raw) == row['pcm_sha256'], "Frozen mix changed")
        wave = prior.enhancement.float_pcm(raw)
        values = np.asarray(new.generate(wave, sample_rate=16_000, chunk_seconds=None).audio, dtype=np.float32)
        require(values.shape == wave.shape and np.isfinite(values).all(), "Enhancement changed the native clock")
        if repeat is None:
            again = np.asarray(new.generate(wave, sample_rate=16_000, chunk_seconds=None).audio, dtype=np.float32)
            repeat = float(np.max(np.abs(values - again)))
            require(repeat <= 1e-5, "Local enhancement repeat exceeds tolerance")
        signals = [prior.enhanced_view(wave, old), prior.canonical(values)]
        measured = {}
        for name, signal in zip(HEADS, signals, strict=True):
            encoded = prior.enhancement.pcm(signal)
            (output / f"{row['id']:03}-{name}.pcm").write_bytes(encoded)
            measured[name] = {"pcm_sha256": common.digest(encoded),
                              "suppression_db": prior.enhancement.db_ratio(prior.enhancement.energy(wave), prior.enhancement.energy(signal))}
        receipts.append({"id": row['id'], 'enhancers': measured})
        if len(receipts) % 20 == 0 or len(receipts) == len(rows):
            print(f"{len(receipts)}/{len(rows)} views enhanced", flush=True)
    save(run / 'enhancements.json', {"freeze_sha256": common.sha(run / 'freeze.json'), "repeat_max_abs": repeat, "rows": receipts})


def score(run, role):
    from app.core.config import settings
    from app.detectors.audio import calibration
    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.run_audio_specialists_v15 import load_voice

    lock, rows = frozen(run)
    require(settings.audio_detector_mode == 'candidate', "Voice calibration mode differs")
    expanded = common.read(run / 'enhancements.json')
    require(expanded['freeze_sha256'] == common.sha(run / 'freeze.json'), "Enhancements lack the frozen input commitment")
    points_hash = None
    if role == 'held':
        point_path = run / 'points.json'
        require(common.read(point_path)['selection_sha256'] == common.sha(run / 'selection.json'), "Points are not frozen to selection")
        points_hash = common.sha(point_path)
    assets = {key: value for key, value in lock['voice_assets'].items() if key not in common.SOUND_ASSETS}
    voice = load_voice({"weights": str(common.WEIGHTS), "assets": assets})
    vad = SpeechActivity(common.WEIGHTS / 'silero_vad.jit')
    scored = []
    for row, extra in zip(rows, expanded['rows'], strict=True):
        if row['role'] != role:
            continue
        require(row['id'] == extra['id'], "Mixed up enhanced inputs")
        raw = (run / 'pcm' / f"{row['id']:03}.pcm").read_bytes()
        require(common.digest(raw) == row['pcm_sha256'], "Input PCM differs")
        original = prior.enhancement.float_pcm(raw)
        incumbent = publisher.radar.adapter_result(original, voice, vad)['decision']
        enhanced = []
        for name in HEADS:
            pcm = (run / 'enhanced' / f"{row['id']:03}-{name}.pcm").read_bytes()
            require(len(pcm) == len(raw) and common.digest(pcm) == extra['enhancers'][name]['pcm_sha256'], "Enhanced PCM differs")
            enhanced.append(prior.enhancement.float_pcm(pcm))
        responses = voice.predict(enhanced)
        candidates = {name: {**extra['enhancers'][name], 'admitted': window_in_speech(vad.spans(pcm), 0., 3.),
                             'logit': calibration.logit(response)}
                      for name, pcm, response in zip(HEADS, enhanced, responses, strict=True)}
        scored.append({**row, 'incumbent': incumbent, 'candidates': candidates})
        if len(scored) % 16 == 0:
            print(f'{role} scored {len(scored)} views', flush=True)
    save(run / f'{role}.json', {'freeze_sha256': common.sha(run / 'freeze.json'), 'points_sha256': points_hash,
                               'enhancements_sha256': common.sha(run / 'enhancements.json'), 'rows': scored})


def point(run):
    from scripts.review_audio_nemotron_noise_transfer import points_for

    _, _ = frozen(run)
    source = run / 'selection.json'
    receipt = common.read(source)
    require(receipt['freeze_sha256'] == common.sha(run / 'freeze.json'), "Unfrozen selection scores")
    save(run / 'points.json', {'selection_sha256': common.sha(source), 'points': points_for(receipt['rows'])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'enhance', 'selection', 'point', 'held'))
    parser.add_argument('run', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--old-model', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.run, args.inputs, args.previous, args.labels)
    elif args.action == 'enhance':
        enhance(args.run, args.model, args.old_model)
    elif args.action == 'point':
        point(args.run)
    else:
        score(args.run, args.action)


if __name__ == '__main__':
    main()
