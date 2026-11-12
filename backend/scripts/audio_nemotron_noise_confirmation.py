# SPDX-License-Identifier: Apache-2.0
"""Test the frozen two-enhancer comparison on unused RADAR clips and backgrounds."""

from __future__ import annotations

import argparse
import tarfile
import tempfile
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from scripts import audio_nemotron_noise_transfer as initial
from scripts import audio_radar_transfer as radar

common, require, save = initial.common, initial.require, initial.save
SCHEMA = 'chipmunk.nemotron-noise-confirmation/1'
PARENT = common.PUBLIC / 'audio-nemotron-noise-transfer-2026-11-12.json'
EARLIER = common.PUBLIC / 'audio-radar-transfer-2026-11-12.json'
BACKGROUNDS = {
    'market.ogg': ('7e677dad7ce4cbebf7b4e9d3555c0af22fe51ceb2514602b5f75a50e6e12ec85', (4, 10, 16, 22)),
    'cori-street.ogg': ('50e5eb59b296580340898f16e737f826a9fd7fef6fc45b07bb3e0318a9e58495', (8, 36, 94, 146)),
}
RULE = {'minimum_added_noisy_synthetic_sources': 4, 'maximum_added_noisy_genuine_sources': 2,
        'maximum_added_clean_or_isolated': 0, 'minimum_scenes_with_gains': 2}


def source_roster(radar_dir):
    pins = {name: pin for name, pin in radar.SOURCE_FILES.items() if name in {'archive-prefix.gz', 'labels.txt', 'languages.txt', 'LICENSE.txt'}}
    require({name: common.sha(radar_dir / name) for name in pins} == pins, 'Official source files differ')
    excluded = {row['id'] for row in common.read(EARLIER)['rows']}
    excluded_pcm = {row['pcm_sha256'] for row in common.read(EARLIER)['rows']}
    labels = {name: rest for name, *rest in map(str.split, (radar_dir / 'labels.txt').read_text().splitlines())}
    languages = dict(map(str.split, (radar_dir / 'languages.txt').read_text().splitlines()))
    candidates = defaultdict(list)
    try:
        with tarfile.open(radar_dir / 'archive-prefix.gz', 'r|gz') as archive:
            for member in archive:
                if not member.isfile() or not (match := radar.MEMBER.fullmatch(member.name)):
                    continue
                with archive.extractfile(member) as source:
                    data = source.read()
                name = match[1]
                duration, label, *intervals = labels[name]
                if (name not in excluded and languages[name] == 'EN' and 3 <= Decimal(duration) <= 14
                        and intervals == [f'0.0000-{duration}-{label}']):
                    candidates[label].append((name, duration, data))
    except (EOFError, tarfile.ReadError) as error:
        require(isinstance(error, EOFError) or 'end of data' in str(error), 'Unexpected prefix read error')
    require({label: len(rows) for label, rows in candidates.items()} == {'bonafide': 22, 'spoof': 17}, 'Unexpected fresh source candidates')
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'source.flac'
        from app.detectors.audio.decode import decode_audio

        seen_pcm = set(excluded_pcm)
        for label, count in (('bonafide', 14), ('spoof', 14)):
            ordered = sorted(candidates[label], key=lambda row: common.digest(f'{SCHEMA}/{row[0]}'.encode()))
            for name, duration, data in ordered[:count]:
                path.write_bytes(data)
                pin = common.digest(data)
                decoded = decode_audio(path, source_sha256=pin, max_seconds=15)
                pcm = decoded.pcm
                require(decoded.mapping['pcm_sha256'] not in seen_pcm and 48_000 <= len(pcm) <= 224_000
                        and decoded.mapping['timeline'] == [{'start_sample': 0, 'end_sample': len(pcm), 'source_pts': 0}]
                        and abs(round(Decimal(duration) * 16_000) - len(pcm)) <= 3, 'Native source clock or identity differs')
                seen_pcm.add(decoded.mapping['pcm_sha256'])
                start = (len(pcm) - 48_000) // 2
                yield {'id': name, 'label': label, 'sha256': pin, 'source_pcm_sha256': decoded.mapping['pcm_sha256'],
                       'samples': len(pcm), 'center_start': start}, pcm[start:start + 48_000]


def prepare(run, radar_dir, inputs):
    parent = common.read(PARENT)
    require(parent['result']['arms']['new_enhancer']['advances'], 'Earlier comparison did not qualify for confirmation')
    selected = list(source_roster(radar_dir))
    scenes = {}
    for name, (pin, _) in BACKGROUNDS.items():
        require(common.sha(inputs / name) == pin, 'Confirmation background differs')
        scenes[name] = initial.prior.enhancement.float_pcm(common.decode(inputs / name))
    run.mkdir(mode=0o700)
    (run / 'pcm').mkdir()
    rows = []

    def keep(values, scene, label, group, origin):
        raw = initial.prior.enhancement.pcm(values)
        require(len(raw) == 96_000, 'Input differs from the exact 3 s clock')
        row = {'id': len(rows), 'role': 'held', 'scene': scene, 'label': label, 'group': group, 'family': None,
               'origin': origin, 'pcm_sha256': common.digest(raw)}
        (run / 'pcm' / f"{row['id']:03}.pcm").write_bytes(raw)
        rows.append(row)

    ranks = Counter()
    for row, pcm in selected:
        rank = ranks[row['label']]
        ranks[row['label']] += 1
        offsets = {name: starts[rank % 4] * 16_000 for name, (_, starts) in BACKGROUNDS.items()}
        noise = {name: scenes[name][a:a + 48_000] for name, a in offsets.items()}
        for scene, values in initial.prior.match_views(pcm, noise, -5).items():
            keep(values, scene, row['label'], row['id'], {**row, 'background_start': offsets.get(scene)})
    for scene, (_, starts) in BACKGROUNDS.items():
        for index, seconds in enumerate(starts):
            a = seconds * 16_000
            keep(scenes[scene][a:a + 48_000], scene, 'isolated', f'{scene}-{index}', {'background_start': a})
    require(len(rows) == 92 and len({row['pcm_sha256'] for row in rows}) == 92, 'Expected distinct holdout waveforms')
    lock = {**parent['freeze'], 'schema': SCHEMA, 'parent_sha256': common.sha(PARENT),
            'runner_sha256': common.sha(Path(initial.__file__)), 'confirmation_sha256': common.sha(Path(__file__)),
            'reviewer_sha256': common.sha(Path(__file__).with_name('review_audio_nemotron_noise_confirmation.py')),
            'roster_sha256': common.digest(common.encoded(rows)), 'backgrounds': BACKGROUNDS,
            'confirmation_rule': RULE, 'decision_points': parent['result']['points'],
            'radar_archive_sha256': radar.SOURCE_FILES['archive-prefix.gz'], 'earlier_radar_receipt_sha256': common.sha(EARLIER)}
    save(run / 'roster.json', rows)
    save(run / 'freeze.json', lock)
    save(run / 'selection.json', {'from_parent_sha256': common.sha(PARENT), 'rows': []})
    save(run / 'points.json', {'selection_sha256': common.sha(run / 'selection.json'), 'points': lock['decision_points']})
    print(f'Frozen {len(rows)} holdout views. {common.sha(run / "freeze.json")}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('radar', type=Path)
    parser.add_argument('inputs', type=Path)
    args = parser.parse_args()
    prepare(args.run, args.radar, args.inputs)


if __name__ == '__main__':
    main()
