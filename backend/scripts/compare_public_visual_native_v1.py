"""Compare completed Mac/Linux authored contracts without changing their checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FILES = ('freeze.json', 'result.json', 'profile/profile.json', 'audit/audit.json',
         'audit-profile/profile.json')
SCIENCE = 'c2ef887cc236490cdf0962f2754edf554d802a27'


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def completed(root):
    data = {name: read(root / name) for name in FILES}
    result, freeze, audit = (data[name] for name in ('result.json', 'freeze.json', 'audit/audit.json'))
    require(result['policy'] == freeze['policy'], 'Within-run policy changed')
    require((result['case_count'], result['generator_forwards'], result['detector_forwards']) == (168, 18, 173),
            'Complete original case/forward denominators required')
    require(len(freeze['code_sha256']) == 8 and freeze['neural_forwards_before_freeze'] == 0,
            'Original eight-file freeze before inference required')
    require(all(result[k] is False for k in ('provider_authenticated', 'used_in_confidence',
                                            'early_stop_eligible', 'serving_admitted')),
            'Research responses cannot become serving admission')
    require(audit['schema'] == 'chipmunk.public-visual-watermark-source-audit/1'
            and audit['cases'] == 168 and audit['presence_pixels_replayed'] == 11010048
            and audit['binary_bit_predictions_replayed'] == 352321536
            and audit['codec_clips_redecoded'] == 6 and audit['codec_frames_redecoded'] == 48
            and audit['source_times_verified'] is True and audit['neural_forwards'] == 0,
            'Complete original non-neural source/mask/time audit required')
    for name, module in (('profile/profile.json', 'scripts.run_public_visual_watermark_v1'),
                         ('audit-profile/profile.json', 'scripts.review_public_visual_watermark_v1')):
        profile = data[name]
        require(profile['module'] == module and profile['child_exit_code'] == 0 and profile['guard'] is None
                and profile['sample_errors'] == 0 and profile['seconds_limit'] == 900
                and 0 < profile['sampled_tree_rss_lower_bound_bytes'] <= 6 * 1024**3
                and 0 < profile['duration_s'] <= 900, 'Original unguarded completion required')
    return data


def compare(mac, linux):
    a, b = completed(mac), completed(linux)
    fa, fb = a['freeze.json'], b['freeze.json']
    for field in ('policy', 'code_sha256', 'source_assets_sha256'):
        require(fa[field] == fb[field], 'Cross-platform frozen science identity differs: ' + field)
    require(set(fa['packages']) == set(fb['packages']) and all(
        fa['packages'][k].split('+')[0] == fb['packages'][k].split('+')[0] for k in fa['packages']),
        'Core package versions changed; CPU build suffix alone may differ')
    ra, rb = a['result.json'], b['result.json']
    require(ra['load_receipt'] == rb['load_receipt'], 'Checkpoint/native architecture identity differs')
    environment = read(linux / 'environment.json')
    require(environment['science_commit'] == SCIENCE and environment['cuda_available'] is False
            and environment['machine'] == 'x86_64' and 'Linux' in environment['platform'],
            'Actual pinned Linux CPU environment required')
    def groups(result):
        rows = result['groups']
        out = {(r['kind'], r['alteration'], r['state']): r for r in rows}
        require(len(out) == len(rows) and sum(r['cases'] for r in rows) == 168, 'Complete unique groups required')
        return out
    ga, gb = groups(ra), groups(rb)
    require(ga.keys() == gb.keys(), 'Cross-platform group selection differs')
    comparisons = []
    for key in sorted(ga):
        x, y = ga[key], gb[key]
        require(x.keys() == y.keys() and all(x[k] == y[k] for k in
                ('cases', 'unmarked_cases', 'marked_cases')), 'Group denominator changed')
        differences = {k: {'mac': x[k], 'linux': y[k]} for k in x if x[k] != y[k]}
        comparisons.append({'kind': key[0], 'alteration': key[1], 'state': key[2],
                            'mac': x, 'linux': y, 'differences': differences})
    def hashes(root):
        return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FILES}
    return {'schema': 'chipmunk.public-visual-cross-platform-replay/1', 'source_commit': SCIENCE,
            'science_identity_matched': True, 'both_original_native_contracts_completed': True,
            'both_model_free_audits_completed': True, 'cross_platform_groups_exact': ga == gb,
            'groups_with_any_numerical_difference': sum(bool(r['differences']) for r in comparisons),
            'groups_with_changed_flags_or_misses': sum(any(k in r['differences'] for k in
                ('flagged', 'unmarked_flagged', 'marked_missed')) for r in comparisons),
            'mac_packages': fa['packages'], 'linux_environment': environment,
            'load_receipt': ra['load_receipt'], 'groups': comparisons,
            'input_receipt_sha256': {'mac': hashes(mac), 'linux': hashes(linux)},
            'outside_human_reviewers': 0, 'customer_files': 0, 'fresh_natural_media_cases': 0,
            'new_comparison_neural_forwards': 0, 'serving_changed': False,
            'scope': 'Same authored experiment re-executed on another machine; compare aggregate responses after each original within-runtime audit. Raw CI arrays are not exported. Not fresh-media/provider accuracy, human review or serving admission.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mac', type=Path)
    parser.add_argument('linux', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = compare(args.mac, args.linux)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('groups', 'input_receipt_sha256')}, indent=2))
