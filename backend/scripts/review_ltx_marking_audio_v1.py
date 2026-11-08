"""Independent full sample/bit/file replay; no model or provider is called."""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from scripts import continue_ltx_marking_audio_v1 as run

EXPECTED = np.array([0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1], np.int64)


@lru_cache(maxsize=17)
def bit_tail(matches):
    return sum(value.bit_count() >= matches for value in range(65536)) / 65536


def review(trial, output):
    run.check(trial)
    result = run.old.base.read(trial / 'result.json')
    native_profile = run.old.base.read(trial / 'profile/profile.json')
    require = run.old.base.require
    require(native_profile['child_exit_code'] == 0 and native_profile['guard'] is None,
            'Original terminal complete native run required')
    require(result['cases'] == 80 and result['generator_forwards'] == 17 and result['detector_forwards'] == 81
            and result['exact_first_generator_and_detector_repeats'] is True
            and result['journal_sha256'] == run.old.base.digest(trial / 'journal.jsonl'), 'Original complete native journal/counters required')
    prefix = run.old.base.read(trial / 'original-prefix-parity.json')
    require(prefix['cases'] == 6 and prefix['waveforms_probabilities_messages_and_results_exact']
            and prefix['original_failure_unchanged'], 'Original failed-attempt prefix must reproduce exactly')
    rows = [json.loads(s) for s in (trial / 'journal.jsonl').read_text().splitlines()]
    require(len(rows) == len({r['id'] for r in rows}) == 80, 'All80 unique cases required')
    expected = {run.case_id(f'{r}-{i}-{n}') for r in (24000, 48000) for i in range(4) for n in
        list(run.old.POLICY['conditions']) + [k + '/' + c for k in run.old.POLICY['codecs'] for c in ('unmarked', 'matching-public-payload')]}
    require({r['id'] for r in rows} == expected, 'No input may be omitted or replaced')
    count, samples, files = Counter(), 0, 0
    for row in rows:
        path = trial / 'private' / row['packet']
        require(path.parent.resolve() == (trial / 'private').resolve() and run.old.base.digest(path) == row['packet_sha256'], 'Saved packet changed')
        with np.load(path, allow_pickle=False) as data:
            probability, messages, pcm = data['probabilities'], data['messages'], data['waveform']
            require(probability.dtype == messages.dtype == pcm.dtype == np.float32
                and probability.shape == pcm.shape == (2, 2 * row['rate']) and messages.shape == (2, 16)
                and all(np.isfinite(v).all() for v in (probability, messages, pcm))
                and ((probability >= 0) & (probability <= 1)).all() and ((messages >= 0) & (messages <= 1)).all(), 'Complete finite native channel/sample/message arrays required')
            require(run.old.method.sha_array(pcm) == row['pcm_sha256'], 'Source PCM commitment changed')
            if row['condition'] == 'unmarked':
                require(np.array_equal(pcm, run.old.method.carrier(row['rate'], row['carrier'])), 'Authored source carrier changed')
            fraction = np.count_nonzero(probability > .5) / probability.size
            bits = (messages[0] + messages[1]) / 2 > .5
            matches = sum(int(a) == int(b) for a, b in zip(bits, EXPECTED, strict=True))
            recorded = row['result']
            require(recorded['family_presence_detected'] == (fraction >= .5)
                and recorded['positive_native_sample_fraction'] == fraction
                and recorded['bit_matches'] == matches and recorded['bit_accuracy'] == matches / 16
                and recorded['bit_match_p_value'] == bit_tail(matches)
                and recorded['public_payload_exact_match'] == (matches == 16)
                and recorded['provider_authenticated'] is recorded['used_in_confidence'] is recorded['early_stop_eligible'] is False,
                'Independent integer-presence/bit enumeration or no-authorship policy differs')
            state = ('recognized_public_payload' if matches == 16 else 'family_signal_only') if fraction >= .5 else 'not_detected'
            require(recorded['state'] == state, 'Source family signal cannot be promoted to authentication')
            count[row['condition']] += 1
            samples += probability.size
            if 'file_proof' in row:
                kind, condition = row['condition'].split('/')
                file = trial / 'private' / f"codec-{row['rate']}-{row['carrier']}-{kind}-{condition}" / ('input.wav' if kind == 'pcm16-wav' else 'input.m4a')
                require(run.old.base.digest(file) == row['file_proof']['file_sha256'], 'Encoded source file changed')
                data = subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(file), '-map', '0:a:0',
                    '-t', '2', '-ar', str(row['rate']), '-ac', '2', '-f', 'f32le', 'pipe:1'], timeout=30)
                decoded = np.frombuffer(data, dtype='<f4').reshape(-1, 2).T.copy()
                require(np.array_equal(decoded, pcm) and run.old.method.sha_array(decoded) == row['file_proof']['decoded_pcm_sha256'], 'Actual codec decode/sample-clock extent changed')
                stream = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(file)], timeout=20))['streams'][0]
                require(all(stream.get(k) == v for k, v in row['file_proof']['native_stream'].items()), 'Actual native stream timing changed')
                files += 1
    summary = {}
    for condition in sorted(count):
        part = [r for r in rows if r['condition'] == condition]
        summary[condition] = {'cases': len(part), 'family_presence': sum(r['result']['family_presence_detected'] for r in part),
            'exact_public_payload_with_presence': sum(r['result']['state'] == 'recognized_public_payload' for r in part),
            'bit_matches': {str(k): v for k, v in Counter(r['result']['bit_matches'] for r in part).items()}}
    require(summary == result['summary'] and files == 32, 'All original aggregate/file results must reproduce')
    run.check(trial)
    run.old.base.write(output, {'schema': 'chipmunk.ltx-audio-complete-sample-review/1', 'cases': 80,
        'native_channel_sample_responses_replayed': samples, 'actual_file_decodes_and_clocks_replayed': files,
        'independent_bit_tests_replayed': 80, 'first_failed_attempt_prefix_cases_exact': 6,
        'complete_summary_exact': True, 'new_model_forwards': 0, 'outside_human_reviewers': 0,
        'provider_authenticated': False, 'used_in_confidence': False, 'early_stop_eligible': False,
        'reviewer_sha256': run.old.base.digest(Path(__file__))})
    print({'reviewed_cases': 80, 'sample_responses': samples, 'file_decodes': files}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    review(args.trial, args.output)
