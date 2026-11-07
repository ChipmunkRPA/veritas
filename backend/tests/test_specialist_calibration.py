"""Authored synthetic pipeline checks, never real-model accuracy evidence."""
import copy
import hashlib

import pytest

from app.bench.specialists import MEDIA, ROLES, audit, digest, evaluate, train


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture():
    policy = {'specialists': []}
    files = []
    for modality in MEDIA:
        for name in ('first', 'second'):
            policy['specialists'].append({'id': modality + '.' + name, 'modality': modality,
                'family': name, 'target': modality + '.ai', 'contexts': ['normal', 'difficult'],
                'producer': {'kind': 'authored_synthetic_fixture', 'implementation_sha256': h('authored-score-generator-v1')},
                'routing_sha256': h(modality + name + 'score-blind-routing'),
                'higher_means_ai': True, 'raw_range': [-10., 10.]})
        for role in ROLES:
            for label in (0, 1):
                for index in range(4):
                    ident = f'{modality}-{role}-{label}-{index}'
                    units = []
                    for name in ('first', 'second'):
                        spec = next(s for s in policy['specialists'] if s['id'] == modality + '.' + name)
                        for k, context in enumerate(spec['contexts']):
                            if modality == 'image':
                                span = {'kind': 'pixels', 'x': k * 10, 'y': 0, 'width': 10,
                                    'height': 10, 'source_width': 20, 'source_height': 10}
                            else:
                                kind, unit, length = {'video': ('time', 'nanoseconds', 10**9),
                                    'audio': ('samples', 'samples', 16000), 'text': ('text', 'unicode_codepoints', 40)}[modality]
                                span = {'kind': kind, 'unit': unit, 'start': k * length,
                                    'end': (k + 1) * length, 'extent': 2 * length}
                                if modality == 'audio':
                                    span['sample_rate'] = 16000
                                elif modality == 'text':
                                    span['text_sha256'] = h('extracted-text-' + ident)
                                elif modality == 'video':
                                    span['frames'] = {'start': k * 24, 'end': (k + 1) * 24, 'extent': 48}
                                    span['region'] = {'kind': 'pixels', 'x': 0, 'y': 0, 'width': 640,
                                        'height': 360, 'source_width': 1920, 'source_height': 1080}
                            raw = 2. * label - 1. + (-4. if k == 0 else 4.) + .03 * index
                            units.append({'id': name + '.' + context, 'specialist': spec['id'],
                                'context': context, 'routing_sha256': spec['routing_sha256'],
                                'context_evidence_sha256': h(ident + context + 'measured-before-scores'),
                                'measurement_sha256': h(ident + name + context + str(raw)),
                                'location': span, 'status': 'ok', 'raw': raw, 'label': label,
                                'label_scope': 'local_unit', 'label_provenance_sha256': h(ident + name + context + 'truth')})
                    geometry = {'image': {'width': 20, 'height': 10},
                        'video': {'duration_ns': 2 * 10**9, 'frame_count': 48, 'width': 1920,
                            'height': 1080, 'pts_table_sha256': h('source-pts-' + ident)},
                        'audio': {'sample_count': 32000, 'sample_rate': 16000},
                        'text': {'codepoints': 80, 'text_sha256': h('extracted-text-' + ident)}}[modality]
                    files.append({'id': ident, 'group': ident, 'role': role, 'modality': modality, 'source_geometry': geometry,
                        'source_sha256': h('source-' + ident), 'input_sha256': h('input-' + ident),
                        'lineage': ['source-family-' + ident], 'label': label, 'label_scope': 'planned_units',
                        'label_provenance_sha256': h('scope-truth-' + ident),
                        'required_families': ['first', 'second'], 'units': units})
    return {'schema': 'chipmunk.specialist-evaluation/2', 'policy': policy, 'files': files}


@pytest.fixture(scope='module')
def trained():
    manifest = fixture()
    # Low support is deliberately for authored synthetic integration tests only.
    return manifest, train(manifest, min_groups_per_class=3)


def test_all_four_media_have_context_heads_and_independently_calibrated_combiners(trained):
    manifest, artifact = trained
    assert audit(manifest) == {'files': 128, 'units': 512, 'roles': {r: 32 for r in ROLES},
        'modalities': {m: 32 for m in MEDIA}, 'metadata_only': True}
    assert len(artifact['specialists']) == 16 and len(artifact['combiners']) == 4
    assert all(m['status'] == 'fitted' for m in artifact['specialists'].values())
    assert all(c['response']['status'] == c['calibration']['status'] == 'fitted' for c in artifact['combiners'].values())
    rows = evaluate(manifest, artifact)['outcomes']
    assert len(rows) == 32 and all(r['status'] == 'ok' and 0 < r['probability'] < 1 for r in rows)
    assert {r['modality'] for r in rows} == set(MEDIA) and artifact['serving_admitted'] is False


@pytest.mark.parametrize('key', ['group', 'source_sha256', 'input_sha256', 'lineage'])
def test_connected_sources_cannot_cross_any_of_the_four_roles(key):
    manifest = fixture()
    first = manifest['files'][0]
    later = next(f for f in manifest['files'] if f['role'] == 'fusion_fit')
    later[key] = copy.deepcopy(first[key])
    with pytest.raises(ValueError, match='leakage'):
        audit(manifest)


def test_whole_file_truth_is_not_local_truth_and_human_sections_in_ai_files_are_allowed():
    manifest = fixture()
    file = next(f for f in manifest['files'] if f['label'] == 1)
    file['units'][0]['label'] = 0
    assert audit(manifest)['units'] == 512
    file['units'][0]['label_scope'] = 'whole_file'
    with pytest.raises(ValueError, match='whole-file'):
        audit(manifest)


def test_confirmation_responses_do_not_fit_any_head_or_combiner(trained):
    manifest, artifact = trained
    changed = copy.deepcopy(manifest)
    for file in changed['files']:
        if file['role'] == 'confirmation':
            for row in file['units']:
                row['raw'] = -row['raw']
                row['measurement_sha256'] = h('new-' + row['measurement_sha256'])
    refit = train(changed, min_groups_per_class=3)
    assert refit == artifact
    assert evaluate(changed, artifact)['outcomes'] != evaluate(manifest, artifact)['outcomes']


@pytest.mark.parametrize('status', ['failed', 'skipped', 'not_applicable', 'unavailable'])
def test_missing_local_evidence_is_not_originality_evidence(status, trained):
    manifest, artifact = trained
    changed = copy.deepcopy(manifest)
    file = next(f for f in changed['files'] if f['role'] == 'confirmation' and f['label'] == 0)
    file['units'][0].update(status=status, raw=None, reason='Authored missing-evidence fixture')
    row = next(r for r in evaluate(changed, artifact)['outcomes'] if r['id'] == file['id'])
    assert row['status'] == 'inconclusive' and row['probability'] is None


def test_same_family_duplicate_observation_does_not_increase_combined_confidence(trained):
    manifest, artifact = trained
    changed = copy.deepcopy(manifest)
    for file in changed['files']:
        if file['role'] == 'confirmation':
            duplicate = copy.deepcopy(file['units'][0])
            duplicate['id'] = 'duplicate-window'
            file['units'].append(duplicate)
    # The planned coverage changes and must be frozen prospectively in a new
    # artifact; same trained numbers then give identical max-family features.
    from app.bench.specialists import plan_digest

    adjusted = copy.deepcopy(artifact)
    adjusted['plan_sha256'] = plan_digest(changed)
    adjusted['sha256'] = digest({k: v for k, v in adjusted.items() if k != 'sha256'})
    assert evaluate(changed, adjusted)['outcomes'] == evaluate(manifest, artifact)['outcomes']


def test_sparse_context_has_no_invented_probability():
    manifest = fixture()
    artifact = train(manifest)  # Default requires25 distinct groups per class.
    assert all(m['status'] == 'insufficient_support' for m in artifact['specialists'].values())
    assert all(r['probability'] is None for r in evaluate(manifest, artifact)['outcomes'])


@pytest.mark.parametrize('modality', MEDIA)
def test_coordinates_remain_bound_to_source_units(modality):
    manifest = fixture()
    file = next(f for f in manifest['files'] if f['modality'] == modality)
    span = file['units'][0]['location']
    if modality == 'image':
        span['x'] = span['source_width']
    else:
        span['end'] = span['extent'] + 1
    with pytest.raises(ValueError, match='outside'):
        audit(manifest)


def test_changed_training_scores_or_context_cannot_use_old_artifact(trained):
    manifest, artifact = trained
    changed = copy.deepcopy(manifest)
    changed['files'][0]['units'][0]['raw'] += .1
    with pytest.raises(ValueError, match='Changed frozen'):
        evaluate(changed, artifact)
    changed = copy.deepcopy(manifest)
    changed['files'][0]['units'][0]['routing_sha256'] = h('score-dependent-router')
    with pytest.raises(ValueError, match='routing'):
        audit(changed)


def test_ambiguous_family_keys_and_failed_scores_are_refused():
    manifest = fixture()
    manifest['policy']['specialists'][0]['family'] = 'a,b'
    with pytest.raises(ValueError):
        audit(manifest)
    manifest = fixture()
    manifest['files'][0]['units'][0].update(status='failed', reason='test', raw=0.)
    with pytest.raises(ValueError, match='no response'):
        audit(manifest)


def test_missing_local_labels_are_not_filled_from_file_truth():
    manifest = fixture()
    for file in manifest['files']:
        if file['role'] == 'specialist_cal':
            for unit in file['units']:
                unit['label'] = None
    artifact = train(manifest, min_groups_per_class=3)
    assert all(v['status'] == 'insufficient_support' for v in artifact['specialists'].values())
    assert all(r['probability'] is None for r in evaluate(manifest, artifact)['unit_outcomes'])


def test_model_observations_require_native_and_commercial_review_commitments():
    manifest = fixture()
    manifest['policy']['specialists'][0]['producer']['kind'] = 'model'
    with pytest.raises(ValueError, match='commercial-license'):
        audit(manifest)


def test_cli_freeze_protects_fitting_data_and_preserves_future_confirmation(trained):
    from scripts.specialist_calibration import fit, freeze

    manifest, _ = trained
    lock = freeze(manifest, 3, .001)
    fitted = fit(manifest, lock)
    changed = copy.deepcopy(manifest)
    for file in changed['files']:
        if file['role'] == 'confirmation':
            for row in file['units']:
                row['raw'] *= -1
    assert fit(changed, lock) == fitted
    changed['files'][0]['units'][0]['raw'] += .1
    with pytest.raises(ValueError, match='changed after freeze'):
        fit(changed, lock)


@pytest.mark.parametrize('modality', MEDIA)
def test_unrelated_genuine_regions_do_not_lower_other_positive_evidence(modality, trained):
    from app.bench.specialists import _features

    manifest, artifact = trained
    file = copy.deepcopy(next(f for f in manifest['files'] if f['role'] == 'confirmation'
        and f['modality'] == modality and f['label'] == 1))
    neutral, genuine = copy.deepcopy(file), copy.deepcopy(file)
    for unit_a, unit_b in zip(neutral['units'], genuine['units'], strict=True):
        if unit_a['specialist'].endswith('.second'):
            head = artifact['specialists'][unit_a['specialist'] + '|' + unit_a['context']]
            coefficient = head['coefficients'][0]
            assert coefficient > 0
            unit_a['raw'] = head['center'][0] - head['scale'][0] * head['intercept'] / coefficient
            unit_b['raw'] = head['center'][0] + head['scale'][0] * (-2 - head['intercept']) / coefficient
            unit_a['label'] = unit_b['label'] = 0
    original = _features(neutral, artifact)
    changed = _features(genuine, artifact)
    assert original[0] > 0  # Positive evidence in the first region stays present.
    assert changed == pytest.approx(original, abs=1e-12)


@pytest.mark.parametrize('modality', MEDIA)
def test_local_geometry_cannot_silently_change_source(modality):
    manifest = fixture()
    file = next(f for f in manifest['files'] if f['modality'] == modality)
    keys = {'image': 'width', 'video': 'duration_ns', 'audio': 'sample_rate', 'text': 'codepoints'}
    file['source_geometry'][keys[modality]] += 1
    with pytest.raises(ValueError):
        audit(manifest)


def test_video_local_region_is_checked_against_source_frame_geometry():
    manifest = fixture()
    file = next(f for f in manifest['files'] if f['modality'] == 'video')
    file['units'][0]['location']['region']['x'] = 1920
    with pytest.raises(ValueError, match='outside'):
        audit(manifest)
