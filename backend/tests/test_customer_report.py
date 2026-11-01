from copy import deepcopy

from app.reports.customer_view import TENDENCY_RATE_EXPLANATION, customer_view
from app.reports.snapshot import snapshot_hash
from app.reports.specimens import specimen


def sample():
    s = specimen('video', 'standard')
    result = deepcopy(s.result)
    result['decision'].update(probability_ai_calibrated=.13, fusion_mode='artefact', code='no_ai_indicators',
                              ci90=[.06, .29], provenance_record=None, driving_hypothesis=None)
    result['condition_profile'] = {'within_validated_envelope': True}
    result['findings'] = [
        {'id': 'F1', 'family': 'F3_whole_image_generation', 'evidence_ids': ['whole'], 'contribution_logit': -1.51},
        {'id': 'F2', 'family': 'A1_voice', 'evidence_ids': ['voice'], 'contribution_logit': 0},
        {'id': 'F3', 'family': 'coverage', 'evidence_ids': ['whole']},
    ]
    result['signals'] = [
        {'id': 'whole', 'display_id': 'S01', 'status': 'ok', 'observed': 'mean frame logit -1.39; 0 of 8 frame(s) at logit > 0'},
        {'id': 'voice', 'display_id': 'S02', 'status': 'ok', 'outside_estimate': {'reason': 'zero_weight'},
         'observed': 'fused p(synthetic)=0.84; per model: private=0.84'},
    ]
    frozen = s.model_copy(update={'result': result})
    out = {'header_left': 'old', 'cover': [{'key': 'Review'}, {'key': 'Report'}], 'custody': {'rows': [{'actor': 'Local examiner'}]},
           'findings': [{'id': 'F1', 'title': 'A detection check', 'sentences': ['raw -1.39']},
                        {'id': 'F2', 'title': 'Uninformative'}, {'id': 'F3', 'title': 'Coverage', 'sentences': ['Two checks ran.']}],
           'summary': {'reasons': [{'id': 'F1'}], 'testing': 'Known error rates are not established.'},
           'sections': [{'key': 'signoff'}, {'key': 'appendix-c'}, {'key': 'findings'}],
           'evidence': {'packets': [{'id': 'A01', 'kind': 'audio', 'finding_ids': ['F2', 'F3'], 'signal_refs': [{'id': 'voice'}],
                                    'locator': 'PCM 4-8 s', 'observation': '0.917 on 4-8 s',
                                    'interpretation': "At least one recorded window response reaches the check's 0.5 inspection threshold."}]}}
    return frozen, out


def test_customer_findings_explain_negative_logits_and_excluded_voice_without_rewriting_records():
    frozen, out = sample()
    before = snapshot_hash(frozen)
    shown = customer_view(frozen, out, lambda x: x)
    f1, f2, coverage = shown['findings']
    assert f1['title'] == 'Visual-generation check (S01)'
    assert 'Leaning towards original content' in f1['inference']
    assert '0 of 8 sampled frames' in f1['observations'][0]
    assert 'negative' not in f1['reason'] and '-1.39' not in f1['reason']
    assert 'excluded from the combined estimate' in f2['inference']
    assert f2['support'][0]['href'] == '#multimedia-evidence-A01'
    assert 'private' not in f2['observations'][0]
    assert coverage['support'] == [] and 'Coverage information only' in coverage['inference']
    assert snapshot_hash(frozen) == before


def test_customer_service_attribution_and_error_rate_explanation():
    frozen, out = sample()
    shown = customer_view(frozen, out, lambda x: x)
    assert shown['custody']['rows'][0]['actor'] == 'Chipmunk Verify'
    assert 'internal audit record' in shown['custody']['actor_note']
    assert shown['summary']['testing'] == TENDENCY_RATE_EXPLANATION
    assert shown['expert_contact'] == 'sales@chipmunkrpa.com'
    assert [s['key'] for s in shown['sections']] == ['findings', 'source-evidence']
    assert not any(r['key'] == 'Review' for r in shown['cover'])
    assert shown['examiner'] is None


def test_customer_packet_inferences_keep_mixed_text_and_acoustics_uncertain():
    frozen, out = sample()
    out['evidence']['packets'] += [
        {'id': 'T01', 'kind': 'document', 'windows': [{'supports_ai': True}, {'supports_ai': False}]},
        {'id': 'B01', 'kind': 'acoustic'}, {'id': 'AV01', 'kind': 'av'},
    ]
    shown = customer_view(frozen, out, lambda x: x)
    packets = {p['id']: p for p in shown['evidence']['packets']}
    assert 'Mixed evidence' in packets['T01']['customer_inference']
    for key in ['B01', 'AV01']:
        assert 'No AI/original conclusion' in packets[key]['customer_inference']


def test_customer_partial_views_are_not_rewritten():
    frozen, _ = sample()
    out = {'findings': [{'id': 'F1', 'sentences': ['Preserved projection test.']}]}
    assert customer_view(frozen, deepcopy(out), lambda x: x) == out


def test_audio_only_excluded_voice_does_not_claim_a_visual_estimate():
    frozen, out = sample()
    frozen = frozen.model_copy(update={"analysis": frozen.analysis.model_copy(update={"kind": "audio"})})
    shown = customer_view(frozen, out, lambda x: x)
    assert "In this video" not in shown["summary"]["context"]
    assert "visual estimate" not in shown["summary"]["context"]


def test_small_contributions_are_not_standalone_leanings_and_new_rule_does_not_reuse_old_rates():
    frozen, out = sample()
    result = deepcopy(frozen.result)
    result['findings'][0]['contribution_logit'] = -.2
    frozen = frozen.model_copy(update={'result': result})
    out['testing'] = {'arrays': [{'caption': 'Old threshold false alarm rate'}], 'line': 'Old rate'}
    out['methods'] = {'headline_rule': {'rates': [{'fpr': .05}], 'sentences': ['old rule'], 'scope': 'old scope'}}
    shown = customer_view(frozen, out, lambda x: x)
    assert 'No reliable AI/original conclusion' in shown['findings'][0]['inference']
    assert 'small contribution' in shown['findings'][0]['inference']
    assert shown['testing']['arrays'] == []
    assert shown['methods']['headline_rule']['rates'] == []
    assert shown['methods']['headline_rule']['scope'] is None
    assert '80%' in shown['methods']['headline_rule']['note']


def test_customer_method_and_identity_do_not_reintroduce_historical_headline_or_draft_workflow():
    frozen, out = sample()
    out['methods'] = {'method': [{'key': 'Fusion', 'value': 'Calibrated fusion; conformal bands for the headline.'}]}
    shown = customer_view(frozen, out, lambda x: x)
    method = shown['methods']['method'][0]['value']
    assert 'conformal bands for the headline' not in method
    assert '80% point-estimate rule' in method
    assert 'DRAFT' not in shown['title'] + shown['keywords'] + shown['analysis_line']
    assert shown['customer_interval_value'].endswith('%')
