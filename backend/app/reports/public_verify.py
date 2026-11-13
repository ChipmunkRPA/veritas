"""Offline public-derivative checks: standard library and canonical JSON only.

This intentionally does not import application settings, model registries, databases,
or modality runtimes. It verifies signed public commitments, not inference accuracy.
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
import wave

from app.core.canonical import canonical_hash
from app.evidence import (
    document_scope,
    email_alternatives,
    face_boundary_binding,
    face_return_coverage,
    face_short_window_binding,
    face_spatial_cut_binding,
)
from app.evidence.face_alignment_context import validate as validate_face_alignment
from app.evidence.face_boundary_binding import validate_packet as validate_face_boundary
from app.evidence.face_interval_binding import validate_packet as validate_face_interval
from app.evidence.face_lighting_context import validate as validate_face_lighting
from app.evidence.image_copy_similarity_binding import (
    AFFINE_VERSIONS,
    REFLECTION_VERSIONS,
    SKIPPED_CREDENTIALS,
    box_bounds,
    copy_records,
    validate_copy_scope,
    validate_similarity,
)
from app.evidence.image_copy_similarity_binding import (
    url as copy_asset_url,
)
from app.evidence.mp4 import inspect_mp4, validate_clip_binding
from app.evidence.video_packet_versions import (
    ACOUSTIC_VERSIONS,
    ALIGNMENT_CONTEXT_VERSIONS,
    CONTINUITY_VERSIONS,
    LIGHTING_CONTEXT_VERSIONS,
    NESTED_VERSIONS,
)
from app.evidence.video_packet_versions import VERSIONS as VIDEO_VERSIONS

SCHEMA = 'chipmunk.public-evidence/1'
_SHA = re.compile(r'[a-f0-9]{64}')
TYPES = {'image/png': 'png', 'audio/wav': 'wav', 'video/mp4': 'mp4'}
IMAGE_PAIR_VERSION = 'image-evidence-15'
IMAGE_EDITOR_PREVIEW = 'thumbnail_evidence'
VERSIONS = {'image': {'image-evidence-1', 'image-evidence-2', 'image-evidence-3', 'image-evidence-4', 'image-evidence-5',
                      'image-evidence-6', IMAGE_PAIR_VERSION} | REFLECTION_VERSIONS, 'video': VIDEO_VERSIONS,
            'audio': {'audio-evidence-1', 'audio-evidence-2', 'audio-evidence-3'}, 'document': document_scope.VERSIONS}


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def _document(archive, names, reference, expected_file):
    _require(isinstance(reference, dict) and reference.get('file') == expected_file and expected_file in names,
             f'The {expected_file} reference is missing or invalid')
    data = archive.read(expected_file)
    _require(hashlib.sha256(data).hexdigest() == reference.get('sha256'), f'{expected_file} has a different committed hash')
    document = json.loads(data)
    _require(isinstance(document, dict), f'{expected_file} must be a JSON object')
    return document


def _checks(archive, names: set[str], bundle: dict) -> list[tuple[str, bool, str]]:
    _require(bundle.get('reproduction_supported') is False, 'A public derivative cannot claim inference reproduction')
    analysis = bundle['analysis']
    _require(isinstance(analysis, dict) and analysis.get('kind') in {'image', 'video', 'audio', 'document'},
             'The public analysis identity is unsupported')
    source = bundle['source_commitment']
    _require(isinstance(source, dict) and bool(_SHA.fullmatch(str(source.get('sha256', '')))), 'Invalid source commitment')
    _require(bool(_SHA.fullmatch(str(analysis.get('original_result_sha256', '')))), 'Invalid original-result commitment')
    result = _document(archive, names, bundle.get('result'), 'result.json')
    _require(result.get('schema') == 'chipmunk.public-result/1' and result.get('is_projection') is True
             and bundle['result'].get('is_projection') is True, 'The result must identify itself as a public projection')
    _require(result.get('original_result_sha256') == analysis['original_result_sha256'], 'The result commits to a different original')
    _require(set(result) <= {'schema', 'is_projection', 'original_result_sha256', 'decision', 'findings', 'checks', 'note',
                             IMAGE_EDITOR_PREVIEW, face_return_coverage.COMMITMENT, email_alternatives.COMMITMENT,
                             face_short_window_binding.COMMITMENT},
             'The public result contains unsupported internal fields')
    evidence = _document(archive, names, bundle.get('evidence'), 'evidence.json')
    binding = bundle['evidence']
    _require(evidence.get('schema') == f"chipmunk.{analysis['kind']}-evidence/1" and evidence.get('status') == 'ready',
             'The evidence schema or readiness does not match the analysis')
    _require(evidence.get('version') in VERSIONS[analysis['kind']], 'The evidence version is not supported by this verifier')
    actual = canonical_hash({k: v for k, v in evidence.items() if k != 'sha256'})
    _require(actual == evidence.get('sha256') == binding.get('set_sha256'), 'The evidence set hash does not recompute')
    _require(evidence.get('id') == binding.get('id') and evidence.get('version') == binding.get('version')
             and evidence['schema'] == binding.get('schema'), 'The evidence set identity or version differs')
    _require(evidence.get('analysis_id') == analysis.get('id') and evidence.get('revision') == analysis.get('revision')
             and evidence.get('original_result_hash') == analysis['original_result_sha256']
             and evidence.get('source', {}).get('sha256') == source['sha256'] == binding.get('source_sha256'),
             'The evidence source/result/revision binding differs')
    if analysis['kind'] == 'video':
        from app.evidence import face_return_binding
        from app.evidence.picture_hold_binding import validate as validate_picture_hold

        validate_picture_hold(evidence)
        face_boundary_binding.validate_set(evidence)
        face_spatial_cut_binding.validate_set(evidence)
        face_return_binding.validate_set(evidence)
        face_return_coverage.validate_commitment(evidence, result.get(face_return_coverage.COMMITMENT))
        window_packets = [p for p in evidence['packets'] if p.get('semantics') == face_short_window_binding.METHOD]
        face_short_window_binding.validate_commitment(evidence, window_packets, result.get(face_short_window_binding.COMMITMENT))
        if window_packets:
            _require(any(check['id'] == window_packets[0]['signal_refs'][0]['id'] and check.get('status') == 'ok'
                         and check.get(face_short_window_binding.CHECKS_COMMITMENT) == [canonical_hash(p['face_return']) for p in window_packets]
                         for check in result.get('checks', [])), 'Overlapping face windows differ from their corresponding public check')
        for packet in evidence.get('packets', []):
            validate_clip_binding(packet, evidence['source'])
            validate_face_interval(packet, evidence['source'])
            validate_face_boundary(packet, evidence['source'], evidence['version'])
            face_spatial_cut_binding.validate_packet(packet, evidence['source'], evidence['version'])
            if cut := packet.get('face_spatial_cut'):
                _require(any(check['id'] == packet['signal_refs'][0]['id'] and check.get('status') == 'ok'
                             and check.get('face_spatial_cut_measurement_sha256') == canonical_hash(cut)
                             for check in result.get('checks', [])),
                         'Picture-cut context differs from its corresponding public check')
            if boundary := packet.get('face_boundary'):
                measurement = canonical_hash(boundary)
                _require(any(check['id'] == packet['signal_refs'][0]['id'] and check.get('status') == 'ok'
                             and check.get('face_boundary_measurement_sha256') == measurement
                             for check in result.get('checks', [])),
                         'Refined face timing differs from its corresponding public check')
            if packet.get('face_return') is not None or packet.get('semantics') in face_return_binding.METHOD_VERSIONS:
                face_return_binding.validate_packet(packet, evidence['source'])
                if packet['semantics'] != face_short_window_binding.METHOD:
                    measurement = canonical_hash(packet['face_return'])
                    commitment = ('second_short_face_change_measurement_sha256' if packet['semantics'] in face_return_binding.SECOND_METHODS
                                  else 'short_face_change_measurement_sha256')
                    _require(any(check['id'] == packet['signal_refs'][0]['id'] and check.get('status') == 'ok'
                                 and check.get(commitment) == measurement
                                 for check in result.get('checks', [])),
                             'Short face-change evidence differs from its corresponding public check')
            if packet.get('whole_face_review') is not None or packet.get('semantics') == 'paired-whole-face-samples/1':
                from app.evidence.whole_face_binding import validate_packet as validate_whole_face

                validate_whole_face(packet, evidence['source'])
                measurement = canonical_hash(packet['whole_face_review'])
                _require(any(check['id'] == packet['signal_refs'][0]['id'] and check.get('status') == 'ok'
                             and check.get('paired_face_measurement_sha256') == measurement
                             for check in result.get('checks', [])),
                         'Paired face evidence differs from its corresponding public check')
            if evidence['version'] in ALIGNMENT_CONTEXT_VERSIONS:
                validate_face_alignment(packet)
            if evidence['version'] in LIGHTING_CONTEXT_VERSIONS:
                validate_face_lighting(packet, evidence['source'])
        if evidence['version'] in NESTED_VERSIONS:
            from app.evidence.audio_binding import validate_audio_channel
            from app.evidence.av_binding import validate_audio_clock_coverage, validate_av_channel

            validate_audio_channel(evidence['audio'], source['sha256'])
            if evidence['version'] in ACOUSTIC_VERSIONS and evidence['audio']['coverage'] is not None:
                _require(bool(evidence['audio'].get('acoustic_context')), 'Video acoustic context is missing')
            if evidence['version'] in CONTINUITY_VERSIONS and evidence['audio']['coverage'] is not None:
                _require(evidence['audio']['acoustic_context']['schema'] == 'chipmunk.acoustic-context/2', 'Video continuity measurements are missing')
            validate_av_channel(evidence['av'], evidence['source'])
            validate_audio_clock_coverage(evidence)
            packet_ids = [p['id'] for part in (evidence, evidence['audio'], evidence['av']) for p in part['packets']]
            _require(len(packet_ids) == len(set(packet_ids)), 'Video evidence packet identifiers collide across channels')
            checks = {check['id'] for check in result.get('checks', [])}
            _require(all(ref['id'] in checks for key in ('audio', 'av') for packet in evidence[key]['packets']
                         for ref in packet['signal_refs']), 'A nested video observation has no corresponding published check')
            if all(evidence[key]['coverage'] is not None for key in ('audio', 'av')):
                _require(all(evidence['audio']['source'][key] == evidence['av']['source'][key]
                             for key in ('pcm_sha256', 'wav_sha256', 'mapping_sha256', 'samples', 'sample_rate', 'timeline')),
                         'Video voice and alignment evidence use different audio inputs')
        else:
            _require(not {'audio', 'av'}.intersection(evidence), 'Historical video evidence has unexpected nested channels')
    else:
        _require(face_return_coverage.COMMITMENT not in result, 'Brief face-change coverage belongs to video evidence')
        _require(face_short_window_binding.COMMITMENT not in result, 'Overlapping face windows belong to video evidence')
    if analysis['kind'] == 'image':
        from app.evidence.image_eligibility import validate_inspection_coverage

        scope = (result.get('decision') or {}).get('image_copy_coverage')
        image_base = evidence
        if evidence['version'] == IMAGE_PAIR_VERSION:
            from app.evidence import image_embedded_pair_binding

            image_embedded_pair_binding.validate(
                evidence, lambda part: canonical_hash({k: v for k, v in part.items() if k != 'sha256'}),
            )
            image_embedded_pair_binding.validate_scope(evidence, scope is not None)
            image_base = image_embedded_pair_binding.base_document(evidence)
        else:
            _require(not {'embedded_pair_difference', 'prior_embedded_pair_evidence'}.intersection(evidence),
                     'Historical image evidence contains a newer comparison')
        if image_base['version'] in AFFINE_VERSIONS or 'copy_affine' in image_base:
            from app.evidence.image_copy_affine_binding import validate as validate_affine

            validate_affine(image_base, box_bounds=box_bounds, url=copy_asset_url, credential_status=SKIPPED_CREDENTIALS)
        validate_similarity(image_base, canonical_hash)
        validate_copy_scope(image_base, scope)
        validate_inspection_coverage({**image_base, 'version': 'image-evidence-3'}
                                     if image_base['version'] in {'image-evidence-4', 'image-evidence-5', 'image-evidence-6'} | REFLECTION_VERSIONS else image_base)
        if image_base['version'] in {'image-evidence-5', 'image-evidence-6'} | REFLECTION_VERSIONS:
            for packet in image_base['packets']:
                cue = packet.get('copy_corroboration')
                assets = packet['artifacts']
                if packet.get('evidence_type') == 'localisation':
                    _require(isinstance(cue, dict) and cue.get('schema') == 'same-image-copy-corroboration/1'
                             and cue.get('authorship_confidence_contribution') == 0
                             and cue.get('highlight_asset_sha256') == assets.get('mask', {}).get('sha256')
                             and (cue.get('status') == 'copy_consistent') == ('copy_donor' in assets),
                             'Copy evidence differs from its committed highlight or matching patch')
                    if image_base['version'] in {'image-evidence-6'} | REFLECTION_VERSIONS:
                        _require('search_strategy' in cue and cue['search_strategy'] in {None, 'native_dense', 'native_anchor_shortlist'}
                                 and (cue['search_strategy'] is None) == (cue.get('status') == 'unsupported_geometry')
                                 and (cue.get('status') != 'unassessed_candidate_budget' or cue['search_strategy'] == 'native_anchor_shortlist'),
                                 'Copy evidence search coverage is inconsistent')
                else:
                    _require(cue is None and 'copy_donor' not in assets, 'Copy evidence is missing its component mask')
    _require(analysis['kind'] == 'document' or email_alternatives.COMMITMENT not in result,
             'Email body rendition commitments require document evidence')
    if analysis['kind'] == 'document':
        from app.evidence.document_matched_context import validate as validate_matched_context
        from app.evidence.document_paragraph_priority import validate as validate_paragraph_priority
        from app.evidence.document_repeat import validate as validate_repeated_passages
        from app.evidence.document_unselected_responses import validate as validate_unselected_responses

        validate_repeated_passages(evidence)
        validate_matched_context(evidence, published_checks={check['id'] for check in result.get('checks', [])})
        validate_paragraph_priority(evidence)
        document_scope.validate_passage_coverage(evidence)
        document_scope.validate_checkpoint_agreement(evidence)
        validate_unselected_responses(evidence, published_checks={check.get('display_id') for check in result.get('checks', [])})
        email_alternatives.validate_set(evidence, result.get(email_alternatives.COMMITMENT))
    if analysis['kind'] == 'document' and evidence['version'] in document_scope.CONTEXT_VERSIONS:
        from app.evidence.document_context import validate as validate_context_comparison

        checks = {check['id'] for check in result.get('checks', [])}
        for packet in evidence['packets']:
            if evidence['version'] in document_scope.SCOPE_VERSIONS:
                document_scope.validate(packet, evidence['source']['chars'])
            validate_context_comparison(packet['context_comparison'], focus=packet['focus'], chars=evidence['source']['chars'])
            _require(all(row['signal_ref']['id'] in checks for row in packet['context_comparison']['checks']),
                     'A context comparison refers to an unpublished classifier check')
            if isolated := packet['context_comparison'].get('isolated_paragraph'):
                _require(isolated['signal_ref']['id'] in checks,
                         'An isolated paragraph comparison refers to an unpublished classifier check')

    refs = {}
    processed_pcm = {}
    parts = [evidence]
    if evidence['version'] in NESTED_VERSIONS:
        parts.extend(evidence[key] for key in ('audio', 'av'))
    if analysis['kind'] == 'audio':
        from app.evidence.audio_binding import validate_audio_stream_scope, validate_unscored_voice_coverage

        validate_audio_stream_scope(evidence["source"])
        validate_unscored_voice_coverage(evidence['coverage'], evidence['source'], evidence.get('stereo_context'))
        acoustic = evidence.get('acoustic_context')
        if evidence['version'] in {'audio-evidence-2', 'audio-evidence-3'}:
            _require(bool(acoustic), 'Audio acoustic context is missing')
        if evidence['version'] == 'audio-evidence-3':
            _require(acoustic['schema'] == 'chipmunk.acoustic-context/2', 'Audio continuity measurements are missing')
        if acoustic is not None:
            from app.evidence.audio_background_binding import validate_acoustic_context

            validate_acoustic_context(acoustic, source=evidence['source'], voice_packets=evidence['packets'])
    groups = [group for part in parts for group in
              [part.get('overview') or {}, *(p.get('artifacts') or {} for p in part.get('packets', []))]]
    thumbnail_assets = set()
    if IMAGE_EDITOR_PREVIEW in result:
        from app.evidence import image_editor_preview_binding

        reports = bundle.get('reports')
        _require(analysis['kind'] == 'image' and (result.get('decision') or {}).get('image_copy_coverage') is None
                 and isinstance(reports, list) and len(reports) == 1
                 and reports[0].get('profile') in image_editor_preview_binding.REPORT_PROFILES,
                 'Editor thumbnail evidence is outside an eligible image report')
        thumbnail_groups = list(image_editor_preview_binding.asset_groups(result[IMAGE_EDITOR_PREVIEW], evidence['source']))
        thumbnail_assets = {asset['sha256'] for group in thumbnail_groups for asset in group.values()}
        groups.extend(thumbnail_groups)
    repeat_pcm = {}
    for part in parts:
        copy_base = part
        if part.get('version') == IMAGE_PAIR_VERSION:
            from app.evidence import image_embedded_pair_binding

            copy_base = image_embedded_pair_binding.base_document(part)
            groups.append(image_embedded_pair_binding.artifacts(part))
        groups.extend(pair['artifacts'] for record in copy_records(copy_base) for pair in record['pairs'])
        acoustic = part.get('acoustic_context')
        if acoustic:
            groups.extend([acoustic.get('overview') or {}, *(event.get('artifacts') or {} for event in acoustic.get('events', []))])
        sound = part.get('sound_events')
        if sound is not None:
            from app.evidence.sound_event_binding import validate as validate_sound_events
            validate_sound_events(sound, part['source'])
            groups.extend(window.get('artifacts') or {} for window in sound['windows'])
        if 'short_voice_review' in part:
            from app.evidence.audio_short_review_binding import validate as validate_short_voice_review

            _require(analysis['kind'] == 'audio' and part is evidence, 'Short speech-window review is outside standalone audio')
            review = part['short_voice_review']
            validate_short_voice_review(review, part['source'])
            groups.append(review['artifacts'])
        if 'long_voice_review' in part:
            from app.evidence.audio_long_review_binding import media_windows
            from app.evidence.audio_long_review_binding import validate as validate_long_voice_review

            _require(analysis['kind'] == 'audio' and part is evidence, 'Long speech-window review is outside standalone audio')
            review = part['long_voice_review']
            validate_long_voice_review(review, part['source'])
            groups.extend(window['artifacts'] for window in media_windows(review))
        if 'background_reduced_voice_review' in part:
            from app.evidence.audio_background_review_binding import validate as validate_background_review

            _require(analysis['kind'] == 'audio' and part is evidence, 'Background-reduced speech review is outside standalone audio')
            review = part['background_reduced_voice_review']
            validate_background_review(review, part['source'])
            groups.append(review['artifacts'])
            if review['status'] == 'flagged':
                digest = review['artifacts']['processed_clip']['sha256']
                processed_pcm[digest] = review['processed_pcm_sha256']
        if 'stereo_listening' in part:
            from app.evidence.audio_binding import _timeline_ranges
            from app.evidence.stereo_binding import validate as validate_stereo
            from app.evidence.stereo_listening_binding import validate as validate_listening

            _require((analysis['kind'] == 'audio' and part is evidence) or part is evidence.get('audio'),
                     'Stereo listening is outside the audio evidence')
            stereo = part['stereo_context']
            validate_stereo(stereo, part['source'], ranges=lambda a, b, src=part['source']: _timeline_ranges(src, a, b))
            validate_listening(part['stereo_listening'], stereo)
            groups.extend(window['artifacts'] for window in part['stereo_listening']['windows'])
        if (repeat := part.get('repetition_context')) is not None:
            from app.evidence.audio_repetition_binding import validate as validate_repetition

            validate_repetition(repeat, part['source'])
            for pair in repeat['pairs']:
                groups.append(pair['artifacts'])
                digest = pair['artifacts']['first_clip']['sha256']
                pcm_digest = pair['pcm_s16le_sha256']
                _require(repeat_pcm.setdefault(digest, pcm_digest) == pcm_digest,
                         'One repeated-audio clip has conflicting PCM commitments')
        if (similarity := part.get('similarity_context')) is not None:
            from app.evidence.audio_similarity_binding import validate as validate_similarity_context

            _require((analysis['kind'] == 'audio' and part is evidence) or part is evidence.get('audio'),
                     'Audio similarity is outside the audio evidence')
            validate_similarity_context(similarity, part['source'], part.get('repetition_context'))
            for pair in similarity['pairs']:
                groups.append(pair['artifacts'])
                for side in ('first', 'second'):
                    digest = pair['artifacts'][side + '_clip']['sha256']
                    pcm_digest = pair[side]['pcm_s16le_sha256']
                    _require(repeat_pcm.setdefault(digest, pcm_digest) == pcm_digest,
                             'One repeated-audio clip has conflicting PCM commitments')
    for group in groups:
        _require(isinstance(group, dict), 'Invalid evidence assets')
        for asset in group.values():
            _require(isinstance(asset, dict) and asset.get('media_type') in TYPES
                     and bool(_SHA.fullmatch(str(asset.get('sha256', '')))), 'Invalid evidence asset reference')
            digest = asset['sha256']
            intrinsic = {k: asset.get(k) for k in ('media_type', 'width', 'height', 'samples', 'sample_rate', 'channels',
                                                 'frame_count', 'time_base', 'pts', 'durations', 'duration_ticks',
                                                 'duration_s', 'codec', 'has_audio')}
            _require(digest not in refs or refs[digest][1] == intrinsic, 'One evidence object has conflicting metadata')
            refs[digest] = (asset, intrinsic)
    entries = bundle.get('artifacts')
    _require(isinstance(entries, list), 'The artifact index is missing')
    indexed = {}
    for entry in entries:
        _require(isinstance(entry, dict) and entry.get('sha256') not in indexed, 'Duplicate artifact index entry')
        indexed[entry.get('sha256')] = entry
    _require(set(indexed) == set(refs), 'Artifact index does not exactly cover the frozen evidence set')
    for digest, (asset, _) in refs.items():
        entry = indexed[digest]
        path = f"assets/{digest}.{TYPES[asset['media_type']]}"
        _require(entry.get('file') == path and path in names and entry.get('media_type') == asset['media_type'],
                 'An artifact path or media type differs from its frozen reference')
        _require(entry.get('bytes') == archive.getinfo(path).file_size, 'An artifact size differs from its index')
        if digest in thumbnail_assets:
            _require(0 < entry['bytes'] <= image_editor_preview_binding.MAX_ASSET_BYTES, 'A thumbnail crop exceeds its size limit')
        recorded = hashlib.sha256()
        with archive.open(path) as stream:
            while chunk := stream.read(1024 * 1024):
                recorded.update(chunk)
        _require(recorded.hexdigest() == digest, 'An artifact differs from its frozen evidence hash')
        with archive.open(path) as stream:
            if asset['media_type'] == 'image/png':
                header = stream.read(24)
                _require(header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR', 'A PNG asset has an invalid header')
                width, height = struct.unpack('>II', header[16:24])
                _require(width == asset.get('width') and height == asset.get('height'), 'A PNG asset has different dimensions')
            elif asset['media_type'] == 'audio/wav':
                with wave.open(stream, 'rb') as audio:
                    _require(audio.getnframes() == asset.get('samples') and audio.getframerate() == asset.get('sample_rate')
                             and audio.getnchannels() == asset.get('channels') and audio.getsampwidth() == 2,
                             'A WAV asset does not match its frozen sample dimensions')
                    if digest in repeat_pcm or digest in processed_pcm:
                        pcm_digest = hashlib.sha256(audio.readframes(audio.getnframes())).hexdigest()
                    if digest in repeat_pcm:
                        _require(pcm_digest == repeat_pcm[digest],
                                 'A repeated-audio clip does not contain its committed samples')
                    if digest in processed_pcm:
                        _require(pcm_digest == processed_pcm[digest], 'The processed audio clip does not contain its committed samples')
            else:
                dimensions = inspect_mp4(stream)
                _require(all(canonical_hash(asset.get(k)) == canonical_hash(v) for k, v in dimensions.items()),
                         'An MP4 asset does not match its frozen frames, timing or track format')
    expected = {'README.txt', 'reproduce.md', 'bundle.json', 'evidence.json', 'result.json', 'report.pdf',
                'instance_public_key.pem', 'manifest.sha256', 'manifest.sha256.sig', *(a['file'] for a in entries)}
    original = bundle.get('source')
    if original is not None:
        _require(isinstance(original, dict) and bool(re.fullmatch(r'source/original\.[a-z0-9]{1,8}', str(original.get('file')))),
                 'Invalid optional original path')
        path = original['file']
        _require(path in names and original.get('sha256') == source['sha256'] and original.get('bytes') == source.get('bytes')
                 and archive.getinfo(path).file_size == source.get('bytes'), 'The optional original identity or size differs')
        digest = hashlib.sha256()
        with archive.open(path) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        _require(digest.hexdigest() == source['sha256'], 'The optional original does not match its source commitment')
        expected.add(path)
    _require(names == expected, 'Unexpected files or missing declared files in the public derivative')
    reports = bundle.get('reports')
    _require(isinstance(reports, list) and len(reports) == 1, 'Exactly one frozen report must be included')
    report = reports[0]
    _require(bundle.get('signing') == {'algorithm': 'Ed25519', 'key_id': report.get('signing_key_id'),
                                      'public_key_file': 'instance_public_key.pem'}, 'The signing declaration differs from the report signer')
    _require(report.get('file') == 'report.pdf' and report.get('analysis_id') == analysis['id']
             and report.get('evidence_set_sha256') == actual
             and bool(_SHA.fullmatch(str(report.get('snapshot_sha256', '')))), 'The report identity does not bind to this evidence')
    _require(bundle.get('kind') in ('draft', 'issue') and report.get('issued') == (bundle['kind'] == 'issue'),
             'The report and package draft/issue states disagree')

    custody = bundle.get('custody')
    _require(isinstance(custody, dict) and custody.get('schema') == 'chipmunk.custody-commitments/1'
             and custody.get('verification') == 'commitments_only' and custody.get('complete_event_chain_included') is False,
             'Public custody must be labelled as commitments only, not a verified original chain')
    events, anchor = custody['events'], custody['anchor']
    _require(isinstance(events, list) and isinstance(anchor, dict) and isinstance(anchor.get('head_seq'), int)
             and bool(_SHA.fullmatch(str(anchor.get('head_hash', '')))), 'Invalid custody commitment anchor')
    previous = 0
    for event in events:
        _require(isinstance(event, dict) and set(event) == {'seq', 'event_hash'} and isinstance(event['seq'], int)
                 and previous < event['seq'] <= anchor['head_seq'] and bool(_SHA.fullmatch(str(event['event_hash']))),
                 'Invalid or unordered custody commitment')
        if event['seq'] == anchor['head_seq']:
            _require(event['event_hash'] == anchor['head_hash'], 'The final custody commitment differs from its anchor')
        previous = event['seq']
    return [('public evidence', True, f'{len(refs)} declared assets; source, result, version and report commitments agree'),
            ('custody commitments', True, f'{len(events)} ordered commitments; original event bodies and chain completeness are NOT verified'),
            ('verification scope', True, 'Signed public derivative integrity only; no inference reproduction or AI-authorship proof')]


def checks(archive, names: set[str], bundle: dict) -> list[tuple[str, bool, str]]:
    try:
        return _checks(archive, names, bundle)
    except (KeyError, TypeError, ValueError, AttributeError, struct.error, wave.Error) as error:
        return [('public evidence', False, str(error) or 'Malformed public evidence derivative')]
