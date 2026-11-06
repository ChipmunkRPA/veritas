"""Customer-facing views of detector data, without modifying stored evidence."""
from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import PurePath
from typing import Any

from app.core.config import settings
from app.engine.registry import get_registry


@lru_cache(maxsize=1)
def _identities() -> tuple[dict[str, str], tuple[str, ...]]:
    registry = get_registry()
    ids: dict[str, str] = {}
    names: set[str] = set()
    for detector in registry.manifest.detectors:
        if not detector.models:
            continue
        ids[detector.id] = 'check.' + hashlib.sha256(detector.id.encode()).hexdigest()[:16]
        name = detector.name.strip()
        if name:
            names.add(name)
            names.add(re.sub(r'\s*\([^()]*\)\s*$', '', name).strip())
    for model in registry.manifest.models:
        ids[model.id] = 'component.' + hashlib.sha256(model.id.encode()).hexdigest()[:16]
    legacy_ids = {'univfd_clip', 'clf_dfdc_effnet_b7', 'classifier_document', 'classifier_secondary', 'binoculars_document'}
    for raw in settings.image_classifiers:
        repo = raw.split('|', 1)[0].strip()
        if repo:
            names.add(repo)
            legacy_ids.add('clf_' + repo.rsplit('/', 1)[-1])
    for legacy_id in legacy_ids:
        ids[legacy_id] = 'check.' + hashlib.sha256(legacy_id.encode()).hexdigest()[:16]
    names.update({'DFDC EfficientNet-B7', 'Community Forensics', 'DDA', 'DINOv2 ViT-L/14 with LoRA',
                  'Dual Data Alignment', 'Ateeqq ai-vs-human-image-detector', 'dima806 deepfake_vs_real_image_detection',
                  'desklib-ft-v2', 'Binoculars', 'YuNet face detector', 'DINOv2-small', 'Gustking wav2vec2-XLSR',
                  'DFDC', 'YuNet', 'DINOv2', 'DeBERTa-v3-large AI-text classifier (document chunks)',
                  'DeBERTa-v3-large', 'DeBERTa', 'Qwen2.5-1.5B pair', 'Qwen2.5-1.5B', 'Veritas fine-tune',
                  'Qwen2.5-3B', 'ChatGPT-2022',
                  'RoBERTa ChatGPT detector (HC3)', 'CLIP feature-space probe', 'FUSED', 'OpenSDI',
                  'UnivFD — CLIP ViT-L/14 universal fake detector', 'DFDC-winner EfficientNet-B7 face-swap classifier'})
    return ids, tuple(sorted((name for name in names if len(name) >= 3), key=len, reverse=True))


_HISTORICAL_DETECTOR = re.compile(
    r'(?<![\w.])(?:img|vid|aud|txt|loc|face|prov)\.[A-Za-z0-9_][A-Za-z0-9_.-]*(?![\w.])'
    r'|(?<![\w.])(?:clf_|classifier_|binoculars_)[A-Za-z0-9_.-]+(?![\w.])')

# Generated references can name an architecture without using a registered
# checkpoint name. Source quotations never enter this replacement boundary.
_ARCHITECTURE_LABEL = re.compile(
    r'(?<![\w.])(?:ViT(?:[-_/][A-Za-z0-9]+)*|VisionTransformer|ViTForImageClassification|'
    r'ConvNeXt(?:[-_][A-Za-z0-9]+)*|SegFormer|EfficientNet(?:[-_][A-Za-z0-9]+)*|'
    r'ResNet\d*|Swin(?:[-_][A-Za-z0-9]+)*|Wav2Vec2?(?:[-_][A-Za-z0-9]+)*|'
    r'XLS[-_]?R|HuBERT|Xception|VideoMAE|MiT[-_][A-Za-z0-9]+|LoRA|VAE|'
    r'(?-i:CLIP)|open_clip)(?![\w.])', re.I)
_FUSION_ID = re.compile(r'(?<![\w.])fuser-(?:image|video|audio|document|text)[A-Za-z0-9_.-]*[A-Za-z0-9](?![\w])', re.I)


def _registered_detectors() -> frozenset[str]:
    return frozenset(detector.id for detector in get_registry().manifest.detectors)


def _check_alias(root: str) -> str:
    return 'check.' + hashlib.sha256(root.encode()).hexdigest()[:16]


def _candidate_aliases(candidates: tuple[str, ...]) -> dict[str, str]:
    ids = dict(_identities()[0])
    registered = _registered_detectors()
    for candidate in candidates:
        root = candidate.split(':', 1)[0]
        if root and root not in registered and root not in ids and not root.startswith(('check.', 'component.')):
            ids[root] = _check_alias(root)
    return ids


@lru_cache(maxsize=128)
def _alias_pattern(identities: tuple[str, ...]) -> re.Pattern[str]:
    # Cache only a pure compiled expression. Candidate mappings depend on the
    # current identity table and registry, so caching by candidates alone could
    # keep retired mappings alive or make references disagree within one result.
    return re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(value) for value in identities) + r')(?!\w)')


def alias_id(value: str, *, candidates: Iterable[str] = ()) -> str:
    """Opaque stable detector roots, retaining scope suffixes and existing aliases.

    Historical namespace IDs remain private even after removal from the current
    registry. Explicit candidates cover older adapters without a namespace. They
    must come from machine records, never arbitrary user text.
    """
    roots = [match.group() for match in _HISTORICAL_DETECTOR.finditer(value)]
    choices = tuple(sorted(set(candidates) | set(roots)))
    ids = _candidate_aliases(choices)
    ids.update({match.group(): 'component.' + hashlib.sha256(match.group().encode()).hexdigest()[:16]
                for match in _FUSION_ID.finditer(value)})
    if not ids:
        return value
    return _alias_pattern(tuple(sorted(ids, key=len, reverse=True))).sub(lambda match: ids[match.group()], value)


def alias_signal(value: str) -> str:
    """Signal/check references are machine identities even without a namespace."""
    candidate = value.removeprefix('detector:')
    return alias_id(value, candidates=(candidate,))


def _model_alias(value: str) -> str:
    existing = alias_id(value)
    return existing if existing != value or value.startswith(('check.', 'component.')) else (
        'component.' + hashlib.sha256(value.encode()).hexdigest()[:16])


def original_id(value: str, *, candidates: Iterable[str] = ()) -> str:
    """Resolve a digest against authorized record candidates; never decode a name.

    The current registry retains backwards compatibility for known aliases. A
    removed adapter is resolvable only through the requested analysis/collection.
    """
    choices = tuple(candidates)
    matches = {candidate for candidate in choices if alias_id(candidate, candidates=choices) == value}
    if len(matches) == 1:
        return matches.pop()
    if len(matches) > 1:
        raise ValueError('Ambiguous public identity')
    for original, alias in _identities()[0].items():
        value = value.replace(alias, original)
    return value


def artifact_alias(name: str) -> str:
    """An opaque alias for an application-generated v1 artifact filename."""
    return 'artifact-' + hashlib.sha256(name.encode()).hexdigest()[:20] + PurePath(name).suffix


def public_cursor(cursor: str | None, *, candidates: Iterable[str] = ()) -> str | None:
    if not cursor:
        return cursor
    try:
        values = json.loads(base64.urlsafe_b64decode(cursor + '=' * (-len(cursor) % 4)))
        if not isinstance(values, list):
            return cursor
        data = [alias_id(v, candidates=candidates) if isinstance(v, str) else v for v in values]
        return base64.urlsafe_b64encode(json.dumps(data, separators=(',', ':')).encode()).decode().rstrip('=')
    except (ValueError, TypeError, UnicodeError):
        return cursor


def neutral_text(value: str, *, candidates: Iterable[str] = (), names: Iterable[str] = ()) -> str:
    # This exact sentence was generated by the historical voice calibration.
    # Keep its uncertainty meaning without publishing a service threshold.
    value = value.replace(
        'The calibrated voice estimate and uncertainty check do not jointly support an 80% tendency.',
        'The calibrated voice estimate and uncertainty check do not jointly support a reliable tendency.')
    ids = _candidate_aliases(tuple(sorted(set(candidates) | {match.group() for match in _HISTORICAL_DETECTOR.finditer(value)})))
    names = sorted(set(_identities()[1]) | {name for name in names if len(name) >= 3}, key=len, reverse=True)
    for original in sorted(ids, key=len, reverse=True):
        value = re.sub(r'(?<![\w.])' + re.escape(original) + r'(?![\w.])', 'analysis check', value)
    for name in names:
        value = re.sub(r'(?<![\w])' + re.escape(name) + r'(?![\w])', 'a detection check', value)
    value = _ARCHITECTURE_LABEL.sub('a detection check', value)
    return _FUSION_ID.sub('combined assessment', value)


def localisation_reference(detector_id: str | None) -> str | None:
    """Customer meaning of supported local checks, independent of private architectures.

    These adapters write their own observation and reference; neither is a
    transcription of the source. Do not apply this rule to arbitrary detectors
    or provenance records whose observed values can be user evidence.
    """
    if not isinstance(detector_id, str) or detector_id.split(':', 1)[0] not in {'loc.mesorch', 'loc.radar', 'loc.fused', 'loc.splice'}:
        return None
    return ('An uncalibrated local editing response. The operating threshold and measured regions are recorded '
            'with this check. A flagged region is an inspection candidate, not proof of AI editing.')


def _unquoted_neutral_text(value: str, *, candidates: Iterable[str] = (), names: Iterable[str] = ()) -> str:
    # Keep any quoted source value byte-for-byte, as report presentation does.
    pieces = re.split(r'("[^"]*"|“[^”]*”|(?<!\w)\'[^\']*\'(?!\w))', value)
    for index in range(0, len(pieces), 2):
        pieces[index] = neutral_text(pieces[index], candidates=candidates, names=names)
    return ''.join(pieces)


_ID_FIELDS = frozenset({'id', 'detector_id', 'created_by', 'source_signal_ids', 'evidence_ids', 'corroborated_by',
                        'detector_ids', 'signal_id', 'signal_ids', 'calibration_ref', 'superseded_by', 'v1_id', 'subject', 'revision',
                        'fuser_id', 'fuser_version', 'fusion_id', 'calibration_id'})
_SOURCE_FIELDS = frozenset({'filename', 'original_filename', 'observed', 'metadata', 'provenance', 'provenance_record',
                            'stored_record', 'input', 'annotation', 'annotations', 'examiner', 'review', 'notes',
                            'note', 'intake_notes', 'supplied_by', 'source', 'source_data', 'content', 'text_original'})
_TEXT_FIELDS = frozenset({'name', 'title', 'statement', 'reasoning', 'reference', 'caption', 'reason', 'status_reason',
                          'error', 'benchmark_note', 'dissent_note', 'description', 'disclosures', 'fp_modes', 'reference_range',
                          'coverage_date', 'reasons', 'caveats'})
_SOURCE_FAMILIES = frozenset({'F1_provenance', 'F2_self_declared_metadata', 'F8_metadata_consistency'})
_MODEL_DETAIL_FIELDS = frozenset({'model', 'model_name', 'model_id', 'model_path', 'model_revision', 'architecture',
                                  'repository', 'repo', 'observer', 'performer', 'weights'})
_SIGNAL_FIELDS = frozenset({'detector_id', 'detector_ids', 'source_signal_ids', 'evidence_ids', 'corroborated_by',
                            'signal_id', 'signal_ids', 'detector', 'intentionally_missing'})
_SENTENCE_SCORES = {'classifier': 'classification_response', 'classifier_secondary': 'classification_response_secondary',
                    'binoculars': 'pattern_ratio', 'binoculars_p': 'pattern_response'}


def _public_ocr(value: Any) -> Any:
    """Keep source words/coordinates/recognition quality; omit OCR installation identities."""
    if isinstance(value, list):
        return [_public_ocr(item) for item in value]
    if isinstance(value, dict):
        return {key: _public_ocr(item) for key, item in value.items()
                if key not in {'engine_version', 'language_data_sha256', 'languages_requested'}}
    return value


def _historical_identities(value: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read names from machine identity records, without inspecting source material."""
    candidates, names = set(), set()

    def visit(item: Any, field: str = '') -> None:
        if field in _SOURCE_FIELDS:
            return
        if isinstance(item, list):
            for child in item:
                visit(child, field)
        elif isinstance(item, dict):
            if item.get('family') in _SOURCE_FAMILIES or item.get('category') == 'provenance':
                return
            identity = item.get('detector_id')
            if not identity and (field == 'signals' or item.get('display_id') or item.get('category') in {'model', 'face'}):
                identity = item.get('id')
            if isinstance(identity, str):
                root = identity.split(':', 1)[0]
                candidates.add(root)
                if root not in _registered_detectors() and not root.startswith(('check.', 'component.')):
                    name = item.get('name')
                    if isinstance(name, str) and len(name.strip()) >= 3:
                        names.add(name.strip())
                    for key in ('model', 'model_name', 'architecture', 'repository', 'repo'):
                        detail = (item.get('details') or {}).get(key) if isinstance(item.get('details'), dict) else None
                        if isinstance(detail, str) and len(detail.strip()) >= 3:
                            names.add(detail.strip())
            for key, child in item.items():
                visit(child, key)

    visit(value)
    return tuple(sorted(candidates)), tuple(sorted(names))


def project(value: Any, *, field: str = '', source_text: bool = False) -> Any:
    """Project machine data; never edit user material or the stored result.

    This is not an anonymization boundary for arbitrary attachments or signed packages.
    """
    candidates, names = _historical_identities(value)
    return _project(value, field=field, source_text=source_text, candidates=candidates, names=names)


def _project(value: Any, *, field: str, source_text: bool, candidates: tuple[str, ...], names: tuple[str, ...]) -> Any:
    def walk(item: Any, *, field: str = '', source_text: bool = False) -> Any:
        return _project(item, field=field, source_text=source_text, candidates=candidates, names=names)

    if field in _SOURCE_FIELDS:
        return value
    if isinstance(value, list):
        return [walk(item, field=field, source_text=source_text) for item in value]
    if isinstance(value, dict):
        is_source = source_text or value.get('family') in _SOURCE_FAMILIES or value.get('category') == 'provenance'
        generated_coverage = value.get('family') == 'coverage'
        local_reference = None if is_source else localisation_reference(value.get('detector_id') or value.get('id'))
        identity = value.get('detector_id') or (value.get('id') if field == 'signals' or value.get('display_id') else None)
        historical_check = (not is_source and isinstance(identity, str)
                            and identity.split(':', 1)[0] not in _registered_detectors())
        legacy_check = (not is_source and isinstance(value.get('id'), str)
                        and (value['id'] in {'univfd_clip', 'clf_dfdc_effnet_b7', 'classifier_document',
                                            'classifier_secondary', 'binoculars_document', 'frames_general'}
                             or value['id'].startswith('clf_')))
        out = {}
        for key, item in value.items():
            if generated_coverage and key in {'title', 'statement', 'sentences'}:
                def redact_generated(part):
                    if isinstance(part, str):
                        return neutral_text(part, candidates=candidates, names=names)
                    if isinstance(part, list):
                        return [redact_generated(p) for p in part]
                    if isinstance(part, dict):
                        return {k: redact_generated(v) for k, v in part.items()}
                    return part
                out[key] = redact_generated(walk(item, field=key))
                continue
            if field in {'libraries', 'library_versions'} and not is_source and _ARCHITECTURE_LABEL.fullmatch(key):
                out['runtime.' + hashlib.sha256(key.encode()).hexdigest()[:16]] = item
                continue
            if key == 'ocr' and isinstance(item, dict):
                out[key] = _public_ocr(item)
                continue
            if field == 'stats' and key == 'binoculars_document':
                out['document_pattern_ratio'] = item
                continue
            if local_reference is not None and key == 'reference':
                out[key] = local_reference
                continue
            if (not is_source and key == 'observed' and isinstance(item, str)
                    and value.get('category') == 'model' and item.startswith('fused p(synthetic)=')
                    and 'per model:' in item):
                # The speech fuser writes this sentence, including legacy video
                # signals whose IDs are absent from today's registry. Handle it
                # before generic historical prose, which cannot know retired
                # checkpoint basenames. Source quotations remain untouched.
                prefix, _, values = item.partition('per model:')
                def redact_score(match: re.Match[str]) -> str:
                    name = match.group(1).strip()
                    return ' component.' + hashlib.sha256(name.encode()).hexdigest()[:16] + '='
                out[key] = prefix + 'per check:' + re.sub(r'\s*([^,=]+)=', redact_score, values)
                continue
            if (local_reference is not None or historical_check) and key == 'observed' and isinstance(item, str):
                out[key] = _unquoted_neutral_text(item, candidates=candidates, names=names)
                continue
            if field == 'details' and key in _MODEL_DETAIL_FIELDS and not is_source:
                continue
            if field == 'details' and key in {'per_model_windows', 'per_model_chunks'} and isinstance(item, dict) and not is_source:
                out[key] = {'component.' + hashlib.sha256(str(k).encode()).hexdigest()[:16]: walk(v) for k, v in item.items()}
                continue
            if field == 'details' and key == 'strongest' and 'per_model_chunks' in value and isinstance(item, str) and not is_source:
                out[key] = 'component.' + hashlib.sha256(item.encode()).hexdigest()[:16]
                continue
            if key == 'next_cursor' and isinstance(item, str):
                out[key] = public_cursor(item, candidates=candidates)
            elif key == 'name' and legacy_check:
                out[key] = 'Analysis check'
            elif key == 'artifacts' and isinstance(item, list):
                out[key] = [artifact_alias(part) if isinstance(part, str) else walk(part) for part in item]
            elif key == 'scores' and field == 'faces' and isinstance(item, dict):
                out[key] = {'check.' + hashlib.sha256(str(k).encode()).hexdigest()[:16]: v for k, v in item.items()}
            elif key == 'scores' and field in {'finding_sentence', 'source_sentence'} and isinstance(item, dict):
                out[key] = {_SENTENCE_SCORES.get(str(k), str(k) if k in _SENTENCE_SCORES.values() else alias_signal(str(k))): v
                            for k, v in item.items()}
            elif key in {'model_revisions', 'scores'}:
                out[key] = {(_model_alias(str(k)) if key == 'model_revisions' else alias_signal(str(k))): v
                            for k, v in item.items()} if isinstance(item, dict) else item
            elif key == 'sentences' and isinstance(item, list):
                out[key] = [walk(sentence, field=('source_sentence' if isinstance(sentence, dict)
                              and isinstance(sentence.get('start'), int) and isinstance(sentence.get('end'), int)
                              else 'finding_sentence'), source_text=is_source) for sentence in item]
            elif key == 'contributions' and isinstance(item, list):
                out[key] = [walk(contribution, field='contribution', source_text=is_source) for contribution in item]
            elif field == 'contribution' and key == 'label' and isinstance(item, str):
                out[key] = item if is_source else _unquoted_neutral_text(item, candidates=candidates, names=names)
            elif field == 'finding_sentence' and key == 'text':
                out[key] = item if is_source or not isinstance(item, str) else _unquoted_neutral_text(item, candidates=candidates, names=names)
            elif field == 'not_validated' and key in {'label', 'text'} and isinstance(item, str):
                # These are application-authored validation disclosures, never
                # transcribed source passages or user annotations.
                out[key] = item if is_source else _unquoted_neutral_text(item, candidates=candidates, names=names)
            elif field == 'assessments' and key == 'note' and isinstance(item, str):
                # Assessment notes are generated by the fusion engine. User
                # annotations/intake notes remain protected source fields.
                out[key] = neutral_text(item, candidates=candidates, names=names)
            else:
                out[key] = walk(item, field=key, source_text=is_source)
        return out
    if isinstance(value, str):
        if field == 'loaded':
            return 'component.' + hashlib.sha256(value.encode()).hexdigest()[:16]
        if field in _SIGNAL_FIELDS:
            return alias_signal(value)
        if field in {'model_id', 'models', 'depends_on'}:
            return _model_alias(value)
        if field in _ID_FIELDS or field in {'detector', 'model_id', 'models', 'depends_on'}:
            return alias_id(value, candidates=candidates)
        if field in _TEXT_FIELDS and not source_text:
            return _unquoted_neutral_text(value, candidates=candidates, names=names)
    return value



def project_card(card: dict[str, Any]) -> dict[str, Any]:
    """Expose availability and evaluation facts without publishing model-card sources."""
    out = {key: project(card[key], field=key) for key in (
        'id', 'version', 'modality', 'family', 'hypotheses', 'profiles', 'default_enabled', 'superseded_by',
        'legacy_weight', 'eval', 'threat', 'lane', 'validated_slices', 'enabled', 'licence_policy', 'installed', 'loaded',
    ) if key in card}
    out['name'] = 'Analysis check'
    out['models'] = [alias_id(str(item)) for item in card.get('models') or []]
    out['model_entries'] = [
        {'id': alias_id(str(item.get('id', ''))), 'installed': item.get('installed'),
         'loaded': item.get('loaded'), 'loaded_devices': item.get('loaded_devices', []),
         'depends_on': [alias_id(str(dep)) for dep in item.get('depends_on') or []],
         'licence': {'class': (item.get('licence') or {}).get('class')}}
        for item in card.get('model_entries') or []
    ]
    admission = card.get('admission')
    out['admission'] = ({'decision': admission.get('decision'), 'date': admission.get('date')}
                        if isinstance(admission, dict) else None)
    out['citation'] = 'Publication recorded' if card.get('citation') else None
    out['enablement_reasons'] = ['Not enabled by the current policy'] if card.get('enablement_reasons') else []
    out['known_limitations'] = []
    out['records'] = []
    out['admission_record'] = False
    return out
