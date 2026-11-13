"""Report presentation that omits implementation identities without mutating evidence snapshots."""
from __future__ import annotations

import copy
import re
from typing import Any

from app.api.public_projection import localisation_reference, neutral_text, project
from app.core.canonical import canonical_hash
from app.engine.fusion import disclosures
from app.engine.fusion.runtime import FACE_SCOPE_CAVEAT
from app.reports.snapshot import ReportSnapshot, snapshot_hash

_SOURCE_FAMILIES = {'F1_provenance', 'F2_self_declared_metadata', 'F8_metadata_consistency'}
_MODEL_ALIASES = {'DFDC', 'DDA', 'YuNet', 'Community Forensics', 'Binoculars', 'DeBERTa AI-text classifier',
                  'Qwen2.5-1.5B', 'Qwen2.5-1.5B-Instruct', 'desklib-ft-v2', 'DINOv2-small',
                  'DeBERTa-v3-large', 'DeBERTa', 'Qwen2.5-3B', 'ChatGPT-2022', 'Veritas fine-tune',
                  'Mesorch', 'RADAR', 'SPAI', 'UnivFD', 'FUSED', 'OpenSDI'}
_FACE_SCOPE_SUFFIX = FACE_SCOPE_CAVEAT.split('{name}', 1)[1]
_PUBLIC_FACE_SCOPE = ('A face-oriented check examined the whole image because no face was detected. '
                      'Its stand-alone evidence covers AI-generated faces. Its response contributed only through '
                      'the combined estimate, whose weights were fitted on images with and without faces.')


def is_named_face_scope_limitation(value: object) -> bool:
    """Recognize the exact generated face-scope caveat on a stored result."""
    return (isinstance(value, str) and value.endswith(_FACE_SCOPE_SUFFIX)
            and bool(value[:-len(_FACE_SCOPE_SUFFIX)].strip()))


def _identities(snapshot: ReportSnapshot) -> tuple[dict[str, str], list[str]]:
    # Use the historical detector names frozen into this snapshot rather than mutable current registry state.
    result = snapshot.result
    ids = sorted({str(d.get('id')) for d in (result.get('manifest') or {}).get('detectors', []) if d.get('id')}
                 | {str(s['detector_id']) for s in result.get('signals') or [] if s.get('detector_id')}
                 | {str(c['subject']).split('@', 1)[0] for c in result.get('calibration') or []
                    if c.get('subject') and not str(c['subject']).startswith('fuser-')})
    aliases = {value: f'Check {index}' for index, value in enumerate(ids, 1)}
    names = set(ids)
    for key, value in snapshot.detector_names.items():
        if key in aliases and value:
            names.add(value)
    for card in snapshot.model_cards:
        if card.get('id') in aliases and card.get('name'):
            names.add(str(card['name']))
    for signal in result.get('signals') or []:
        if signal.get('detector_id') in aliases and signal.get('name'):
            names.add(str(signal['name']))
    # Fitted detector prose often uses the name before the architecture in parentheses.
    names.update(re.sub(r'\s*\([^()]*\)\s*$', '', n).strip() for n in list(names))
    names.update(_MODEL_ALIASES | {'DINOv2 ViT-L/14 with LoRA', 'Dual Data Alignment'})
    return aliases, sorted((x for x in names if len(x) >= 3), key=len, reverse=True)


def _text(value: Any, names: list[str]) -> Any:
    if not isinstance(value, str):
        return value
    # A generated explanation can contain a verbatim source value. Never rewrite quoted evidence.
    pieces = re.split(r'("[^"]*"|“[^”]*”|(?<!\w)\'[^\']*\'(?!\w))', value)
    for index in range(0, len(pieces), 2):
        pieces[index] = _unquoted_text(pieces[index], names)
    return ''.join(pieces)


def _unquoted_text(value: str, names: list[str]) -> str:
    for name in names:
        pattern = r'(?<![\w])' + re.escape(name) + r'(?![\w])'
        def substitute(match: re.Match[str]) -> str:
            prefix = match.string[:match.start()]
            if re.search(r'\b(?:the|a|an)\s+$', prefix, re.I):
                return 'detection check'
            return 'A detection check' if not prefix or re.search(r'[.!?]\s+$', prefix) else 'a detection check'
        value = re.sub(pattern, substitute, value)
    return neutral_text(value, names=names)


def _configuration_text(value: Any, names: list[str]) -> Any:
    """Derived implementation metadata is not a quotation of the submitted source."""
    if not isinstance(value, str):
        return value
    value = re.sub(r'\bweights? revision\b[^;)]*', 'implementation revision withheld', value, flags=re.I)
    value = re.sub(r'\bupstream code commit\s+[a-f0-9]{7,64}(?:\s+of\s+\d{4}-\d{2}-\d{2})?',
                   'implementation revision withheld', value, flags=re.I)
    value = value.replace('upstream README', 'training documentation')
    return _unquoted_text(value, names)


def _machine(value: Any, names: list[str]) -> Any:
    """Only invoke on detector-authored prose, never user-supplied source/metadata/annotations."""
    if isinstance(value, dict):
        return {k: _machine(v, names) for k, v in value.items()}
    if isinstance(value, list):
        return [_machine(v, names) for v in value]
    if type(value) is tuple:
        return tuple(_machine(v, names) for v in value)
    # Markup includes SVG used in numerical charts; its labels are not detector names.
    if type(value) is str:
        return _text(value, names)
    return value


def public_view(snapshot: ReportSnapshot, original: dict[str, Any]) -> dict[str, Any]:
    """Redact derived presentation only; preserve source quotations and immutable snapshot bytes."""
    out = copy.deepcopy(original)
    aliases, names = _identities(snapshot)
    families = {str(f.get('id')): str(f.get('family', '')) for f in snapshot.result.get('findings') or []}
    # Internal anchors also contain detector IDs. Replace only link destinations, not source content.
    anchors = {}
    for index, signal in enumerate(snapshot.result.get('signals') or [], 1):
        original_anchor = 'sig-' + re.sub(r'[^A-Za-z0-9_-]', '-', str(signal.get('id', '')))
        anchors[original_anchor] = f'sig-check-{index}'
    for index, detector_id in enumerate(sorted(aliases), 1):
        anchors['det-' + re.sub(r'[^A-Za-z0-9_-]', '-', detector_id)] = f'det-check-{index}'

    def remap_links(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'anchor' and isinstance(child, str) and child in anchors:
                    value[key] = anchors[child]
                elif key == 'href' and isinstance(child, str) and child.startswith('#') and child[1:] in anchors:
                    value[key] = '#' + anchors[child[1:]]
                else:
                    remap_links(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                remap_links(child)
    remap_links(out)
    for question in (out.get('scope') or {}).get('questions') or []:
        if 'how' in question:
            question['how'] = _unquoted_text(question['how'], names)
    for row in out.get('findings') or []:
        if families.get(str(row.get('id'))) not in _SOURCE_FAMILIES:
            for field in ('title', 'reason', 'sentences', 'notes', 'thumbnail_caption', 'thumbnail_alt'):
                if field in row:
                    row[field] = _machine(row[field], names)
            if families.get(str(row.get('id'))) == 'coverage':
                # Quotes around skipped-check reasons are generated diagnostics,
                # not quotations of the customer's submitted content.
                def coverage_text(value):
                    if isinstance(value, str):
                        return _unquoted_text(value, names)
                    if isinstance(value, list):
                        return [coverage_text(v) for v in value]
                    if isinstance(value, dict):
                        return {k: coverage_text(v) for k, v in value.items()}
                    return value
                for field in ('title', 'reason', 'sentences', 'notes'):
                    if field in row:
                        row[field] = coverage_text(row[field])
        # References are display IDs / anchors, not the detector's name.
        for ref in row.get('refs') or []:
            if isinstance(ref, dict) and str(ref.get('href', '')).startswith(('#sig-check-', '#det-check-')):
                ref['label'] = _text(ref.get('label'), names)
    summary = out.get('summary') or {}
    # Abstention diagnostics are generated from skipped-check names. They are
    # separate from finding cards and also appear when no statistical signal ran.
    for field in ('insufficient', 'text'):
        if field in summary:
            summary[field] = _machine(summary[field], names)
    if 'reasons' in out:
        out['reasons'] = _machine(out['reasons'], names)
    for row in summary.get('reasons') or []:
        if families.get(str(row.get('id'))) not in _SOURCE_FAMILIES:
            for field in ('title', 'text'):
                row[field] = _text(row.get(field), names)
    # These paragraphs are authored by the disclosure catalogue, including
    # references to checks that need not have run in this particular snapshot.
    # Match that generated prose rather than rewriting arbitrary stored caveats.
    generated_limitations = {item.text for item in disclosures.DISCLOSURES}
    def limitation_text(text: Any) -> Any:
        # lay_summary adds this presentation heading before public projection.
        # Recognize the same generated limitation with or without that heading;
        # arbitrary source caveats still do not enter the prose replacement path.
        content = text.removeprefix('Main limitation: ') if isinstance(text, str) else text
        if isinstance(content, str) and is_named_face_scope_limitation(content):
            prefix = text[:len(text) - len(content)]
            return prefix + _PUBLIC_FACE_SCOPE
        if isinstance(content, str) and (content in generated_limitations or content.startswith((
                'Text classifier that scored this document:', 'allowlisted component with unreviewed training-data terms:',
                'A check that this analysis depends on did not run:', 'Requested full profile reduced:',
                'Requested quick profile reduced:', 'Requested deep profile reduced:'))
                or re.match(r'^\d+ checks? could not run on this server:', content)):
            return _text(text, names)
        return text
    if 'limitation' in summary:
        summary['limitation'] = limitation_text(summary['limitation'])
    for row in out.get('limitations') or []:
        row['text'] = limitation_text(row.get('text'))
    if isinstance(out.get('not_validated'), str):
        # The summary profile builds this line only from the catalogue's labels.
        out['not_validated'] = _text(out['not_validated'], names)
    if isinstance(out.get('estimate'), dict):
        estimate = out['estimate']
        # The contribution narrative is generated from the frozen decision's
        # check labels (including checks with a fixed 50 baseline). It is not
        # source text. Apply the same name projection as its adjacent bars.
        if 'narrative' in estimate:
            estimate['narrative'] = _machine(estimate['narrative'], names)
        for bar in estimate.get('bars') or []:
            if bar.get('kind') in {'detector', 'missing'}:
                bar['label'] = 'Detection check'
        for row in estimate.get('calibration') or []:
            subject = str(row.get('subject') or '')
            detector_id = subject.split('@', 1)[0]
            if detector_id in aliases:
                row['subject'] = aliases[detector_id]
            elif subject.startswith('fuser-'):
                row['subject'] = 'Combined assessment'
        for field in ('statement', 'none', 'fallback'):
            if field in estimate:
                estimate[field] = _machine(estimate[field], names)
    source_ids = {str(s.get('display_id')) for s in snapshot.result.get('signals') or []
                  if str(s.get('detector_id', '')).startswith(('prov.', 'img.meta.', 'vid.meta.'))}
    signal_categories = {str(s.get('display_id')): s.get('category') for s in snapshot.result.get('signals') or []}
    local_references = {str(s.get('display_id')): reference for s in snapshot.result.get('signals') or []
                        if s.get('display_id') and s.get('category') != 'provenance'
                        and s.get('family') not in _SOURCE_FAMILIES
                        and (reference := localisation_reference(s.get('detector_id'))) is not None}
    for row in (out.get('detectors') or {}).get('ledger') or []:
        row['name'] = f"Analysis check {row.get('id') or ''}".strip()
        local_reference = local_references.get(str(row.get('id')))
        if local_reference is not None:
            row['reference'] = local_reference
            row['observed'] = _text(row.get('observed'), names)
        # Other observed values can quote source evidence and stay unchanged.
        elif str(row.get('id')) not in source_ids:
            row['reference'] = _text(row.get('reference'), names)
            # This legacy audio sentence is generated from check outputs. Use
            # the same narrow projection as JSON; source observations stay exact.
            observed = row.get('observed')
            if signal_categories.get(str(row.get('id'))) == 'model' and isinstance(observed, str) and 'per model:' in observed:
                row['observed'] = project({'category': 'model', 'observed': observed})['observed']
    for index, row in enumerate((out.get('detectors') or {}).get('not_run') or [], 1):
        row['name'] = f'Analysis check {index}'
        row['reason'] = _text(row.get('reason'), names)
    for index, block in enumerate((out.get('detectors') or {}).get('detectors') or [], 1):
        block['id'], block['name'] = f'Check {index}', ''
        block['version'] = 'withheld'
        if block.get('weights') != 'model-free':
            block['weights'] = 'Withheld in public report'
        block['citation'] = 'Detector publication withheld in this report.'
        block['false_alarms'] = _text(block.get('false_alarms'), names)
        block['testing'] = _machine(block.get('testing'), names)
    for index, row in enumerate((out.get('methods') or {}).get('validation') or [], 1):
        row['id'] = aliases.get(str(row.get('id')), f'Check {index}')
        row['version'] = 'withheld'
    for row in (out.get('methods') or {}).get('method') or []:
        row['value'] = _configuration_text(row.get('value'), names)
    appendix = out.get('appendix') or {}
    for index, row in enumerate(appendix.get('registry') or [], 1):
        row['id'] = aliases.get(str(row.get('id')), f'Check {index}')
        row['version'] = 'withheld'
        if row.get('weights') != 'model-free':
            row['weights'] = 'Withheld in public report'
        coverage = str(row.get('coverage') or '')
        row['coverage'] = _configuration_text(coverage.replace('DFDC face-swap methods', 'face-swap benchmark methods'), names)
    for row in appendix.get('rows') or []:
        if row.get('key') == 'Re-run':
            row['value'] = 'Exact reproduction requires the original internal analysis record and model registry.'
        elif row.get('key') == 'Calibration id':
            row['value'] = 'Calibration record identity withheld in public report'
        elif row.get('key') == 'Libraries':
            row['value'] = 'Implementation library versions withheld in public report'
        elif row.get('key') == 'Sampling and seeds':
            sampling = snapshot.result.get('manifest', {}).get('sampling') or {}
            # New immutable source records contain internal check identifiers.
            # Their commitment is public, their nested implementation records are not prose.
            bound_records = {'audio_evidence', 'document_source', 'video_audio_evidence', 'brief_face_review'}
            if bound_records.intersection(sampling):
                from app.reports.forensic import SAMPLING_WORDS

                visible = {key: value for key, value in sampling.items() if key not in bound_records}
                row['value'] = ' - '.join(f"{SAMPLING_WORDS.get(key, key.replace('_', ' '))} {value}" for key, value in sorted(visible.items()))
                row['value'] += ('; ' if row['value'] else '') + 'Source-coordinate records are committed by the frozen evidence-set hash.'
    # These are generated exhibit labels, not metadata table cells or analyst annotations.
    for row in out.get('exhibits') or []:
        for field in ('title', 'caption', 'creator', 'created_by'):
            if field in row:
                row[field] = _text(row[field], names)
        # Text-span table sentence columns quote the submitted document. Only the generated Reasons column is transformed.
        table = row.get('table') or {}
        if row.get('kind') == 'text_spans' and 'Reasons' in (table.get('columns') or []):
            reasons = table['columns'].index('Reasons')
            for cells in table.get('rows') or []:
                if len(cells) > reasons:
                    cells[reasons] = _text(cells[reasons], names)
        series = row.get('series') or {}
        for strip in series.get('strips') or []:
            strip['label'] = _text(strip.get('label'), names)
    from app.reports.customer_view import customer_view

    return customer_view(snapshot, out, lambda text: _text(text, names))


def public_manifest(snapshot: ReportSnapshot, version: str, template_set: str) -> dict[str, Any]:
    """Public hash commitments do not disclose the private detector registry or claim standalone reproduction."""
    from app.reports.image_evidence import manifest as evidence_manifest
    from app.reports.multimedia_evidence import manifest as multimedia_manifest

    out = {
        'schema': 'veritas.redacted-report-manifest/1',
        'analysis_id': snapshot.analysis.id,
        'analysis_revision': snapshot.analysis.revision,
        'result_sha256': canonical_hash(snapshot.result),
        'snapshot_sha256': snapshot_hash(snapshot),
        'manifest_sha256': canonical_hash(snapshot.result.get('manifest') or {}),
        'input_sha256': snapshot.media.sha256,
        'report': {'id': snapshot.report.id, 'revision': snapshot.report.revision,
                   'engine': version, 'template_set': template_set},
        'note': 'Model identities withheld. Exact reproduction requires the internal analysis record.',
    }
    evidence = evidence_manifest(snapshot)
    if evidence is not None:
        out['image_evidence'] = evidence
    multimedia = multimedia_manifest(snapshot)
    if multimedia is not None:
        out['evidence'] = multimedia
    return out
