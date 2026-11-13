"""Separately signed public evidence derivatives, without the internal model registry.

Only frozen public packets and their explicitly declared passive media are included.
Original analysis/custody hashes are commitments, not claims that a redacted view
can reproduce their bytes. Legacy packages remain in their original format.
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import and_, or_, select

from app.api.public_projection import alias_signal, project
from app.audit import chain as audit
from app.cases import review
from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.evidence import face_short_window_binding as face_windows
from app.evidence import image_editor_preview_binding as editor_preview
from app.evidence import scene_return_binding
from app.evidence.service import ASSET_TYPES, asset_refs
from app.reports import package as legacy
from app.reports import signing, snapshot
from app.reports.public_policy import legacy_face_scope_safe, package_template
from app.storage.db import Analysis, Artefact, AuditEvent, ReportRow
from app.storage.objects import CHUNK


def public_result(result: dict) -> dict:
    """A bounded customer view; no registry, model cards, internal details or arbitrary exhibits."""
    public = project(result)
    decision = public.get('decision') or {}
    copy_coverage = {}
    source_coverage = {}
    if result.get('kind') == 'image':
        from app.evidence.image_copy_similarity_binding import result_skip

        if scope := result_skip(result):
            copy_coverage['image_copy_coverage'] = scope
    if result.get('kind') == 'video':
        from app.evidence.face_return_coverage import COMMITMENT, from_result

        if brief_review := from_result(result):
            source_coverage[COMMITMENT] = canonical_hash(brief_review)
        if window_review := face_windows.from_result(result):
            source_coverage[face_windows.COMMITMENT] = canonical_hash(window_review)
        sampling = result.get("manifest", {}).get("sampling", {})
        for key in (scene_return_binding.COMMITMENT, scene_return_binding.COVERAGE_COMMITMENT):
            if commitment := sampling.get(key):
                source_coverage[key] = commitment
    if result.get('kind') == 'document':
        from app.evidence import document_passage_position
        from app.evidence.email_alternatives import COMMITMENT, result_review

        if alternatives := result_review(result):
            source_coverage[COMMITMENT] = canonical_hash(alternatives)
        if position := document_passage_position.from_result(result):
            source_coverage[document_passage_position.COMMITMENT] = canonical_hash(position)
    checks = [{k: s[k] for k in ('id', 'display_id', 'category', 'family', 'scope', 'status', 'raw_score',
                                'raw_score_kind', 'calibrated_probability') if k in s}
              for s in public.get('signals') or []]
    checks_by_id = {check["id"]: check for check in checks}
    for signal in result.get("signals") or []:
        check = checks_by_id.get(alias_signal(signal["id"]))
        if check is None or signal.get("effective_weight") != 0.0:
            continue
        details = signal.get("details") or {}
        for key, commitment in (("whole_face_review", "paired_face_measurement_sha256"),
                                ("face_boundary", "face_boundary_measurement_sha256"),
                                ("face_spatial_cut", "face_spatial_cut_measurement_sha256"),
                                ("face_return", "short_face_change_measurement_sha256"),
                                ("face_return_secondary", "second_short_face_change_measurement_sha256")):
            if record := details.get(key):
                check[commitment] = canonical_hash(record)
        if records := details.get("face_short_windows"):
            check[face_windows.CHECKS_COMMITMENT] = [canonical_hash(record) for record in records]
    return {
        'schema': 'chipmunk.public-result/1', 'is_projection': True,
        'original_result_sha256': canonical_hash(result),
        'decision': {k: decision[k] for k in ('code', 'verbal', 'reason', 'confidence', 'band',
                     'probability_ai_calibrated', 'probability_ai_uncalibrated') if k in decision} | copy_coverage,
        'findings': [{k: f[k] for k in ('id', 'family', 'title', 'statement', 'evidence_ids') if k in f}
                     for f in public.get('findings') or []],
        'checks': checks,
        **source_coverage,
        'note': 'This public projection has its own hash. It does not reproduce or hash to the original internal result.',
    }


def _evidence(frozen):
    if frozen.analysis.kind == 'image':
        from app.reports.image_evidence import verified_set
    else:
        from app.reports.multimedia_evidence import verified_set
    document = verified_set(frozen)
    if document is None:
        raise legacy.PackageError('This report has no frozen evidence set. Generate a new report after analysing the source.')
    return document


def _editor_preview(db, frozen, evidence):
    """Project only supported frozen preview exhibits and resolve their exact object IDs."""
    from app.evidence.image_thumbnail_evidence import CREATOR, from_result, is_preview_exhibit

    preview = from_result(frozen.result)
    if not preview or preview.get("preview_carrier") not in editor_preview.CARRIER_SCHEMAS:
        return None, {}
    if frozen.report.redacted:
        raise legacy.PackageError("Thumbnail crops require an unredacted image report.")
    rows = [{**row, "crops": {role: {k: v for k, v in asset.items() if k != "url"}
                             for role, asset in row["crops"].items()}} for row in preview["regions"]]
    projection = {"schema": editor_preview.CARRIER_SCHEMAS[preview["preview_carrier"]], "regions": rows,
                  **{key: preview[key] for key in ("source_sha256", "preview_carrier", "main_exif_orientation",
                     "preview_alignment_orientation", "authorship_confidence_contribution")}}
    exhibits = {item["id"]: item for item in frozen.result.get("exhibits", [])
                if item.get("kind") == "image" and item.get("role") == "comparison" and is_preview_exhibit(item)}
    try:
        groups = list(editor_preview.asset_groups(projection, evidence["source"]))
    except ValueError as error:
        raise legacy.PackageError(str(error)) from error
    indexed = {}
    for group in groups:
        for exhibit_id, asset in group.items():
            sha = asset["sha256"]
            exhibit = exhibits.get(exhibit_id) or {}
            artefact = db.get(Artefact, exhibit["payload_artefact_id"]) if exhibit.get("payload_artefact_id") else None
            if (artefact is None or artefact.analysis_id != frozen.analysis.id or artefact.created_by != CREATOR
                    or artefact.role != "exhibit_payload" or artefact.media_type != asset["media_type"]
                    or not 0 < artefact.bytes <= editor_preview.MAX_ASSET_BYTES
                    or artefact.sha256 != sha or exhibit.get("sha256") != sha
                    or frozen.exhibit_images.get(exhibit_id) != sha):
                raise legacy.PackageError("A thumbnail crop no longer matches the frozen image report.")
            if sha in indexed and indexed[sha][0] != asset:
                raise legacy.PackageError("One thumbnail crop has conflicting frozen geometry.")
            indexed[sha] = asset, artefact
    return projection, indexed


def build_public(db, package, *, store, key: signing.InstanceKey, exported_at: datetime) -> legacy.Built:
    analysis = db.get(Analysis, package.analysis_id)
    report = db.get(ReportRow, package.report_id)
    if analysis is None or report is None or report.analysis_id != analysis.id:
        raise legacy.PackageError('The report does not belong to the requested analysis.')
    if (report.status != 'done' or not report.artefact_id or not report.signature or report.redacted
            or not package_template(report.template_set, analysis.kind)):
        raise legacy.PackageError('Generate a current, signed, unredacted report before exporting its evidence package.')
    if package.kind == 'issue':
        if analysis.withdrawn_at is not None or review.superseded(db, analysis.id):
            raise legacy.PackageError('A withdrawn or superseded analysis cannot be issued.')
        if not report.issuable or not review.brief(db, analysis)['issuable']:
            raise legacy.PackageError('The report or current review state is not issuable.')
    frozen = snapshot.load(legacy._verified(db, store, report.snapshot_artefact_id, analysis.id))
    if (snapshot.snapshot_hash(frozen) != report.snapshot_hash or frozen.analysis.id != analysis.id
            or frozen.analysis.revision != analysis.revision or frozen.media.sha256 != analysis.sha256
            or frozen.analysis.kind != analysis.kind or frozen.media.size_bytes != analysis.size_bytes
            or canonical_hash(frozen.result) != analysis.result_hash
            or frozen.report.id != report.id or frozen.report.template_set != report.template_set
            or frozen.report.profile != report.profile):
        raise legacy.PackageError('The report snapshot does not match its analysis and stored identity.')
    if not legacy_face_scope_safe(frozen):
        raise legacy.PackageError('Generate a current, signed, unredacted report before exporting its evidence package.')
    evidence = _evidence(frozen)
    result = public_result(frozen.result)
    previews = {}
    if analysis.kind == "image" and report.profile in editor_preview.REPORT_PROFILES:
        projection, previews = _editor_preview(db, frozen, evidence)
        if projection is not None:
            if result["decision"].get("image_copy_coverage") is not None:
                raise legacy.PackageError("Thumbnail evidence conflicts with the original provenance scope.")
            result[editor_preview.KEY] = projection
    if analysis.kind != 'image':
        from app.reports.multimedia_evidence import verify_assets

        verify_assets(evidence, store, result=frozen.result)
    if analysis.kind == 'document':
        from app.evidence import document_email_contexts

        if scope := (evidence.get("coverage") or {}).get(document_email_contexts.KEY):
            result[document_email_contexts.COMMITMENT] = canonical_hash(scope)
    files = {'report.pdf': legacy._verified(db, store, report.artefact_id, analysis.id),
             'evidence.json': canonical_json(evidence), 'result.json': canonical_json(result)}
    if (hashlib.sha256(files['report.pdf']).hexdigest() != report.sha256 or report.signing_key_id != key.key_id
            or not signing.verify(key.public, files['report.pdf'], report.signature)):
        raise legacy.PackageError('The frozen report signature or hash does not verify with this instance key.')

    assets = []
    extensions = {mime: suffix for suffix, mime in ASSET_TYPES.items()}
    refs = asset_refs(evidence)
    for sha, (asset, _) in previews.items():
        reference = refs.setdefault(sha, asset)
        if any(asset[key] != reference.get(key) for key in ("media_type", "width", "height")):
            raise legacy.PackageError("A thumbnail crop conflicts with another frozen asset.")
    for sha, asset in sorted(refs.items()):
        mime = asset['media_type']
        if sha in previews:
            indexed = previews[sha][1]
        else:
            indexed = db.scalar(select(Artefact).where(Artefact.analysis_id == analysis.id, Artefact.sha256 == sha,
                                Artefact.media_type == mime, Artefact.role.in_(('preview', 'working_copy'))))
        if indexed is None:
            raise legacy.PackageError('A declared evidence asset is no longer indexed for this analysis.')
        name = f'assets/{sha}.{extensions[mime]}'
        files[name] = legacy._read(store, indexed)
        assets.append({'file': name, 'sha256': sha, 'media_type': mime, 'bytes': len(files[name])})
    source = None
    if package.include_original:
        suffix = Path(analysis.filename).suffix.lower()
        suffix = suffix if re.fullmatch(r'\.[a-z0-9]{1,8}', suffix) else '.bin'
        source = {'file': 'source/original' + suffix, 'sha256': analysis.sha256,
                  'content_type': analysis.content_type, 'bytes': analysis.size_bytes,
                  'note': 'User-supplied original bytes, not redacted. They may contain identifying metadata or text.'}

    scope = AuditEvent.analysis_id == analysis.id
    if analysis.media_id is not None and analysis.case_id is not None:
        scope = or_(scope, and_(AuditEvent.analysis_id.is_(None), AuditEvent.media_id == analysis.media_id,
                               AuditEvent.case_id == analysis.case_id))
    events = [{'seq': e.seq, 'event_hash': e.event_hash} for e in db.scalars(
        select(AuditEvent).where(scope).order_by(AuditEvent.seq))]
    tail = db.execute(select(AuditEvent.seq, AuditEvent.event_hash).order_by(AuditEvent.seq.desc()).limit(1)).first()
    anchor = {'head_seq': int(tail[0]) if tail else 0, 'head_hash': str(tail[1]) if tail else audit.GENESIS,
              'at': legacy._ts(exported_at)}
    record = legacy._report_entry(report, 'report.pdf', issued=package.kind == 'issue')
    record.update(analysis_id=analysis.id, snapshot_sha256=report.snapshot_hash, evidence_set_sha256=evidence['sha256'])
    bundle = {
        'schema': legacy.PUBLIC_SCHEMA, 'bundle_id': package.id, 'package_id': package.id, 'kind': package.kind,
        'generated_at': legacy._ts(exported_at), 'reproduction_supported': False,
        'analysis': {'id': analysis.id, 'revision': analysis.revision, 'kind': analysis.kind,
                     'original_result_sha256': analysis.result_hash},
        'source_commitment': {'sha256': analysis.sha256, 'bytes': analysis.size_bytes},
        'result': {'file': 'result.json', 'sha256': hashlib.sha256(files['result.json']).hexdigest(), 'is_projection': True},
        'evidence': {'file': 'evidence.json', 'sha256': hashlib.sha256(files['evidence.json']).hexdigest(),
                     'id': evidence['id'], 'set_sha256': evidence['sha256'], 'schema': evidence['schema'],
                     'version': evidence['version'], 'source_sha256': analysis.sha256},
        'reports': [record], 'artifacts': assets, 'source': source,
        'custody': {'schema': 'chipmunk.custody-commitments/1', 'verification': 'commitments_only',
                    'complete_event_chain_included': False, 'events': events, 'anchor': anchor,
                    'note': 'Original event hashes are signed commitments. Original event bodies are withheld; their chain cannot be recomputed from this derivative.'},
        'signing': {'algorithm': signing.ALGORITHM, 'key_id': key.key_id, 'public_key_file': 'instance_public_key.pem'},
    }
    files['bundle.json'] = canonical_json(bundle)
    files['instance_public_key.pem'] = key.public_pem()
    files['reproduce.md'] = b'# Public evidence derivative\n\nThis package cannot reproduce model inference. Obtain the authorised internal analysis record separately. Adding the source does not change this limitation.\n'
    preview_note = ('result.json also declares the saved editor thumbnail crops and footprint. '
                    'Use verify with --source-image to check them against the submitted JPEG.\n' if previews else '')
    files['README.txt'] = (
        'Chipmunk Verify public evidence package\n\n'
        f"Status: {package.kind.upper()}; report {report.display_id}.\n"
        'evidence.json is the exact frozen evidence set used by the report and API.\n'
        'assets/ contains all declared evidence images, audio clips and silent video context clips, named by SHA-256.\n'
        'result.json is a public projection with its own hash, not the original internal result.\n'
        f'{preview_note}'
        'Source excerpts and crops are included. The optional source/ file is unredacted user content.\n'
        'No internal model registry, arbitrary internal artifacts, or original custody event bodies are included.\n'
        'Custody entries are commitments only; this export cannot prove the completeness of the original event chain.\n'
        'A valid signature proves integrity and signer identity against a separately obtained key; it does not prove an AI verdict.\n\n'
        'Offline verification (obtain the trusted key separately from the issuing instance):\n'
        '  python -m app.cli verify <package.zip> --public-key <trusted-instance.pem>\n'
        'Without a trusted key the verifier can only report self-consistency, not authenticate the signer.\n'
        'Do not run reproduce on this public derivative; see reproduce.md.\n'
    ).encode()
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    if source:
        hashes[source['file']] = analysis.sha256
    manifest = ''.join(f'{hashes[name]}  {name}\n' for name in sorted(hashes)).encode('utf-8')
    signature = key.sign(manifest)
    files['manifest.sha256'], files['manifest.sha256.sig'] = manifest, (signature + '\n').encode('ascii')
    names = sorted([*files, *([source['file']] if source else [])])
    workdir = settings.data_dir / 'tmp' / 'packages'
    workdir.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=workdir, prefix=f'{package.id}-public-', suffix='.zip')
    path = Path(temporary)
    stamp = (exported_at if exported_at.tzinfo else exported_at.replace(tzinfo=UTC)).astimezone(UTC).timetuple()[:6]
    try:
        with os.fdopen(fd, 'wb') as stream, zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                info = zipfile.ZipInfo(name, date_time=stamp)
                info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
                if name in files:
                    archive.writestr(info, files[name])
                    continue
                copied = hashlib.sha256()
                with legacy._open_original(db, store, analysis) as original, archive.open(info, 'w', force_zip64=True) as dst:
                    while chunk := original.read(CHUNK):
                        copied.update(chunk)
                        dst.write(chunk)
                if copied.hexdigest() != analysis.sha256:
                    raise legacy.PackageError('The optional original changed during export.')
        digest, size = hashlib.sha256(), 0
        with path.open('rb') as stream:
            while chunk := stream.read(CHUNK):
                digest.update(chunk)
                size += len(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return legacy.Built(path, size, digest.hexdigest(), hashlib.sha256(manifest).hexdigest(), signature, len(names), anchor)
