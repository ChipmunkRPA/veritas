"""Veritas command-line interface.

    python -m app.cli models list [--json] [--policy PRESET|JSON] [--manifest PATH]
    python -m app.cli models verify [--installed] [--force] [--manifest PATH]
    python -m app.cli models fetch (MODEL_ID ... | --all | --enabled | --profile PROFILE) [--dry-run] [--manifest PATH]
    python -m app.cli models pin MODEL_ID [--write] [--manifest PATH]
    python -m app.cli models footprint [--manifest PATH]
    python -m app.cli models plan [--profile PROFILE] [--budget-mb MB] [--device cpu|mps|cuda] [--json]
    python -m app.cli verify PACKAGE.zip (--public-key PEM | --expect-key-id ID) [--source-image IMAGE | --source-audio MEDIA | --source-video VIDEO] [--json]
    python -m app.cli compare-image REFERENCE SUBMITTED [--evidence-dir DIRECTORY]
    python -m app.cli reproduce PACKAGE.zip (--public-key PEM | --expect-key-id ID) [--tolerance 0.001] [--json]
    python -m app.cli custody export [--format jsonl|csv] [--case-id ID] [--out PATH] [--database-url URL]
    python -m app.cli audit verify [--jsonl PATH] [--database-url URL]
    python -m app.cli db setup --app-role ROLE [--database-url URL] [--password-env NAME]
    python -m app.cli db grants --app-role ROLE [--database-url URL] [--apply]
    python -m app.cli retention purge [--dry-run]
    python -m app.cli principals create --username NAME --role ROLE [--display-name TEXT]
    python -m app.cli principals list
    python -m app.cli principals set-password --username NAME
    python -m app.cli principals disable --username NAME
    python -m app.cli principals enable --username NAME
    python -m app.cli principals set-role --username NAME --role ROLE
    python -m app.cli principals api-key --username NAME
    python -m app.cli store gc [--dry-run]
    python -m app.cli store verify
    python -m app.cli backup create DIR
    python -m app.cli backup verify DIR
    python -m app.cli backup restore DIR [--data-dir PATH] [--replace]
    python -m app.cli keys adopt-derived [--api-key-env NAME]
    python -m app.cli fusers status [--profile PROFILE] [--json]
    python -m app.cli fusers review --modality KIND --reviewer NAME --statement TEXT [--sha256 PREFIX]
    python -m app.cli fusers withdraw-review --modality KIND [--sha256 PREFIX] [--reason TEXT]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    for row in (headers, *rows):
        print("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)).rstrip())


def _pin(model) -> str:
    if model.source.type == "hf":
        return (model.source.revision or "")[:12]
    first = next((f.sha256 for f in model.source.files if f.sha256), None)
    return f"sha256:{first[:12]}" if first else "-"


def _manifest_or_exit(path: str | None):
    from app.engine.manifest import ManifestError, load_manifest

    try:
        return load_manifest(path)
    except ManifestError as e:
        print(f"error: {e}", file=sys.stderr)
        return None


def _models_list(args: argparse.Namespace) -> int:
    from app.core.config import settings
    from app.engine.fetchers import installed
    from app.engine.registry import DetectorRegistry, parse_policy

    manifest = _manifest_or_exit(args.manifest)
    if manifest is None:
        return 2
    try:
        policy = parse_policy(settings.licence_policy if args.policy is None else args.policy)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    registry = DetectorRegistry(manifest, policy)
    decisions = {e.detector_id: e for e in registry.decisions()}

    if args.json:
        print(json.dumps({
            "licence_policy": policy.model_dump(),
            "disclosures": registry.disclosures(),
            "models": [{**m.model_dump(mode="json", by_alias=True), "installed": installed(m)} for m in manifest.models],
            "detectors": [{**d.model_dump(mode="json", by_alias=True), "enabled": decisions[d.id].enabled,
                           "disabled_reasons": list(decisions[d.id].reasons)} for d in manifest.detectors],
        }, indent=2))
        return 0

    print(f"licence policy: {policy.label} (commercial={policy.commercial}, non_commercial={policy.non_commercial}, "
          f"unknown={policy.unknown}, allow={policy.allow_detectors}, deny={policy.deny_detectors})")
    for line in registry.disclosures():  # D-34
        print(f"disclosure: {line}")
    print("\nMODELS")
    _table(("ID", "CLASS", "SPDX", "SOURCE", "PIN", "INSTALLED"),
           [(m.id, m.licence.class_, m.licence.spdx, m.source.type, _pin(m), "yes" if installed(m) else "no")
            for m in manifest.models])
    print("\nDETECTORS")
    _table(("ID", "MODALITY", "PROFILES", "DEFAULT", "STATE", "WHY"),
           [(d.id, d.modality.value, ",".join(d.profiles), "yes" if d.default_enabled else "no",
             "enabled" if decisions[d.id].enabled else "disabled", "; ".join(decisions[d.id].reasons))
            for d in manifest.detectors])
    return 0


def _models_verify(args: argparse.Namespace) -> int:
    from app.engine.fetchers import (
        FetchError,
        ModelNotInstalled,
        WeightIntegrityError,
        ensure_verified,
        local_install_dir,
    )
    from app.engine.manifest import lint

    manifest = _manifest_or_exit(args.manifest)
    if manifest is None:
        return 1
    issues = lint(manifest)
    for issue in issues:
        print(issue)
    errors = sum(issue.level == "error" for issue in issues)
    print(f"{len(manifest.models)} models, {len(manifest.detectors)} detectors: "
          f"{errors} error(s), {len(issues) - errors} warning(s)")
    integrity = 0
    if args.installed:
        counts = {"verified": 0, "not installed": 0}
        for model in manifest.models:
            try:
                ensure_verified(model, force=args.force)
                counts["verified"] += 1
                print(f"VERIFIED  {model.id}")
            except ModelNotInstalled:
                counts["not installed"] += 1
                where = f" (a local artefact: copy it to {local_install_dir(model)})" if model.source.type == "local" else ""
                print(f"WARNING   {model.id}: not installed{where}")
            except (WeightIntegrityError, FetchError) as e:
                integrity += 1
                print(f"ERROR     {model.id}: {e}")
        print(f"installed weights: {counts['verified']} verified, {counts['not installed']} not installed, "
              f"{integrity} integrity error(s)")
    return 1 if errors or integrity else 0


# N-14a: a `models-init` that fails or is killed left an operator with an exit code and no word that a second run goes on
FETCH_GOES_ON = "Files that were fetched and verified are kept and are not fetched again: the fetch goes on with the rest."


def _models_fetch(args: argparse.Namespace) -> int:
    from app.engine.fetchers import (
        FetchError,
        ModelNotInstalled,
        NoRecord,
        OfflineError,
        WeightIntegrityError,
        fetch_model,
        file_path,
        installed,
        local_install_dir,
    )

    manifest = _manifest_or_exit(args.manifest)
    if manifest is None:
        return 2
    if args.all:
        targets = list(manifest.models)
    elif args.enabled or args.profile:
        from app.core.config import settings
        from app.engine.registry import DetectorRegistry, parse_policy

        if args.model_ids:
            print("error: name models or use --enabled / --profile, not both", file=sys.stderr)
            return 2
        registry = DetectorRegistry(manifest, parse_policy(settings.licence_policy))
        enabled = {e.detector_id for e in registry.decisions() if e.enabled}
        wanted = {m for d in manifest.detectors if d.id in enabled and (args.profile is None or args.profile in d.profiles)
                  for m in registry.required_models(d)}
        targets = [m for m in manifest.models if m.id in wanted]
        print(f"{len(targets)} model(s) of the detectors enabled by licence policy {registry.policy.label}"
              + (f" in the {args.profile} profile" if args.profile else ""))
    else:
        unknown = [i for i in args.model_ids if manifest.model(i) is None]
        if unknown or not args.model_ids:
            print(f"error: unknown model(s): {', '.join(unknown)}" if unknown else "error: name models or use --all",
                  file=sys.stderr)
            return 2
        targets = [manifest.model(i) for i in args.model_ids]
    failures = 0
    if targets and not args.dry_run:  # said first: a fetch that is killed (exit code 137 in a container) says nothing more
        print(f"If this stops before it ends, run the same command again. {FETCH_GOES_ON}", flush=True)
    for model in targets:
        if model.source.type == "local":  # D-62: distributed out of band, never downloaded
            size = sum(f.bytes or 0 for f in model.source.files)
            print(f"LOCAL     {model.id}: a local artefact, not downloaded ({len(model.source.files)} files, {size / 1e9:.1f} GB, "
                  f"produced by {model.source.produced_by}). Copy its directory to {local_install_dir(model)} and run "
                  f"`python -m app.cli models verify --installed` ({'installed' if installed(model) else 'missing'})")
            failures += 0 if args.all or args.enabled or args.profile or args.dry_run else 1  # only a named local model is a failure
            continue
        if args.dry_run:
            try:
                where = ", ".join(str(file_path(model, f)) for f in model.source.files)
            except FetchError as e:
                where = str(e)
            print(f"PLAN      {model.id}: {model.source.type} -> {where} ({'installed' if installed(model) else 'missing'})")
            continue
        try:
            paths = fetch_model(model, record=NoRecord()) if args.no_record else fetch_model(model)
            # FETCHED: something was downloaded. INSTALLED: every file was there and was only hashed (N-14c, E-12)
            word = "FETCHED" if getattr(paths, "downloaded", True) else "INSTALLED"
            print(f"{word:<9} {model.id}: {len(paths)} file(s) verified")
        except OfflineError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        except (FetchError, WeightIntegrityError, ModelNotInstalled) as e:
            failures += 1
            print(f"ERROR     {model.id}: {e}")
    if failures and not args.dry_run:
        print(f"{failures} model(s) could not be fetched. Run the same command again once the cause is cured. {FETCH_GOES_ON}")
    return 1 if failures else 0


def _models_pin(args: argparse.Namespace) -> int:
    from app.core.config import settings
    from app.engine.fetchers import FetchError, ModelNotInstalled, OfflineError, pin_manifest
    from app.engine.manifest import DEFAULT_MANIFEST, ManifestError

    path = Path(args.manifest) if args.manifest else (settings.models_manifest or DEFAULT_MANIFEST)
    try:
        diff = pin_manifest(path, args.model_id, write=args.write)
    except KeyError as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 2
    except OfflineError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (FetchError, ModelNotInstalled, ManifestError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(diff or f"{args.model_id}: already pinned")
    if args.write and diff:
        print(f"wrote {path}")
    return 0


FOOTPRINT_NOTE = ("MEASURED MB: the allocator's count across the last load on mps or cuda; the largest ever recorded is "
                  "reserved for the model there.\n"
                  "CPU PROCESS MB: the change of this process's resident memory across the last load on CPU, an observation of "
                  "the process and not the model's size; it reserves nothing (the declaration does).")


def _models_footprint(args: argparse.Namespace) -> int:
    from app.core.config import settings
    from app.engine.models import FootprintStore, MiB

    manifest = _manifest_or_exit(args.manifest)
    if manifest is None:
        return 2
    store = FootprintStore(settings.data_dir / "state" / "model_footprints.json")
    rows = []
    for model in manifest.models:
        fp = model.footprint
        measured = ", ".join(f"{kind} {nbytes / MiB:.1f}" for kind in ("mps", "cuda")
                             if (nbytes := store.measured(model.id, kind))) or "-"
        on_cpu = f"{nbytes / MiB:.1f}" if (nbytes := store.measured(model.id, "cpu")) else "-"
        rows.append((model.id, "-" if fp.fp32_mb is None else f"{fp.fp32_mb:g}", "-" if fp.fp16_mb is None else f"{fp.fp16_mb:g}",
                     "estimate" if fp.estimated else "measured", measured, on_cpu, store.pin(model.id) or "-"))
    _table(("ID", "FP32 MB", "FP16 MB", "DECLARED", "MEASURED MB", "CPU PROCESS MB", "DEVICE PIN"), rows)
    print(FOOTPRINT_NOTE)
    return 0


def _models_plan(args: argparse.Namespace) -> int:
    """What the profile needs of the model memory budget, and which detectors cannot run under it (BUILD_SPEC N-14a)."""
    from app.core.config import settings
    from app.engine import capacity
    from app.engine.models import MiB

    if args.budget_mb is not None and args.budget_mb <= 0:
        print("error: --budget-mb must be a positive number of MB", file=sys.stderr)
        return 2
    try:
        plan = capacity.plan(args.profile, device=args.device,
                             budget_bytes=None if args.budget_mb is None else int(args.budget_mb * MiB))
    except ValueError as e:  # VERITAS_MODEL_MEMORY_BUDGET_MB that is neither `auto` nor a number
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
        return 1 if plan.cannot_run else 0

    source = "--budget-mb" if args.budget_mb is not None else f"VERITAS_MODEL_MEMORY_BUDGET_MB={settings.model_memory_budget_mb}"
    print(f"Profile {plan.profile} on {plan.device}; model memory budget {plan.budget_bytes // MiB} MB ({source}).")
    print("A detector's need is the weights that are loaded together while it runs plus the activation allowance of its "
          "plan. No weights were loaded for this table.")
    print()
    shown = plan.with_models
    if shown:
        _table(("DETECTOR", "DEVICE", "PRECISION", "WEIGHTS MB", "ALLOWANCE MB", "NEED MB", "RUNS"),
               [(d.detector_id, d.device or "-", d.dtype or "-", capacity.mb_up(d.weight_bytes), capacity.mb_up(d.activation_bytes),
                 capacity.mb_up(d.need_bytes), "yes" if d.can_run else f"no: {(d.refusal or '').removeprefix('memory budget: ')}")
                for d in shown])
        print()
    print(f"{len(plan.detectors) - len(shown)} of the {len(plan.detectors)} detectors of this profile load no model.")
    refused = plan.cannot_run
    if refused:
        print(f"{len(refused)} of the {len(plan.detectors)} cannot run on this server under this budget: "
              f"{', '.join(d.detector_id for d in refused)}. Each is recorded as skipped in every analysis it applies to.")
    else:
        print("Every detector of this profile can run under this budget.")
    for d in refused:
        if not d.plannable:
            print(f"No budget cures {d.detector_id}: {d.refusal}.")
    largest = plan.largest
    if largest is not None and largest.need_bytes:
        kept = (f"; {capacity.mb_up(largest.beside_bytes)} MB of it are models that stay loaded once they were used"
                if largest.beside_bytes else "")
        memory = {"cuda": "a CUDA accelerator", "mps": "an Apple-silicon host (MPS)"}.get(plan.device, "a CPU host")
        print(f"Smallest budget that runs the whole profile: {capacity.mb_up(plan.smallest_budget_bytes)} MB "
              f"(set by {largest.detector_id}{kept}). Under the default rule (VERITAS_MODEL_MEMORY_BUDGET_MB=auto) "
              f"{memory} reaches it with {capacity.mb_up(plan.memory_for_default_rule_bytes)} MB of memory.")
    return 1 if refused else 0


def _sessions(database_url: str | None):
    """Sessions on a database copy or on the configured database, read as it is: a command that only reads migrates nothing
    (N-14; OPS-08)."""
    if database_url is None:
        from app.storage.db import sessions_as_found

        return sessions_as_found()
    from sqlalchemy.orm import sessionmaker

    from app.storage.db import make_engine

    return sessionmaker(bind=make_engine(database_url), expire_on_commit=False)


UNREADABLE_LOG = ("the custody log cannot be read from this database as it is ({error}). This command migrates nothing: a "
                  "database of an earlier version is brought up to date by starting the service on it.")


def _custody_export(args: argparse.Namespace) -> int:
    from sqlalchemy.exc import SQLAlchemyError

    from app.audit import chain

    try:
        events = chain.iter_events(_sessions(args.database_url), case_ids=None if args.case_id is None else [args.case_id])
        lines = chain.jsonl_lines(events) if args.format == "jsonl" else chain.csv_lines(events)
        if args.out is None:
            for line in lines:
                sys.stdout.write(line)
            return 0
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            for line in lines:
                f.write(line)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except SQLAlchemyError as e:
        print("error: " + UNREADABLE_LOG.format(error=type(getattr(e, "orig", e)).__name__), file=sys.stderr)
        return 2
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


def _audit_verify(args: argparse.Namespace) -> int:
    from sqlalchemy.exc import SQLAlchemyError

    from app.audit import chain

    try:
        events = chain.read_jsonl(Path(args.jsonl)) if args.jsonl else chain.iter_events(_sessions(args.database_url))
        result = chain.verify_events(events)
    except SQLAlchemyError as e:
        print("error: " + UNREADABLE_LOG.format(error=type(getattr(e, "orig", e)).__name__), file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if result.valid:
        print(f"VALID     {result.checked} event(s)")
        return 0
    print(f"BROKEN    at seq {result.first_broken_seq} after {result.checked} event(s): {result.reason}")
    return 1


def _pin_arguments(parser: argparse.ArgumentParser) -> None:
    """The instance key a package is checked against; it comes from the instance or its operator, never from the package."""
    parser.add_argument("--public-key", default=None, metavar="PEM",
                        help="file with the instance public key (GET /api/v2/instance/public-key, or the export dialog)")
    parser.add_argument("--expect-key-id", default=None, metavar="ID",
                        help="the instance key id, ed25519:<16 hex>, as the export dialog shows it; without this or --public-key "
                             "the signer is not authenticated (exit 3)")


def _public_key(path: str | None) -> bytes | None:
    return None if path is None else Path(path).read_bytes()


def _signer_not_authenticated(package: str, key_id: str | None, bundle: dict | None = None) -> None:
    from app.reports import verify

    print(f"{verify.SIGNER_NOT_AUTHENTICATED}  {package} (key {key_id or 'unknown'}, from inside the package)"
          f"{verify.draft_note(bundle)}")
    for line in verify.HOW_TO_PIN.splitlines():
        print(f"          {line}")


def _verify(args: argparse.Namespace) -> int:
    """Offline package verification (RP-05): no server, database or network. Exit 0 verified against the pinned instance key,
    1 not verified, 2 unreadable or a pin of the wrong form, 3 consistent with the key it carries and no key pinned (N-11)."""
    from app.reports import verify

    try:
        result = verify.verify_package(Path(args.package), public_key_pem=_public_key(args.public_key),
                                       expect_key_id=args.expect_key_id,
                                       source_image=Path(args.source_image) if args.source_image else None,
                                       source_audio=Path(args.source_audio) if args.source_audio else None,
                                       source_video=Path(args.source_video) if args.source_video else None)
    except (verify.PackageUnreadable, verify.PinInvalid, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return verify.EXIT_ERROR
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
        return verify.EXIT[result.outcome]
    for check in result.checks:
        print(f"{'PASS' if check.ok else 'FAIL':<9} {check.name}: {check.detail}")
    if result.outcome == "self_consistent":
        _signer_not_authenticated(args.package, result.key_id, result.bundle)
    else:
        print(f"{'VERIFIED' if result.ok else 'NOT VERIFIED':<9} {args.package} (key {result.key_id or 'unknown'})"
              f"{verify.draft_note(result.bundle)}")
    return verify.EXIT[result.outcome]


def _compare_image(args: argparse.Namespace) -> int:
    from PIL import Image

    from app.evidence import image_reference

    try:
        result = image_reference.inspect(Path(args.reference), Path(args.submitted),
                                         evidence_dir=Path(args.evidence_dir) if args.evidence_dir else None)
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def _db_grants(args: argparse.Namespace) -> int:
    """Postgres grants for the application role (ARCHITECTURE §4.8; J-6): prints the SQL; --apply runs it over an owner connection."""
    from sqlalchemy.exc import DBAPIError

    from app.storage import pg_grants
    from app.storage.db import make_engine

    try:
        sql = pg_grants.statements(args.app_role)
        if args.apply:
            if not args.database_url:
                print("error: --apply needs --database-url (an owner connection)", file=sys.stderr)
                return 2
            engine = make_engine(args.database_url)
            try:
                pg_grants.apply(engine, args.app_role)
            finally:
                engine.dispose()
    except pg_grants.GrantError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except pg_grants.NotBroughtAbout as e:  # N-19g (d): the reading back at the end
        print(f"error: {e}", file=sys.stderr)
        return 1
    except DBAPIError as e:  # N-19g: no statement ends in a traceback; the first line only, as `db setup`
        print(f"error: {(str(e.orig).strip().splitlines() or [type(e.orig).__name__])[0]}", file=sys.stderr)
        return 1
    print(";\n".join(sql) + ";")
    return 0


def _db_setup(args: argparse.Namespace) -> int:
    """The owner's step on Postgres (N-14; SEC-17, PG-13): migrate, default zone UTC, the application role and its grants."""
    import os

    from sqlalchemy.exc import DBAPIError

    from app.core.config import settings
    from app.storage import migrations, pg_grants
    from app.storage.db import make_engine

    password = None
    if args.password_env:
        password = os.environ.get(args.password_env)
        if not password:
            print(f"error: the environment variable {args.password_env} is not set or is empty", file=sys.stderr)
            return 2
    engine = make_engine(args.database_url or settings.sqlalchemy_url)
    try:
        lines = pg_grants.setup(engine, args.app_role, password=password)
    except pg_grants.GrantError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    # N-19g: a refused role statement, and a reading back at the end that does not hold, are one sentence each
    except (migrations.MigrationOwnerError, pg_grants.RoleRefused, pg_grants.NotBroughtAbout) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except DBAPIError as e:  # the first line only: no statement, no parameters
        print(f"error: {(str(e.orig).strip().splitlines() or [type(e.orig).__name__])[0]}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    print("\n".join(lines))
    return 0


def _retention_purge(args: argparse.Namespace) -> int:
    """The D-55 retention rule over every revision chain: each chain with its decision and the reason."""
    from app import retention
    from app.core import times
    from app.core.config import settings

    report = retention.purge(dry_run=args.dry_run)
    print(f"retention purge{' (dry run)' if args.dry_run else ''}: {len(report.chains)} chain(s); "
          f"unsigned quick analyses are kept for {settings.retention_days} day(s) (VERITAS_RETENTION_DAYS)")
    changed = set(report.changed)
    rows = []
    for chain in report.chains:
        newest = max(times.to_utc(r.created_at) for r in chain.revisions)  # the UTC date, on every engine
        decision, reason = ("PURGE" if chain.verdict.purge else "KEEP"), chain.verdict.reason
        if chain.root_id in changed:
            decision, reason = "KEEP", "changed while the purge ran; decided again by the next run"
        elif report.failed.get(chain.root_id) == "TailMoved":
            decision, reason = "FAILED", "the custody log kept being extended while the purge ran; nothing of this chain was removed"
        elif chain.root_id in report.failed:
            decision, reason = "FAILED", f"the database refused the delete ({report.failed[chain.root_id]}); nothing of this chain was removed"
        rows.append((decision, chain.root_id, chain.case.reference if chain.case is not None else "-", len(chain.revisions),
                     f"{newest:%Y-%m-%d}", chain.verdict.rule, reason))
    if rows:
        _table(("DECISION", "CHAIN", "CASE", "REVISIONS", "NEWEST", "RULE", "REASON"), rows)
    counted = f"{report.purged_revisions} revision(s) in {report.purged_chains} chain(s)"
    # N-19e: the media records of uploads that were never recorded (a failed or interrupted intake left them)
    unrecorded = f"{report.unrecorded_media} media record(s) of uploads that were never recorded"
    if args.dry_run:
        if report.unrecorded_media:
            print(f"would remove {unrecorded}")
        print(f"would purge {counted}; nothing was changed")
        return 0
    freed = report.released
    print(f"purged {counted}; removed {report.cases_removed} implicit case(s) and {report.media_removed} media record(s); "
          f"freed {freed.unlinked} object(s), {freed.bytes} bytes")
    if report.unrecorded_media:
        print(f"removed {unrecorded}")
    if freed.deferred:
        print(_waiting(freed))
    if report.leftovers:
        print(f"removed the directories of {report.leftovers} revision(s) that an earlier, interrupted run had deleted; "
              "run `python -m app.cli store gc` to free their bytes")
    if report.failed:
        print(f"error: {len(report.failed)} chain(s) could not be purged; see the log", file=sys.stderr)
        return 1
    return 0


def _waiting(released) -> str:
    if released.busy:
        # N-17a (O2-02): what blocks is a job under a live lease, not what waits in the queue
        return (f"{released.deferred} unreferenced object(s) left in place: {released.busy} job(s) that a worker holds may still "
                "index what they wrote. Wait for them to finish, or for their leases to run out, then run "
                "`python -m app.cli store gc`. Queued work does not block.")
    return f"{released.deferred} object(s) left in place: referenced again, or a worker took a job, while the sweep ran."


def _store_gc(args: argparse.Namespace) -> int:
    """Unlink the store objects that no row references (the orphans of earlier deletes, and what a purge left while a worker
    held a job)."""
    from app import retention

    released = retention.store_gc(dry_run=args.dry_run)
    if args.dry_run:
        for sha in released.sha256s:
            print(sha)
    if released.busy:
        print(f"store gc{' (dry run)' if args.dry_run else ''}: nothing removed; {_waiting(released)}")
    elif args.dry_run:
        print(f"store gc (dry run): {released.unlinked} unreferenced object(s), {released.bytes} bytes would be removed")
    else:
        print(f"store gc: removed {released.unlinked} object(s), {released.bytes} bytes")
        if released.deferred:
            print(_waiting(released))
    if released.young:
        print(f"{released.young} object(s) written in the last {int(retention.GC_GRACE_S // 60)} minutes were not examined.")
    return 0


def _short(names: Sequence[str], limit: int = 20) -> list[str]:
    return [*names[:limit], *([f"… and {len(names) - limit} more"] if len(names) > limit else [])]


def _store_verify(args: argparse.Namespace) -> int:
    """Re-hash every store object the database references (N-12). Exit 1 when one is missing or altered."""
    from app.storage import backup

    result = backup.store_verify()
    for word, names in (("MISSING", result.missing), ("ALTERED", result.altered)):
        for sha in _short(names):
            print(f"{word:<9} {sha}")
    print(f"{'VERIFIED' if result.ok else 'DAMAGED':<9} {result.checked} object(s), {result.bytes} bytes re-hashed; "
          f"{len(result.missing)} missing, {len(result.altered)} altered"
          + (f"; {result.elsewhere} referenced hash(es) are not indexed as store objects (an exhibit without a stored payload)"
             if result.elsewhere else ""))
    return 0 if result.ok else 1


def _backup_create(args: argparse.Namespace) -> int:
    """A backup of this instance in a new directory (N-12): exit 0 complete, 1 written without objects the database names,
    2 refused (Postgres, a destination that holds files)."""
    from app.storage import backup

    try:
        made = backup.create(Path(args.directory))
    except backup.BackupRefused as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    manifest = made.manifest
    for word, names in (("MISSING", made.missing), ("ALTERED", made.altered)):
        for sha in _short(names):
            print(f"{word:<9} {sha}")
    head = manifest["database"]["custody_head"]
    print(f"database  {manifest['database']['file']}: revision {manifest['database']['revision']}, {manifest['database']['analyses']} "
          f"analysis row(s), custody head {head['seq']}")
    print(f"objects   {manifest['objects']['copied']} copied, {len(made.missing)} missing, {len(made.altered)} altered")
    for name, entry in manifest["keys"].items():
        if name == "instance_signing_key":
            print(f"key       instance signing key: {entry['key_id'] or 'none yet'}")
        else:
            held = {"file": f"in the backup ({entry['file']})", "environment": "in the environment, NOT in the backup",
                    "derived": "derived from VERITAS_API_KEY (dev), NOT in the backup", "absent": "none yet"}[entry["source"]]
            print(f"key       {name.replace('_', ' ')}: {held}" + (f", fingerprint {entry['fingerprint']}" if entry["fingerprint"] else ""))
    print(f"{'WRITTEN' if made.ok else 'INCOMPLETE':<9} {made.path}: {len(manifest['files'])} file(s). It holds the keys in the "
          f"clear: store it accordingly. Check it with `python -m app.cli backup verify {made.path}`.")
    return 0 if made.ok else 1


def _backup_verify(args: argparse.Namespace) -> int:
    """Check a backup directory with no running instance (N-12): exit 0 every check passes, 1 one fails, 2 no manifest."""
    from app.storage import backup

    try:
        checks = backup.verify(Path(args.directory))
    except backup.BackupUnreadable as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for check in checks:
        print(f"{'PASS' if check.ok else 'FAIL':<9} {check.name}: {check.detail}")
    ok = all(check.ok for check in checks)
    print(f"{'VERIFIED' if ok else 'NOT VERIFIED':<9} {args.directory}")
    return 0 if ok else 1


def _backup_restore(args: argparse.Namespace) -> int:
    """Put a backup into a data directory on which no instance runs (N-12a): exit 0 restored, 1 the backup does not verify
    (nothing written), 2 refused (a running instance, a target that holds files without --replace, no manifest, Postgres)."""
    from app.storage import backup, running

    try:
        done = backup.restore(Path(args.directory), Path(args.data_dir) if args.data_dir else None, replace=args.replace)
    except backup.BackupNotVerified as e:
        for check in e.checks:
            print(f"{'PASS' if check.ok else 'FAIL':<9} {check.name}: {check.detail}")
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (backup.BackupUnreadable, backup.BackupRefused, running.InstanceRunning) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if done.set_aside:
        print(f"set aside {', '.join(done.set_aside)}: the database that was there, with its log. Remove them once the restored "
              "instance is checked.")
    head = done.manifest["database"]["custody_head"]
    print(f"RESTORED  {done.data_dir}: {done.files} file(s) of the backup of {done.manifest['created_at']}; the database passes its "
          f"integrity check; custody head {head['seq']}.")
    for name, entry in done.manifest["keys"].items():
        if entry.get("source") in ("environment", "derived"):
            print(f"          The {name.replace('_', ' ')} is not in the backup ({entry['source']}, fingerprint {entry['fingerprint']}): "
                  "start with the value the instance ran with.")
    print("          Start the instance, then: python -m app.cli audit verify; python -m app.cli store verify.")
    return 0


def _keys_adopt_derived(args: argparse.Namespace) -> int:
    """The first prod start of this build on the data of an earlier one (N-17a; ops re-review O2-01): the signing key that
    build derived from the API key is written to `data/keys/signing.key`. Exit 0 written, 1 refused (nothing written), 2 the
    named variable is not set or the custody event could not be stored."""
    from sqlalchemy.exc import SQLAlchemyError

    from app.audit import chain as audit
    from app.core.config import settings
    from app.storage import keyguard
    from app.storage.db import init_db

    named = args.api_key_env or "VERITAS_API_KEY"
    api_key = settings.api_key if args.api_key_env is None else os.environ.get(args.api_key_env)
    if args.api_key_env is not None and not api_key:
        print(f"error: the environment variable {named} is not set", file=sys.stderr)
        return 2
    try:
        init_db()  # a command that writes brings the database to the head, as a start does
        done = keyguard.adopt_derived(api_key, named=named)
    except keyguard.NotAdopted as refused:
        print(f"error: {refused}", file=sys.stderr)
        return 1
    except (SQLAlchemyError, audit.TailMoved, OSError) as e:
        print(f"error: the key file and its custody event could not be written ({type(getattr(e, 'orig', e)).__name__}); "
              "nothing was changed", file=sys.stderr)
        return 2
    signed = done.signed
    print(f"WRITTEN   {done.path} (mode 0600), fingerprint {done.fingerprint}: the signing key an earlier build derived from "
          f"{named}.")
    print(f"          {signed.conclusions} conclusion(s), {signed.signoffs} sign-off(s) and {signed.webhooks} webhook(s) verify "
          f"under it; custody event {done.seq} (signing_key_adopted) records the act.")
    print("          The key is now this file and no longer follows the API key: start the instance, then take a backup, "
          "which holds the file.")
    print("          A key derived from the API key lets a holder of a signed record test API-key candidates: change "
          "VERITAS_API_KEY once the instance runs. The review records stay valid.")
    return 0


def _fusers_status(args: argparse.Namespace) -> int:
    """What serves each kind (D-57): the release artefact file with its version, sha256, directory and review state, and
    whether it serves under the profile or the uncalibrated fallback does, with the reason."""
    from app.engine.fusion import registry as fusers
    from app.engine.pipelines import default_profile
    from app.engine.registry import get_registry

    profile = args.profile or default_profile().value
    status = fusers.fuser_status(get_registry(), profile)
    release = fusers.verify_release()
    if args.json:
        print(json.dumps({"profile": profile, "kinds": status, "release_directory": str(fusers.RELEASE_DIR),
                          "release_problems": release}, indent=2))
        return 1 if release else 0
    rows = []
    for kind, row in status.items():
        artefact = row["artefact"]
        if artefact is None:
            rows.append((kind, "-", "-", "-", "-", "fallback"))
            continue
        review = ("-" if artefact["review"] is None else f"reviewed by {artefact['reviewed_by']} ({artefact['review_date']})"
                  if artefact["review"] == "reviewed" else "pending human review")
        rows.append((kind, artefact["version"], (artefact["sha256"] or "unreadable")[:12], artefact["directory"], review, row["mode"]))
    _table(("KIND", "VERSION", "SHA256", "DIRECTORY", "REVIEW", f"SERVES ({profile})"), rows)
    for kind, row in status.items():
        if row["mode"] == "fallback":
            print(f"{kind}: uncalibrated fallback: {row['reason']}")
        if row["artefact"] is not None and row["artefact"].get("review_problem"):
            print(f"{kind}: review record ignored: {row['artefact']['review_problem']}")
        for shadow in row["shadowed"]:
            print(f"{kind}: {fusers.shadow_sentence(shadow)}")
    print(f"packaged release directory {fusers.RELEASE_DIR}: " + ("matches its manifest" if not release else "; ".join(release)))
    return 1 if release else 0


def _reviewed_artefact(args: argparse.Namespace) -> dict[str, Any] | None:
    """The release artefact of the kind a review command names, or None with the refusal printed."""
    from app.engine.fusion import registry as fusers

    artefact = fusers.release_artefact(args.modality)
    if artefact is None or artefact["sha256"] is None:
        print(f"error: no readable fuser artefact for {args.modality}", file=sys.stderr)
        return None
    if args.sha256 and not artefact["sha256"].startswith(args.sha256.lower()):
        print(f"error: {artefact['file']} has sha256 {artefact['sha256'][:12]}…, not {args.sha256}", file=sys.stderr)
        return None
    return artefact


def _fusers_review(args: argparse.Namespace) -> int:
    """Record a human review of the served fuser of one kind (D-57): a sidecar beside the data directory's artefacts that
    names the artefact's sha256, and its custody event (N-9). The artefact is not changed. A review that replaces another
    is recorded as replacing it; when the custody log cannot take the event, the sidecar is put back as it was."""
    from app.core.config import settings
    from app.engine.fusion import registry as fusers
    from app.engine.fusion import review_log

    artefact = _reviewed_artefact(args)
    if artefact is None:
        return 2
    target = Path(artefact["directory"]) / artefact["file"]
    sidecar = Path(settings.fusion_dir) / fusers.review_name(artefact["file"])
    before = sidecar.read_bytes() if sidecar.is_file() else None
    try:
        path = fusers.write_review(target, artefact["sha256"], reviewer=args.reviewer, statement=args.statement)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    subject = review_log.Subject(kind=args.modality, version=artefact["version"], file=artefact["file"], sha256=artefact["sha256"])
    try:
        review, problem = fusers.read_review(target, artefact["sha256"])
        if review is None:
            raise RuntimeError(f"the review record written to {path} does not count: {problem}")
        event = review_log.reconcile(subject, review, source=review_log.COMMAND)
    except Exception as e:  # the review and its custody event are recorded together or not at all
        if before is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(before)
        print(f"error: the custody log could not record the review, and the review record was put back as it was: {e}", file=sys.stderr)
        return 1
    print(f"recorded the review of {artefact['file']} (sha256 {artefact['sha256']}) by {review.reviewer} in {review.path}")
    print(f"custody event {event['seq']} ({event['action']})" if event is not None
          else "The custody log already names this review; no event was added.")
    print("Analyses from now on carry this reviewer; results published earlier keep what they recorded.")
    return 0


def _fusers_withdraw_review(args: argparse.Namespace) -> int:
    """Withdraw the review of the served fuser of one kind (N-9, D-57): the sidecar is removed and `fuser_review_withdrawn` is
    recorded, together or not at all. A sidecar that was already deleted by hand gets its event here. A review record inside
    the packaged release directory is not removed by this command."""
    from app.engine.fusion import registry as fusers
    from app.engine.fusion import review_log

    reason = (args.reason or "").strip() or None
    if reason is not None and len(reason) > review_log.REASON_MAX:
        print(f"error: --reason is longer than {review_log.REASON_MAX} characters", file=sys.stderr)
        return 2
    artefact = _reviewed_artefact(args)
    if artefact is None:
        return 2
    target = Path(artefact["directory"]) / artefact["file"]
    subject = review_log.Subject(kind=args.modality, version=artefact["version"], file=artefact["file"], sha256=artefact["sha256"])
    review, _ = fusers.read_review(target, artefact["sha256"])
    held: Path | None = None
    if review is not None:
        if review.path.parent.resolve() == fusers.RELEASE_DIR.resolve():
            print(f"error: the review record {review.path} is part of the packaged release; it is withdrawn by a release without it",
                  file=sys.stderr)
            return 2
        held = review.path.with_name(review.path.name + ".withdrawing")
    try:
        if review is not None and held is not None:
            # a record this instance has not put in the log yet (copied in, never used) is recorded before it is withdrawn
            review_log.reconcile(subject, review, source=review_log.FOUND_ON_WITHDRAWAL)
            os.replace(review.path, held)
        remaining, _ = fusers.read_review(target, artefact["sha256"])  # another record of this artefact may count now
        event = review_log.reconcile(subject, remaining, source=review_log.COMMAND, reason=reason)
    except Exception as e:  # the removal and its custody event are recorded together or not at all
        if review is not None and held is not None and held.is_file():
            os.replace(held, review.path)
        print(f"error: the custody log could not record the withdrawal, and the review record was left in place: {e}", file=sys.stderr)
        return 1
    if held is not None:
        held.unlink(missing_ok=True)
    if review is None and event is None:
        print(f"error: {artefact['file']} has no review to withdraw", file=sys.stderr)
        return 2
    if review is None:
        print(f"the review record of {artefact['file']} was already removed; its withdrawal is now recorded")
    else:
        print(f"withdrew the review of {artefact['file']} (sha256 {artefact['sha256']}) by {review.reviewer}; removed {review.path}")
    if event is not None:
        print(f"custody event {event['seq']} ({event['action']})")
    if remaining is not None:
        print(f"The review by {remaining.reviewer} in {remaining.path} counts now.")
    else:
        print("Analyses from now on say that the fusion model is pending human review; results published earlier keep what they recorded.")
    return 0


def _reproduce(args: argparse.Namespace) -> int:
    from app.reports import reproduce, verify

    try:
        result = reproduce.reproduce(Path(args.package), tolerance=args.tolerance, public_key_pem=_public_key(args.public_key),
                                     expect_key_id=args.expect_key_id)
    except (verify.PackageUnreadable, verify.PinInvalid, reproduce.ReproduceRefused, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return verify.EXIT_ERROR
    except (reproduce.PackageNotVerified, reproduce.SignerNotAuthenticated) as e:
        # Nothing was run. The exit codes are those of `verify`: 1 not verified, 3 no instance key pinned (N-11).
        if args.json:
            print(json.dumps({"ok": False, "status": "not_reproduced", "error": str(e), "verification": e.verification.as_dict()}, indent=2))
        elif isinstance(e, reproduce.SignerNotAuthenticated):
            _signer_not_authenticated(args.package, e.verification.key_id, e.verification.bundle)
        else:
            for check in e.verification.checks:
                if not check.ok:
                    print(f"FAIL      {check.name}: {check.detail}")
        print(f"error: {e}", file=sys.stderr)
        return verify.EXIT[e.verification.outcome]
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, default=str))
        return 0 if result.ok else 1
    if result.status != "done":
        print(f"RE-RUN    {result.status}: {result.error}")
        return 1
    for d in result.deltas:
        if not d.ok:
            delta = "" if d.delta is None else f" (Δ {d.delta:.6f})"
            print(f"DIFFERS   {d.item} {d.field}: packaged {d.packaged}, re-run {d.rerun}{delta}")
    largest = max((d.delta for d in result.deltas if d.delta is not None), default=0.0)
    print(f"{'REPRODUCED' if result.ok else 'DIFFERENT':<9} {sum(1 for d in result.deltas if not d.ok)} of {len(result.deltas)} "
          f"values outside tolerance {args.tolerance:g}; largest numeric Δ {largest:.6f}")
    return 0 if result.ok else 1


PRINCIPAL_ROLES = ("administrator", "examiner", "reviewer", "viewer", "api_client")
PASSWORD_SHOWN_ONCE = ("The password is shown once and is stored only as a hash. Hand it over in private: its user signs in with it "
                       "and replaces it at once (the account menu, \"Change password…\").")
KEY_SHOWN_ONCE = "The key is shown once and is stored only as a hash. Clients send it as X-API-Key."


def _principals_create(args: argparse.Namespace) -> int:
    """The first administrator of a server, and any principal after it, without an API call (OPS-04).

    A person gets a generated password, an api_client a generated API key; either is printed once and stored only as a hash."""
    import secrets

    from pydantic import ValidationError
    from sqlalchemy import select
    from sqlalchemy.exc import IntegrityError

    from app.api import auth
    from app.api.v2_auth import PrincipalCreate
    from app.audit import chain as audit
    from app.storage.db import PrincipalRow

    password = None if args.role == "api_client" else secrets.token_urlsafe(18)  # 24 characters
    try:
        wanted = PrincipalCreate(username=args.username, role=args.role, display_name=args.display_name, password=password)
    except ValidationError as refused:
        for error in refused.errors():
            print(f"error: {'.'.join(str(part) for part in error['loc'])}: {error['msg']}", file=sys.stderr)
        return 2
    if wanted.username in auth.RESERVED_USERNAMES:
        print(f"error: {wanted.username} names a built-in principal", file=sys.stderr)
        return 1
    def unit() -> tuple[str, str, str | None] | None:
        with audit.writing() as db:  # N-9: the account, its key and their `permission_changed` events are one transaction
            if db.scalars(select(PrincipalRow.id).where(PrincipalRow.username == wanted.username)).first() is not None:
                return None
            row = auth.new_principal(wanted.username, wanted.role, display_name=wanted.display_name, password=password)
            db.add(row)
            try:
                db.flush()
                key, key_id = auth.issue_api_key(db, row) if password is None else (None, None)
                db.flush()
            except IntegrityError:
                db.rollback()
                return None
            events = [{"action": "permission_changed", "actor_id": audit.SYSTEM_ACTOR, "object_ids": [row.id],
                       "details": {"change": "principal_created", "principal_id": row.id, "username": wanted.username,
                                   "role": wanted.role, "by": "command line"}}]
            if key_id is not None:
                events.append({"action": "permission_changed", "actor_id": audit.SYSTEM_ACTOR, "object_ids": [row.id],
                               "details": {"change": "api_key_issued", "principal_id": row.id, "key_id": key_id,
                                           "by": "command line"}})
            created = row.id, row.display_name, key
            audit.commit_with(db, events)
            return created

    created = audit.repeated(unit)
    if created is None:
        print(f"error: a principal named {wanted.username} already exists", file=sys.stderr)
        return 1
    principal_id, display_name, key = created
    print(f"created {wanted.role} {wanted.username} ({display_name}), id {principal_id}")
    if password is not None:
        print(f"password: {password}")
        print(PASSWORD_SHOWN_ONCE)
    else:
        print(f"api key: {key}")
        print(KEY_SHOWN_ONCE)
    return 0


def _sessions_ended(count: int) -> str:
    return "no open session" if count == 0 else f"{count} session{'' if count == 1 else 's'} ended"


def _principal_change(username: str, change) -> int:
    """One change of an existing principal from the command line, through the functions `PATCH /api/v2/principals/{id}` and
    `POST …/api-keys` use (`app.api.auth`).

    `change(db, row)` returns an error text (nothing is stored, exit 1), or the details of the `permission_changed` event and
    the lines to print once the change is committed. A secret is in the lines only, never in the details."""
    from sqlalchemy import select

    from app.audit import chain as audit
    from app.storage.db import PrincipalRow

    def unit() -> str | list[str]:
        with audit.writing() as db:  # N-9: the change and its `permission_changed` event are one transaction
            row = db.scalars(select(PrincipalRow).where(PrincipalRow.username == username)).first()
            if row is None:
                return f"no principal is named {username}; see `python -m app.cli principals list`"
            outcome = change(db, row)
            if isinstance(outcome, str):
                db.rollback()
                return outcome
            details, lines = outcome
            db.flush()
            if details["change"] == "principal_updated":  # the fields the API's event of the same change holds
                details = {**details, "role": row.role, "disabled": row.disabled_at is not None}
            audit.commit_with(db, [{"action": "permission_changed", "actor_id": audit.SYSTEM_ACTOR, "object_ids": [row.id],
                                    "details": {"principal_id": row.id, **details, "by": "command line"}}])
            return list(lines)

    done = audit.repeated(unit)
    if isinstance(done, str):
        print(f"error: {done}", file=sys.stderr)
        return 1
    for line in done:
        print(line)
    return 0


def _principals_set_password(args: argparse.Namespace) -> int:
    """A forgotten password: a generated one replaces it and every session of the principal ends."""
    import secrets

    from app.api import auth

    def change(db, row):
        if (refused := auth.password_rule(row.role, True)) is not None:
            return refused
        password = secrets.token_urlsafe(18)  # 24 characters
        ended = auth.set_password(db, row, password)
        state = "; the principal is disabled and cannot sign in until it is enabled" if row.disabled_at is not None else ""
        return ({"change": "principal_updated", "fields": ["password"], "sessions_ended": ended},
                [f"password of {row.username} ({row.display_name}) replaced; {_sessions_ended(ended)}{state}",
                 f"password: {password}", PASSWORD_SHOWN_ONCE])

    return _principal_change(args.username, change)


def _principals_disable(args: argparse.Namespace) -> int:
    """A leaver: the sessions end, and the password, the API key and the resource tokens are refused from now on. The row
    stays: the custody log and signed reports keep naming the principal."""
    from app.api import auth

    def change(db, row):
        if row.disabled_at is not None:
            return f"{row.username} is already disabled"
        ended = auth.set_disabled(db, row, True)
        return ({"change": "principal_updated", "fields": ["disabled"], "sessions_ended": ended},
                [f"disabled {row.username} ({row.display_name}); {_sessions_ended(ended)}; its password, API key and "
                 "download links are refused"])

    return _principal_change(args.username, change)


def _principals_enable(args: argparse.Namespace) -> int:
    from app.api import auth

    def change(db, row):
        if row.disabled_at is None:
            return f"{row.username} is not disabled"
        auth.set_disabled(db, row, False)
        key = (" Its API key works again: replace it with `principals api-key` unless it was kept safe."
               if row.api_key_hash is not None else "")
        return ({"change": "principal_updated", "fields": ["disabled"]},
                [f"enabled {row.username} ({row.display_name}); the password it had works again.{key}"])

    return _principal_change(args.username, change)


def _principals_set_role(args: argparse.Namespace) -> int:
    """A changed role ends the principal's sessions. An api_client that becomes a person gets a generated password."""
    import secrets

    from app.api import auth

    def change(db, row):
        if row.role == args.role:
            return f"{row.username} is already {args.role}"
        password = secrets.token_urlsafe(18) if row.role == "api_client" else None
        if (refused := auth.password_rule(args.role, password is not None or row.password_hash is not None)) is not None:
            return f"{refused} Create a separate api_client principal for a program."
        before = row.role
        ended = auth.set_role(db, row, args.role)
        fields, lines = ["role"], [f"{row.username} ({row.display_name}) is now {args.role} (was {before}); {_sessions_ended(ended)}"]
        if password is not None:
            auth.set_password(db, row, password)
            fields, lines = ["password", "role"], [*lines, f"password: {password}", PASSWORD_SHOWN_ONCE]
        return ({"change": "principal_updated", "fields": fields, "sessions_ended": ended}, lines)

    return _principal_change(args.username, change)


def _principals_api_key(args: argparse.Namespace) -> int:
    """A lost or exposed API key: a new key replaces it, and the old one is refused from this moment."""
    from app.api import auth

    def change(db, row):
        if row.disabled_at is not None:
            return f"{row.username} is disabled; enable the principal before issuing it an API key"
        replaced = row.api_key_hash is not None
        key, key_id = auth.issue_api_key(db, row)
        return ({"change": "api_key_issued", "key_id": key_id},
                [f"api key of {row.username} ({row.display_name}) {'replaced; the previous key is refused' if replaced else 'issued'}",
                 f"api key: {key}", KEY_SHOWN_ONCE])

    return _principal_change(args.username, change)


def _principals_list(_args: argparse.Namespace) -> int:
    from sqlalchemy import select

    from app.core import times
    from app.storage.db import PrincipalRow, get_session

    def day(moment) -> str:
        return f"{times.to_utc(moment):%Y-%m-%d}" if moment is not None else "-"

    with get_session() as db:
        rows = [(r.username, r.role, r.display_name, "disabled" if r.disabled_at is not None else "active",
                 "yes" if r.password_hash is not None else "no", "yes" if r.api_key_hash is not None else "no",
                 day(r.created_at), day(r.last_login_at), r.id)
                for r in db.scalars(select(PrincipalRow).order_by(PrincipalRow.created_at, PrincipalRow.username))]
    if not rows:
        print("no principals; create the first administrator with "
              "`python -m app.cli principals create --username NAME --role administrator`")
        return 0
    _table(("USERNAME", "ROLE", "NAME", "STATE", "PASSWORD", "API KEY", "CREATED", "LAST SIGN-IN", "ID"), rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="Veritas command-line tools")
    commands = parser.add_subparsers(dest="command", required=True)

    custody = commands.add_parser("custody", help="the custody log (audit hash chain)")
    custody_actions = custody.add_subparsers(dest="action", required=True)
    p_export = custody_actions.add_parser("export", help="stream the audit log (only a whole-chain export verifies offline)")
    p_export.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    p_export.add_argument("--case-id", default=None)
    p_export.add_argument("--out", default=None, help="file to write (default: stdout)")
    p_export.add_argument("--database-url", default=None, help="SQLAlchemy URL of a database copy (default: the configured database)")
    p_export.set_defaults(func=_custody_export)

    audit = commands.add_parser("audit", help="audit log verification")
    audit_actions = audit.add_subparsers(dest="action", required=True)
    p_audit_verify = audit_actions.add_parser("verify", help="re-hash the whole chain, from a database or an exported JSONL file")
    p_audit_verify.add_argument("--jsonl", default=None, help="a whole-chain export from `custody export --format jsonl`")
    p_audit_verify.add_argument("--database-url", default=None)
    p_audit_verify.set_defaults(func=_audit_verify)

    p_package_verify = commands.add_parser("verify", help="verify an evidence package offline against the instance key: hashes, "
                                                          "completeness, signatures, custody")
    p_package_verify.add_argument("package", help="veritas-evidence-….zip")
    _pin_arguments(p_package_verify)
    source = p_package_verify.add_mutually_exclusive_group()
    source.add_argument("--source-image", metavar="IMAGE", help="compare crops in a public image package with a local original")
    source.add_argument("--source-audio", metavar="MEDIA", help="compare clip samples and timestamps with a local audio or video original")
    source.add_argument("--source-video", metavar="VIDEO", help="compare frames, crops and mouth patches with a local video original")
    p_package_verify.add_argument("--json", action="store_true")
    p_package_verify.set_defaults(func=_verify)

    p_compare_image = commands.add_parser("compare-image", help="inspect local differences against a separately supplied aligned image")
    p_compare_image.add_argument("reference", help="earlier reference image")
    p_compare_image.add_argument("submitted", help="image to compare with the reference")
    p_compare_image.add_argument("--evidence-dir", metavar="DIRECTORY", help="write the mask, source crops and JSON to a new folder")
    p_compare_image.set_defaults(func=_compare_image)

    database = commands.add_parser("db", help="database administration")
    db_actions = database.add_subparsers(dest="action", required=True)
    p_grants = db_actions.add_parser("grants", help="Postgres grants for the application role: custody tables INSERT/SELECT only")
    p_grants.add_argument("--app-role", required=True, help="the role the service connects as (must own no table)")
    p_grants.add_argument("--database-url", default=None, help="an owner connection, needed with --apply")
    p_grants.add_argument("--apply", action="store_true", help="run the statements in one transaction instead of only printing them")
    p_grants.set_defaults(func=_db_grants)
    p_setup = db_actions.add_parser(
        "setup", help="Postgres, as the owner: migrate to the head, set the database's default time zone to UTC, create the "
                      "application role (or set its password) and apply its grants; run it again after every upgrade")
    p_setup.add_argument("--app-role", required=True, help="the role the service connects as (must own no table)")
    p_setup.add_argument("--database-url", default=None, help="an owner connection (default: VERITAS_DATABASE_URL)")
    p_setup.add_argument("--password-env", default=None, metavar="NAME",
                         help="the environment variable that holds the role's password: the role is created with it, or "
                              "its password is set to it; without this option the role must exist")
    p_setup.set_defaults(func=_db_setup)

    retention = commands.add_parser("retention", help="the retention rule (D-55)")
    retention_actions = retention.add_subparsers(dest="action", required=True)
    p_purge = retention_actions.add_parser(
        "purge", help="purge the revision chains the rule allows: a case past its retention date, and unsigned quick analyses "
                      "older than VERITAS_RETENTION_DAYS; held, retained, signed, in-progress and named-case records are kept")
    p_purge.add_argument("--dry-run", action="store_true", help="print each chain with its decision and reason; change nothing")
    p_purge.set_defaults(func=_retention_purge)

    principals = commands.add_parser("principals", help="the people and clients who may sign in")
    principal_actions = principals.add_subparsers(dest="action", required=True)
    p_principal = principal_actions.add_parser(
        "create", help="create a principal and print its generated password once (an api_client: its API key); "
                       "the first administrator of a server is made this way")
    p_principal.add_argument("--username", required=True, help="1 to 64 letters, digits, dots, underscores and hyphens")
    p_principal.add_argument("--role", required=True, choices=PRINCIPAL_ROLES)
    p_principal.add_argument("--display-name", default=None, help="the name reports and the custody log print (default: the username)")
    p_principal.set_defaults(func=_principals_create)
    p_principals = principal_actions.add_parser("list", help="every principal with its role and state; no secret is printed")
    p_principals.set_defaults(func=_principals_list)
    for action, func, text in (
            ("set-password", _principals_set_password, "replace a forgotten password by a generated one, printed once; the sessions end"),
            ("disable", _principals_disable, "a leaver: end the sessions and refuse the password, the API key and the download links"),
            ("enable", _principals_enable, "let a disabled principal sign in again"),
            ("set-role", _principals_set_role, "change the role; the sessions end"),
            ("api-key", _principals_api_key, "issue an API key, printed once; it replaces a lost or exposed one")):
        p_action = principal_actions.add_parser(action, help=text)
        p_action.add_argument("--username", required=True, help="the principal, as `principals list` prints it")
        if action == "set-role":
            p_action.add_argument("--role", required=True, choices=PRINCIPAL_ROLES)
        p_action.set_defaults(func=func)

    store = commands.add_parser("store", help="the evidence object store")
    store_actions = store.add_subparsers(dest="action", required=True)
    p_gc = store_actions.add_parser("gc", help="unlink objects that no row references (nothing while a worker holds a job)")
    p_gc.add_argument("--dry-run", action="store_true", help="list the unreferenced objects; change nothing")
    p_gc.set_defaults(func=_store_gc)
    p_store_verify = store_actions.add_parser("verify", help="re-hash every store object the database references")
    p_store_verify.set_defaults(func=_store_verify)

    backups = commands.add_parser("backup", help="back up an instance (SQLite), check a backup, restore one")
    backup_actions = backups.add_subparsers(dest="action", required=True)
    p_backup_create = backup_actions.add_parser("create", help="database, referenced objects, per-analysis files, keys and a "
                                                               "manifest into a new directory")
    p_backup_create.add_argument("directory", help="a new or empty directory; it will hold the keys")
    p_backup_create.set_defaults(func=_backup_create)
    p_backup_verify = backup_actions.add_parser("verify", help="check a backup: file hashes, database, objects, custody, keys")
    p_backup_verify.add_argument("directory")
    p_backup_verify.set_defaults(func=_backup_verify)
    p_backup_restore = backup_actions.add_parser("restore", help="put a backup that verifies into a data directory on which no "
                                                                 "instance runs")
    p_backup_restore.add_argument("directory", help="a backup made by `backup create`")
    p_backup_restore.add_argument("--data-dir", default=None, help="the target (default: the configured data directory)")
    p_backup_restore.add_argument("--replace", action="store_true",
                                  help="restore over a data directory that holds files; its database is set aside, not deleted")
    p_backup_restore.set_defaults(func=_backup_restore)

    keys = commands.add_parser("keys", help="the keys of the data directory")
    key_actions = keys.add_subparsers(dest="action", required=True)
    p_adopt = key_actions.add_parser(
        "adopt-derived", help="an instance an earlier build ran in prod without a signing key of its own: write the key that "
                              "build derived from the API key to data/keys/signing.key, when the signed records verify under it")
    p_adopt.add_argument("--api-key-env", default=None, metavar="NAME",
                         help="the environment variable that holds the API key the earlier build ran with, when "
                              "VERITAS_API_KEY has changed since (default: VERITAS_API_KEY)")
    p_adopt.set_defaults(func=_keys_adopt_derived)

    fusers = commands.add_parser("fusers", help="the calibrated fusion artefacts and their review records")
    fuser_actions = fusers.add_subparsers(dest="action", required=True)
    p_fuser_status = fuser_actions.add_parser("status", help="per kind: version, sha256, directory, review state and what serves")
    p_fuser_status.add_argument("--profile", choices=("quick", "full", "deep"), default=None, help="default: the served default profile")
    p_fuser_status.add_argument("--json", action="store_true")
    p_fuser_status.set_defaults(func=_fusers_status)
    p_fuser_review = fuser_actions.add_parser(
        "review", help="record a named human review of the served fuser of one kind (a sidecar; the artefact is not changed)")
    p_fuser_review.add_argument("--modality", required=True, choices=("image", "video", "audio", "document"))
    p_fuser_review.add_argument("--reviewer", required=True, help="the reviewer's name")
    p_fuser_review.add_argument("--statement", required=True, help="what was reviewed and accepted")
    p_fuser_review.add_argument("--sha256", default=None, help="refuse unless the artefact's sha256 starts with this")
    p_fuser_review.set_defaults(func=_fusers_review)
    p_fuser_withdraw = fuser_actions.add_parser(
        "withdraw-review", help="withdraw the review of the served fuser of one kind (the sidecar is removed; the custody log records it)")
    p_fuser_withdraw.add_argument("--modality", required=True, choices=("image", "video", "audio", "document"))
    p_fuser_withdraw.add_argument("--sha256", default=None, help="refuse unless the artefact's sha256 starts with this")
    p_fuser_withdraw.add_argument("--reason", default=None, help="why the review is withdrawn (kept in the custody event)")
    p_fuser_withdraw.set_defaults(func=_fusers_withdraw_review)

    p_reproduce = commands.add_parser("reproduce", help="re-run a packaged analysis on its original and diff the raw scores")
    p_reproduce.add_argument("package", help="a package exported with the original (source/)")
    p_reproduce.add_argument("--tolerance", type=float, default=0.001, help="numeric tolerance (0.02 between MPS and CPU)")
    _pin_arguments(p_reproduce)
    p_reproduce.add_argument("--json", action="store_true")
    p_reproduce.set_defaults(func=_reproduce)

    models = commands.add_parser("models", help="detector and weight manifest")
    actions = models.add_subparsers(dest="action", required=True)

    p_list = actions.add_parser("list", help="models, licence classes and what the licence policy enables")
    p_list.add_argument("--json", action="store_true")
    p_list.add_argument("--policy", default=None, help="licence policy preset or JSON/YAML (default: VERITAS_LICENCE_POLICY)")
    p_list.add_argument("--manifest", default=None)
    p_list.set_defaults(func=_models_list)

    p_verify = actions.add_parser("verify", help="check manifest invariants; --installed also hashes installed weights")
    p_verify.add_argument("--installed", action="store_true")
    p_verify.add_argument("--force", action="store_true", help="re-hash even when the signed verify record is fresh")
    p_verify.add_argument("--manifest", default=None)
    p_verify.set_defaults(func=_models_verify)

    p_fetch = actions.add_parser("fetch", help="download and verify pinned weights (refused when VERITAS_OFFLINE=true)")
    p_fetch.add_argument("model_ids", nargs="*",
                         help="exact model IDs (no dependency expansion); use --enabled / --profile for complete detectors")
    p_fetch.add_argument("--all", action="store_true")
    p_fetch.add_argument("--enabled", action="store_true", help="models and dependencies of every detector the licence policy enables")
    p_fetch.add_argument("--profile", choices=("quick", "full", "deep"), default=None,
                         help="models and dependencies of enabled detectors in this profile (implies --enabled)")
    p_fetch.add_argument("--dry-run", action="store_true")
    p_fetch.add_argument("--no-record", action="store_true",
                         help="hash every file and leave no verify record: for a fetch container without the data volume, "
                              "whose key signs that record (the processes that load the weights keep their own)")
    p_fetch.add_argument("--manifest", default=None)
    p_fetch.set_defaults(func=_models_fetch)

    p_pin = actions.add_parser("pin", help="resolve revision and per-file sha256 into models.yaml (prints a diff)")
    p_pin.add_argument("model_id")
    p_pin.add_argument("--write", action="store_true")
    p_pin.add_argument("--manifest", default=None)
    p_pin.set_defaults(func=_models_pin)

    p_footprint = actions.add_parser("footprint", help="declared vs measured model memory and device pins")
    p_footprint.add_argument("--manifest", default=None)
    p_footprint.set_defaults(func=_models_footprint)

    p_plan = actions.add_parser(
        "plan", help="what a profile needs of the model memory budget and which detectors cannot run under it "
                     "(exit code 1 when one cannot); loads no weights")
    p_plan.add_argument("--profile", choices=("quick", "full", "deep"), default=None,
                        help="default: VERITAS_DEFAULT_PROFILE (auto: full on an accelerator, quick on CPU)")
    p_plan.add_argument("--budget-mb", type=float, default=None,
                        help="default: VERITAS_MODEL_MEMORY_BUDGET_MB (auto: the rule on this machine's or container's memory)")
    p_plan.add_argument("--device", choices=("cpu", "mps", "cuda"), default=None, help="default: the device this server uses")
    p_plan.add_argument("--json", action="store_true")
    p_plan.set_defaults(func=_models_plan)
    return parser


def _start_refused(error: BaseException) -> bool:
    """Whether the error is one of the refusals of a start (`storage/refusals.py`). A command that loaded none of the
    modules that raise them cannot have met one, and none is loaded for the question: `verify` runs without the
    server's database modules."""
    if not any(name in sys.modules for name in ("app.storage.migrations", "app.storage.keyguard", "app.storage.running")):
        return False
    from app.storage import refusals

    return isinstance(error, refusals.REFUSALS)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as e:
        # N-19e (E-17): a refusal of the start is one sentence, on the command line too (it was a traceback whose last
        # line was the sentence)
        if not _start_refused(e):
            raise
        print(f"error: {e}", file=sys.stderr)
        return 1
