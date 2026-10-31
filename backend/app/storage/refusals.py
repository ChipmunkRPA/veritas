"""The refusals of a start (N-19e; Postgres re-run at a511ecf, E-17).

A start that the service refuses raises inside the server's lifespan, and the server library logs the traceback as one
line that ends with the sentence (9,856 characters in the reviewer's run for the migration that the role may not make);
on the command line the same refusal was a traceback whose last line was the sentence. The runbook says that the service
stops with one sentence.

The API and the worker log one event `startup.refused` at error level, with the sentence as `reason`, and then let the
exception leave as before. A command of the command line prints `error:` and the sentence and ends with exit code 1
(`cli.main`).
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.storage.keyguard import KeyMissing
from app.storage.migrations import LegacySchemaError, MigrationOwnerError
from app.storage.running import RestoreInProgress

REFUSALS: tuple[type[Exception], ...] = (MigrationOwnerError, LegacySchemaError, KeyMissing, RestoreInProgress)


@contextmanager
def logged(log: Any) -> Iterator[None]:
    """The steps of a start that can refuse it: a refusal is logged once, as `startup.refused`, and raised as it is."""
    try:
        yield
    except REFUSALS as e:
        log.error("startup.refused", reason=str(e))
        raise
