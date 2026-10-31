"""Times are UTC on every engine (D-72 "Time"; BUILD_SPEC N-4b).

SQLite returns a stored time without a zone, and it was written in UTC. Postgres returns it in the zone of the session, which
`db.make_engine` pins to UTC. A time is converted to UTC before it is stored, hashed or printed, so both engines print one
form (`…Z`), and a date sent by a client carries an offset and lies in a range every engine and driver can read back.
"""
from __future__ import annotations

from datetime import UTC, datetime

MIN_YEAR, MAX_YEAR = 1970, 9000  # of a client date, after conversion to UTC


def to_utc(moment: datetime | None) -> datetime | None:
    """The same instant in UTC. A value without a zone is a stored UTC time."""
    if moment is None:
        return None
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def iso_z(moment: datetime | None) -> str | None:
    """ISO 8601 in UTC with the suffix `Z`, as the response models print an aware UTC time."""
    utc = to_utc(moment)
    return None if utc is None else utc.isoformat().replace("+00:00", "Z")


def micros(moment: datetime | None) -> int:
    """Microseconds since 1970 in UTC: an order on the time itself, not on its text. No time sorts first."""
    utc = to_utc(moment)
    if utc is None:
        return -1
    delta = utc - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def client_date(name: str, value: datetime | None) -> datetime | None:
    """A date a client sent, in UTC. Raises ValueError, in plain words, for a value without an offset or outside the range."""
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} needs a UTC offset, for example 2026-09-11T09:12:00Z.")
    try:
        utc = value.astimezone(UTC)
    except OverflowError:  # the first or the last day a datetime can hold, moved across it by its offset
        utc = None
    if utc is None or not MIN_YEAR <= utc.year <= MAX_YEAR:
        raise ValueError(f"{name} must lie between the years {MIN_YEAR} and {MAX_YEAR}.")
    return utc
