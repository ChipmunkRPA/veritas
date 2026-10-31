"""Text a request may carry (N-10; Postgres QA PG-05).

A NUL character is no text. SQLite stores it and PostgreSQL refuses the statement (`psycopg.DataError`, which was a 500), so it
is refused before any statement, in one place for each way a string arrives:

- a JSON body: every request model is a `RequestModel`, which looks at every string of the body, keys and nested values
  included, before the fields are validated (422);
- a form field: its type is `Text` (422);
- a path or query value: `app.api.security.refuse_nul`, asked by every route before anything is looked up (404);
- what none of these names: the `sqlalchemy.exc.DataError` handler of `app.api.v2` (422).

The sign-in is the one body that is not a `RequestModel`: a name that holds a NUL is no account, and its answer is the 401 of
any other failed sign-in.
"""
from __future__ import annotations

from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator

NUL = "\x00"
REFUSAL = "Text must not hold a NUL character (U+0000)."


def holds_nul(value: object) -> bool:
    return isinstance(value, str) and NUL in value


def any_nul(data: Any) -> bool:
    """Whether a string anywhere in parsed JSON holds a NUL. Iterative: the depth of a body is the client's choice."""
    pending = [data]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, list | tuple):
            pending.extend(value)
        elif holds_nul(value):
            return True
    return False


def _checked(value: str) -> str:
    if holds_nul(value):
        raise ValueError(REFUSAL)
    return value


Text = Annotated[str, AfterValidator(_checked)]  # the string type of a form field


class RequestModel(BaseModel):
    """The base of every JSON request model. A number that is not finite is refused (N-14b, S2-03): `1e999`, which a JSON
    reader takes for infinity, has no place in a request and no canonical form in a record."""

    model_config = ConfigDict(allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def _no_nul(cls, data: Any) -> Any:
        if any_nul(data):
            raise ValueError(REFUSAL)
        return data
