"""Serve built assets and the browser router's page URLs without hiding missing resources."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

_PAGE_FAMILIES = {"analyses", "cases", "batches", "models"}
_SINGLE_PAGES = {"upload", "demo", "design", "sign-in"}


def _sane(path: str) -> list[str] | None:
    """The segments of a decoded path without traversal, control characters or empty parts, else None."""
    if not path.startswith("/") or "\\" in path or any(ord(char) < 32 or ord(char) == 127 for char in path):
        return None
    parts = path.removeprefix("/").removesuffix("/").split("/")
    return None if any(part in {"", ".", ".."} for part in parts) else parts


def is_page_path(path: str) -> bool:
    """Allow only BrowserRouter pages, checking the decoded path before filesystem normalization."""
    parts = _sane(path)
    return parts is not None and (parts[0] in _PAGE_FAMILIES or (len(parts) == 1 and parts[0] in _SINGLE_PAGES))


def is_lost_navigation(scope: Scope) -> bool:
    """A browser navigating to an address that is no page (L-7): a GET or HEAD that asks for HTML by name, outside the API and
    asset namespaces, whose last segment is not a file name. It gets the shell with status 404, so the workspace can say the page
    does not exist; every other missing resource keeps its plain 404."""
    if scope.get("method") not in ("GET", "HEAD"):
        return False
    accept = next((value.decode("latin-1") for name, value in scope.get("headers") or [] if name == b"accept"), "")
    if "text/html" not in accept.lower():
        return False
    parts = _sane(scope["path"])
    if parts is None or parts[0] in {"api", "assets"} or any(part.startswith(".") for part in parts):
        return False
    return "." not in parts[-1]


class SPAStaticFiles(StaticFiles):
    """Keep Starlette's asset handling; use the shell only for known page families, and for a lost navigation with status 404."""

    def __init__(self, directory: Path):
        self._index_path = os.path.realpath(directory / "index.html")
        super().__init__(directory=directory, html=True)

    async def get_response(self, path: str, scope: Scope) -> Response:
        page = is_page_path(scope["path"])
        lost = not page and is_lost_navigation(scope)
        try:
            response = await super().get_response(path, scope)
        except HTTPException as error:
            if error.status_code != 404 or not (page or lost):
                raise
        else:
            # HTML mode can return a custom 404 page instead of raising an exception.
            if response.status_code != 404 or not (page or lost):
                return response
        # The first lookup preserves method errors and path/symlink containment checks.
        # Delegating the shell lookup also retains HEAD, range and conditional requests.
        shell = await super().get_response("index.html", scope)
        if lost and shell.status_code == 200:
            shell.status_code = 404  # the address is no page: the shell renders its not-found page, and the status says so
        return shell

    def file_response(self, full_path: str | os.PathLike[str], stat_result: os.stat_result,
                      scope: Scope, status_code: int = 200) -> Response:
        response = super().file_response(full_path, stat_result, scope, status_code)
        if os.fspath(full_path) == self._index_path:
            # Apply after Starlette's conditional handling so 304 responses carry this too.
            response.headers["Cache-Control"] = "no-cache"
        return response


def mount_frontend(app: FastAPI, directory: Path) -> None:
    if not directory.is_dir() or not (directory / "index.html").is_file():
        raise ValueError(f"Frontend build must contain index.html: {directory}")
    app.mount("/", SPAStaticFiles(directory), name="frontend")
