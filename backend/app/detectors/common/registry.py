"""Legacy registry backed by ModelManager (ARCHITECTURE §2.5).

Inference uses `borrow` to hold its weight reservation throughout execution. `get` remains a load-only
compatibility/warmup getter. Models are shared and can be reloaded after an unpinned eviction.
"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any


class ModelRegistry:
    def borrow(self, name: str, loader: Callable[[], Any]) -> AbstractContextManager[Any]:
        from app.engine.models import get_model_manager

        return get_model_manager().borrow_v1(name, loader)

    def get(self, name: str, loader: Callable[[], Any]) -> Any:
        from app.engine.models import get_model_manager

        return get_model_manager().get_v1(name, loader)

    def loaded(self) -> dict[str, dict]:
        from app.engine.models import get_model_manager

        return get_model_manager().v1_loaded()

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded()


registry = ModelRegistry()
