"""A missed live sample cannot be relabeled as a clean worker exit."""

from types import SimpleNamespace

import psutil
import pytest

from scripts.data_resource_guard_v21 import sample


class Root:
    def children(self, recursive):
        assert recursive
        return []

    def memory_info(self):
        raise psutil.NoSuchProcess(123)


@pytest.mark.parametrize("code", [0, 7])
def test_verified_process_exit_is_retained_with_its_actual_status(code):
    memory, event = sample(Root(), SimpleNamespace(poll=lambda: code))
    assert memory == 0 and event == {"type": "NoSuchProcess", "child_exit_code": code, "classification": "verified_terminal_transition"}


def test_live_sampling_failure_stays_an_error():
    with pytest.raises(psutil.NoSuchProcess):
        sample(Root(), SimpleNamespace(poll=lambda: None))


def test_actual_tree_sample_counts_all_members():
    root = SimpleNamespace(children=lambda recursive: [SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=7))], memory_info=lambda: SimpleNamespace(rss=11))
    memory, event = sample(root, SimpleNamespace(poll=lambda: None))
    assert memory == 18 and event is None
