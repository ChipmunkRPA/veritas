"""Distinguish verified descendant exits from loss of a live worker sample."""
from __future__ import annotations

import psutil


def exited(process):
    try:
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return True
        # macOS can report task memory unavailable just before process status
        # changes. Wait for that exact descendant, never silently ignore it.
        try:
            process.wait(timeout=0.25)
            return True
        except psutil.TimeoutExpired:
            return False
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return True


def sample_tree(root, child):
    events = []
    try:
        rss = root.memory_info().rss
        # Recursive enumeration itself can race with an exiting grandchild.
        # Re-enumerate only after verifying that exact descendant is gone.
        for attempt in range(3):
            try:
                descendants = root.children(recursive=True)
                break
            except (psutil.NoSuchProcess, psutil.ZombieProcess) as error:
                if error.pid == root.pid or error.pid is None:
                    raise
                try:
                    process = psutil.Process(error.pid)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    process = None
                if process is not None and not exited(process):
                    raise
                events.append({"pid": error.pid, "type": type(error).__name__,
                               "classification": "verified_descendant_exit_during_enumeration"})
                if attempt == 2:
                    raise RuntimeError("Cannot obtain complete live descendant enumeration") from error
        for process in descendants:
            try:
                rss += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.ZombieProcess) as error:
                if not exited(process):
                    raise
                events.append({"pid": process.pid, "type": type(error).__name__,
                               "classification": "verified_descendant_exit_during_memory_sample"})
        return rss, events, False
    except (psutil.NoSuchProcess, psutil.ZombieProcess) as error:
        status = child.poll()
        if status is None:
            raise
        events.append({"pid": root.pid, "type": type(error).__name__, "child_exit_code": status,
                       "classification": "verified_worker_terminal_transition"})
        return 0, events, True
