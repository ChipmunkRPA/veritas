"""Exact resource observations for the explicit authored comparison; no model import."""

import math
import shutil
import time

import psutil

RESERVE = 1536 * 1024**2
RSS_LIMIT = 6 * 1024**3
SECONDS = 900
START_RAM = 6 * 1024**3
START_DISK = 3 * 1024**3


def observe(folder, started, rss=0):
    return {"elapsed_s": time.monotonic() - started, "tree_rss_bytes": rss, "ram_available_bytes": psutil.virtual_memory().available, "disk_free_bytes": shutil.disk_usage(folder).free}


def violations(snapshot, *, initial=False):
    keys = ("tree_rss_bytes", "ram_available_bytes", "disk_free_bytes", "elapsed_s")
    if not all(k in snapshot for k in keys) or not all(type(snapshot[k]) is int and snapshot[k] >= 0 for k in keys[:3]) or type(snapshot["elapsed_s"]) not in (int, float) or not math.isfinite(snapshot["elapsed_s"]) or snapshot["elapsed_s"] < 0:
        raise ValueError("Complete finite nonnegative resource observations required")
    checks = {
        "tree_rss_limit": snapshot["tree_rss_bytes"] > RSS_LIMIT,
        "ram_reserve": snapshot["ram_available_bytes"] < RESERVE,
        "disk_reserve": snapshot["disk_free_bytes"] < RESERVE,
        "elapsed_time_limit": snapshot["elapsed_s"] > SECONDS,
    }
    if initial:
        checks.update({"initial_ram_headroom": snapshot["ram_available_bytes"] < START_RAM, "initial_disk_headroom": snapshot["disk_free_bytes"] < START_DISK})
    return [name for name, failed in checks.items() if failed]
