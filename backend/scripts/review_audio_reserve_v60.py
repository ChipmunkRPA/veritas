"""Additive alias/coverage review of the recorded reserve audit; no inference."""
from __future__ import annotations

import json
from collections import Counter

from scripts.audio_history_v58 import row_keys
from scripts.audio_reserve_audit_v60 import BASE, TRIAL, digest, matches, write


def review():
    source = json.loads((BASE / "audio-v9/source-plan.json").read_text())
    reserved = [row for row in source["sources"] if row["role"] == "confirmation"]
    consumed = [row for row in source["sources"] if row["role"] != "confirmation"]
    keys = [set().union(*(row_keys(row) for row in rows)) for rows in (reserved, consumed)]
    common = keys[0] & keys[1]
    coverage = {name: {kind: sum(any(key.startswith(kind + ":") for key in row_keys(row)) for row in rows)
                       for kind in ("bytes", "pcm", "voice", "text", "prefix8")}
                for name, rows in (("reserve", reserved), ("consumed", consumed))}
    # The first audit's literal transcript/source-prefix fields are absent in
    # reserve rows. This separately normalizes actual text and PCM field aliases;
    # it does not reinterpret those literal zero overlaps as complete coverage.
    aliases = {value for row in reserved for key, value in row.items()
               if key.endswith("pcm_sha256") and isinstance(value, str)}
    bindings = json.loads((TRIAL / "input-bindings.json").read_text())
    alias_matches, screening = [], []
    for relative, expected in bindings.items():
        path = BASE / relative
        if digest(path) != expected:
            raise ValueError("Historical metadata changed before additive review")
        with path.open() as stream:
            values = enumerate(stream, 1) if path.suffix == ".jsonl" else [(None, stream.read())]
            for line, raw in values:
                for found in matches(json.loads(raw), aliases):
                    item = {"path": relative, "line": line, **found}
                    alias_matches.append(item)
                    if found["screening_fields"]:
                        screening.append(item)
    write(TRIAL / "additional-alias-matches.json", alias_matches)
    result = {"schema": "chipmunk.audio-reserve-alias-review/60",
              "original_summary_sha256": digest(TRIAL / "summary.json"),
              "review_code_sha256": digest(__import__("pathlib").Path(__file__)),
              "normalized_identity_code_sha256": digest(__import__("pathlib").Path(__file__).with_name("audio_history_v58.py")),
              "rows": {"reserve": len(reserved), "consumed": len(consumed)}, "coverage": coverage,
              "normalized_cross_role_overlaps": dict(Counter(key.split(":", 1)[0] for key in common)),
              "pcm_aliases": len(aliases), "additional_alias_matches": len(alias_matches),
              "alias_screening_matches": screening, "historical_files_reverified": len(bindings),
              "new_model_calls": 0,
              "scope": "All source rows have normalized identity coverage; no known reserve/consumed byte, PCM, voice or declared text connection is allowed. Recorded output search supplements, not replaces, source-role separation. No claim of unseen voices across all history or pretraining."}
    write(TRIAL / "additive-review.json", result)
    print(json.dumps(result), flush=True)
    if common or screening:
        raise ValueError("Preserve detected exposure/role conflicts; do not score reserve")


if __name__ == "__main__":
    review()
