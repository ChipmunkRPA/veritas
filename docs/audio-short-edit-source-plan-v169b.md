# Stock voice shortlist for the short-edit pilot

The first metadata freeze connected the full stock archive into two
components through shared text in unused alternatives. Each role received
one stock test slot. This successor fixes the stock candidate lists before
assigning roles. It retains the original freeze and reproduces every
genuine, cloned, expressive and noise assignment, including all backup
candidates. [Original source method](audio-short-edit-source-plan-v169.md)

Visit known stock profiles in the fixed salted order. Within each profile,
visit rows in the original salted order. Keep the first six candidates
whose original source, script, normalized text, eight-word opening and
encoded bytes have no overlap with previously admitted candidates. Each
backup can be used only for a pre-score eligibility failure. Join the actual
shortlist and all backups by those identities and confirm that every stock
component is separate. Alternate the whole components between holdout and
development. The retained profiles have recorded historical exposure.

Run from the repository root after the first metadata freeze.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_short_edit_sources_v169b freeze
```

The successor retains six stock components per role. The next local stage
must verify the selected publisher archive hashes, recover complete source
bytes, decode through the app's production PCM helper, and compare full and
short-prefix sample hashes to the frozen exclusions and other selected rows.
Use only deterministic earlier candidates within each frozen group.
Materializing holdout audio can verify bytes, samples and ordinary edit
construction. Holdout scoring and fitting remain separate operations.
[Aggregate freeze](evaluation-panels/audio-short-edit-source-plan-v169b.json)
