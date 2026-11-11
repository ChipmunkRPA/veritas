# Resume the webcam confound readout

The completed v224 readout returned three unavailable results. The archived,
checkpointed SQLite files retain a WAL-mode header; SQLite does not open a
WAL-mode database through in-memory deserialization. This continuation binds
that unchanged failed result and v224 freeze. It reuses the frozen v224 archive,
guard, source, frame, role and v222 measurement checks.

Only after the v224 archive verification and its check that the SQLite WAL and
rollback-journal sidecars are absent or empty, replace bytes 18 and 19 with
`01 01` in a temporary in-memory database copy. A normal `01 01` header remains
unchanged; unsupported headers are unavailable. Set SQLite query-only mode as
in v224. Do not alter the tar, source database, receipts or original result.
SQLite documents the [WAL deserialization limitation and workaround](https://www.sqlite.org/c3ref/deserialize.html).

This is one consumed matched webcam source. Unavailable windows remain visible;
frame-matched comparison is reported only where role frame indices agree.
Measurements are descriptive, not an edit or origin decision. No neural call
or serving change is permitted.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_webcam_confound_resume_v227 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_webcam_confound_resume_v227 readout
```
