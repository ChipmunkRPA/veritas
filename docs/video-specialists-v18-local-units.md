# Video local-unit source-clock audit

The source-bound unit planner now distinguishes whole-clip manipulation from
manipulation among the frames supporting each local check. It does not run a
detector, fit a calibration or change serving behavior. Full native video
scoring and its improvement comparison remain unfinished.

Across all432 retained clips, the audit constructed9,504 planned units from
previously measured native clocks and known inserted-frame boundaries. It
verified401 actual24fps and31 actual25fps inputs. It preserved local genuine
truth for2,650 units inside manipulated clips. Whole-file AI labels therefore
cannot be copied into every frame or time window.

The explicit research sampler chooses native frames nearest each half-second
target, resolving ties earlier. Crop checks target their sampled source frame.
Fixed temporal plans use three middle consecutive-frame triplets and two
triplets on either flank. Appearance and photometric views share the same
recorded middle support and context. An edit confined to a flank does not label
the middle as edited. This is a new measurement plan, not a replay claim for
every adaptive production window or a validated sampling improvement.

Source coordinates use actual PTS and exact integer frame indices, including
explicit sampled/context lists within the enclosing time range. Genuine
ordinary25fps controls retain their actual clock. Inserted positives remain
bound to their verified24fps construction. Truth describes manipulation
presence in sampled source frames; it does not establish which depicted face
or which pixels were AI-produced. Face-specific claims would require separate
face-region truth, and no such pixel/identity calibration is claimed here.

All228 retained manipulated clips contain at least one positive planned unit,
but6,450 known edited frame instances across those clips are not directly in
the sampled support. The sparse plan never covers whole-file originality.
Short clips can have no temporal context; those views are explicitly unsupported.
All planned readouts remain present when a native observation is missing, with
a null response and reason. They cannot become numeric negatives. A response
cannot rewrite its planned truth, semantic target or source location.

The [aggregate audit](evaluation-panels/video-specialists-v18-sampled-unit-clock-audit.json)
contains input-journal and exact plan commitments, with no raw clip, face
representation or source identity list. Ten new source-scope/status tests and
the ten existing input-clock/truth tests passed. The audit loaded no detector
response and performed zero new decodes or neural forwards. It is not fresh
detection testing, a full648-clip preparation receipt or outside human review.
