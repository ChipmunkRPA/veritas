# Independent source roles prepared before combined-confidence training

The next audio calibration has **400 qualified original foreground recordings**
and **48 qualified backgrounds**, with 64 matched synthetic recordings still to
generate. The corrected source plan keeps fitting, confidence calibration,
method selection and final confirmation separate. No generator, detector, VAD or
confidence learner has used this new panel. This is source preparation, not an
improved confidence score or accuracy result.

## Actual qualified sources and planned additions

Each role has the same source counts:

| Source | Qualified originals per role | Declared foreground voices per role | Exposure and limits |
| --- | ---: | ---: | --- |
| Genuine speech | 56 | 14 | 56 foreground voices across the study are absent from recorded project history. Whole-file bytes and production PCM verified. |
| Cloned speech | 20 | 10 | 40 parent identities across the study are absent from that history and from the entire genuine candidate bank. Shared dialogue origins are joined before assigning roles. |
| Expressive synthesis | 24 | 6 | New recording bytes and declared scripts; all 24 stock profiles appeared historically. This targets the expressive-speech weakness without claiming unseen-provider voices. |
| Matched stock synthesis | **16 pending** | 4 | 16 history-unseen profiles planned across the study. Each profile will render four qualified genuine transcripts, paired to one genuine voice in the same role/group. No generation has occurred. |
| Background speech, noise, music | 12 | — | Four of each kind. Source-disjoint between roles and from study foregrounds. Music recordings were used historically; speech-background voices were seen historically but these recordings/scripts are new. Noise originals pass recorded byte/PCM exclusions. |

There are 464 planned foreground sources and 136 declared foreground voices:
112 history-unseen and 24 historically seen expressive profiles. These labels
refer to recorded project history, not biometric verification or absence from
pretrained model data. Matched synthesis is an intended-script control; actual
spoken-word alignment and generation quality still need verification.

The existing voice confidence comparison covers four-second windows. This
source preparation does not extend that scope. The cloned archive contains many
shorter utterances; they remain an explicit future qualification need. No short
source is padded or repeated merely to satisfy this panel's duration criterion.

## Source eligibility and preserved preparation failures

The original v70 metadata recipe requested three globally distinct utterances
from each of 40 new cloned voices. Only 21 voices meet that combined prefix/
content constraint in its fixed ordering. The failed preparation remains intact.
The v71 successor retains the original seed/order and uses two qualifying
utterances per cloned voice; 41 voices support that requirement, of which the
fixed rule selects 40. No model response selected this change.

The v72 original-source qualification checks original payload hashes, actual
production decoding, finite/non-silent samples, full/prefix PCM exclusions and
declared text connections. All 400 foregrounds and 48 backgrounds qualified.
One candidate in an ordered genuine-voice pool was shorter than the fixed
minimum; the prespecified rule chose the next eligible recording from that same
voice. All explicitly fixed synthetic originals qualified without substitution.
The original files, rejected candidate and complete journal remain private.

That bounded decoding stage exited successfully in 83.01 seconds, with no guard,
sampling failure or resource-limit observation. Maximum sampled child-tree RSS
was 892,157,952 bytes, a lower bound rather than a true peak. Four verified
short-lived decoder exits are recorded. These facts establish source integrity
and duration, not model usefulness.

## The additional ancestry audit changed the roles

The initial planner separated voices and declared scripts but missed shared
original dialogue contexts across some cloned voices. A deeper check found
**eight dialogue origins crossing roles**. The v71/v72 role-independence claim
is superseded for that connection; their completed original-file/PCM checks
remain valid. No new source had entered generation, fitting or detector inference.

The v75 correction joins cloned voice, original dialogue, declared text/prefix,
byte and PCM links. The 40 cloned voices form 29 components: eleven pairs and
18 singletons. Whole components are assigned together, retaining ten cloned
voices and all 20 cloned recordings per role. Sixty cloned records change role;
every selected byte, label, script and pending matched prompt is preserved.
Genuine, expressive, stock-prompt and background role assignments stay unchanged.

Actual expressive generation sidecars add their distinct challenge IDs; genuine
and background speech add their original chapter connections. The complete
source graph includes the intended matched-synthesis links. It has no known
cross-role component after correction.

| Corrected role | Foreground connected groups including pending | Genuine class groups | AI class groups including pending | Background groups |
| --- | ---: | ---: | ---: | ---: |
| Fitting | 27 | 14 | 17 | 12 |
| Confidence calibration | 27 | 14 | 17 | 12 |
| Method selection | 27 | 14 | 17 | 12 |
| Final confirmation | 28 | 14 | 18 | 12 |

Matched genuine/synthetic groups occur in both class pools. Windows, repeated
utterances, shared voices and shared backgrounds are not independent people.
These modest group counts limit future uncertainty estimates; they do not
support a universal reliability or customer-confidence guarantee.

## Separate verification and next execution

A separate breadth-first reviewer reconstructs all 157 connected components
across 512 planned source nodes, using different traversal code. It rehashes
all 448 physical originals, verifies immutable labels/content, preserves all
64 pending prompts and exactly reproduces the corrected role support. Three
actual altered-manifest checks reject a cross-role clone, changed original
truth and changed paired prompt. This is local independent calculation review,
not participation by an outside human reviewer or a second full decoding pass.

An additional primary-context review verifies the 96 expressive generation
sidecars, 80 cloned parent/utterance/dialogue declarations, genuine chapter IDs
and every declared source permission. Public-domain noise permission is retained
through its exact frozen recipe reference. Three further actual alterations of
clone, expressive and genuine source-context fields are rejected, so renaming
a context cannot hide an ancestry link. A declared-script comparison also finds
no match with the recorded LJ Speech prompt bank; its exact DFADD per-file
mapping remains unknown. These checks do not prove unseen pretraining.
All 30 focused source/role/eligibility/graph tests and lint passed locally.

Next download/verify the 16 planned stock-profile assets, publish the exact
generator/runtime/prompt freeze and render the 64 matched sources. Then qualify
the complete sources/mixtures and freeze the confidence experiment before its
first detector forward. Current, retained pair and separately fitted/calibrated
combination methods will be compared using the assigned roles. Gains and both
error directions remain tradeoffs; a modest extra error is not an automatic
rejection. Do not fit or select using final confirmation.

Source terms remain CC BY 4.0 for genuine/expressive material, publisher-declared
MIT PLN terms for the cloned material, public-domain noise and CC BY 3.0 music,
with source/model attribution retained. The stock generator retains its pinned
Apache-2.0 weight terms. These terms do not prove all real-person permissions,
biometric identity, unknown near-duplicates or absence from pretraining. See the
[source-bank preparation](audio-calibration-data-preparation-v68.md).

Public receipts: [v70 stopped recipe](evaluation-panels/audio-confidence-roles-v70-stop.json),
[v71 candidate recipe summary](evaluation-panels/audio-confidence-roles-v71.json),
[v72 original integrity summary](evaluation-panels/audio-source-qualification-v72.json),
[v72 resource profile](evaluation-panels/audio-source-qualification-v72-profile.json),
[v75 corrected ancestry](evaluation-panels/audio-source-ancestry-v75.json),
[v76 separate review](evaluation-panels/audio-source-review-v76.json) and
[adverse checks](evaluation-panels/audio-source-adverse-v76.json).
The [primary-context review](evaluation-panels/audio-source-context-review-v77.json),
[context alterations](evaluation-panels/audio-source-context-adverse-v77.json) and
[known prompt-bank check](evaluation-panels/audio-source-prompt-review-v76.json)
retain the additional verification separately.
Only code, aggregate metadata and whole-artifact hashes are published. New source
audio, text, individual identities, pairings, PCM fingerprints and learned
features remain private on the development machine/external drive. No serving
default, protected calibration, payment, workflow, outreach or GCP change.
