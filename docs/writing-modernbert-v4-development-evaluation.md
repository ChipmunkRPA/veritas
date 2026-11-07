# Reject the binary alternative for localized mixed-writing screening

7 November 2026. The candidate passes only1/9 fixed requirements. It misses more
AI-origin passages and highlights more human-origin passages than the current
context-based check. Both candidate views also have poorer origin ranking.
Keep the current model and do not spend unused authors or change the operating
references to rescue this candidate.

The [published prospective freeze](evaluation-panels/writing-v4-modernbert-freeze.json)
preceded all project classifier responses. All40 consumed documents,40 author
groups and674 exact focus/context pairs completed under both models, retaining
all four arms and2,696 scored inputs. There were no failures or omitted source
positions. This is reused mixed-writing development, not fresh validation,
an unassisted-human false-alarm benchmark or a customer accuracy estimate.

## Fixed primary comparison

The primary comparison is candidate/focus versus baseline/context. References
0.5/0.8 are frozen research raw-response points, not calibrated confidence levels
or public screening-threshold claims. All rows retain the same322 pure-Bot
passages,153 pure-User-origin passages,57,604 Bot-origin characters and26,095
User-origin characters.

| Research reference and method | Recovered pure-Bot passages | Highlighted pure-User passages | Bot-character recall | Pooled character IoU |
| --- | ---: | ---: | ---: | ---: |
| 0.5, baseline/context | 295 | 52 | 89.35% | 0.7353 |
| 0.5, candidate/focus | 269 | 87 | 85.41% | 0.6508 |
| 0.8, baseline/context | 276 | 38 | 83.20% | 0.7036 |
| 0.8, candidate/focus | 233 | 63 | 73.29% | 0.5880 |

At each reference, the candidate fails all four quality requirements: more
pure-Bot recoveries, limited pure-User highlights, Bot-character retention and
a positive lower bound for paired author-mean IoU gain. Only complete,
failure-free processing passes. These human-origin highlights occur inside
assisted mixed documents; they are not an estimate for verified unassisted
human writing. Context predictions also do not prove individual word origins.

The controls show this is more than a changed operating point. Pure-origin
rank AUC is0.8782 for baseline/context and0.6987 for candidate/focus. On the
same focus-only inputs, baseline is0.7373 versus candidate0.6987; on the same
contexts, baseline is0.8782 versus candidate0.5811. The
[complete aggregate](writing-v4-modernbert-development-evaluation.json)
retains all counts, rates, ranks, coverage and context/focus shifts for all arms.
These ranks use the focus's retained-origin label. Surrounding context can
contain AI text even when the focus is human; a context-level AI signal does
not establish that the highlighted focus was AI-written.

Author-mean IoU differences use40 author groups and2,000 paired bootstrap
replicates. At0.5 the difference is-0.03906,95% interval[-0.10050,0.01805];
at0.8 it is-0.07131, interval[-0.14188,-0.00433]. The mixed authors and prepared
inputs are already consumed. Unknown training overlap, possible off-platform
assistance and task-specific origin annotations remain limitations; these are
not customer confidence intervals or proofs of how a person wrote a sentence.

## Verification and decision

Both native replays pass80 predeclared probes each, covering every author's
first focus in both views and all required overflow cases. Complete token
coverage is retained during every scored input. Maximum native response
differences are2.3245811462402344e-6 for baseline and4.827976226806641e-6 for the
candidate, below the unchanged1e-5 tolerance. Historical baseline contexts
match all674 earlier responses within1.1920928955078125e-7.
[Baseline native summary](evaluation-panels/writing-v4-modernbert-baseline-native-summary.json),
[candidate native summary](evaluation-panels/writing-v4-modernbert-candidate-native-summary.json).

[Separate arithmetic review](evaluation-panels/writing-v4-modernbert-numerical-review.json)
reproduces all counts, ranks, rates, shifts, author intervals and decisions.
Its declared gate/input helpers remain shared; this is a separate arithmetic
implementation, not outside human validation. Source words, author identifiers,
edit histories, per-passage annotations and individual native probes remain
private; public native summaries bind the full local receipts by hash.

Reject this model for the proposed replacement. The45 history-supported unused
human controls and unused mixed authors remain unscored. A future method must
improve the actual passage/context task before unused-author confirmation,
own calibration and full customer/PDF/JSON/prepaid testing. A larger or newer
binary detector does not inherit the current model's rewriting/humanization
scope. Serving models, overall confidence, billing and GCP remain unchanged.
Actual outside human reviewers remain zero.
