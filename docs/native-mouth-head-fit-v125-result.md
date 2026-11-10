# Three native mouth heads fitted locally

The frozen fitting-only stage completed successfully on the audited full-native
feature bank. All three independently learned linear heads converged. This is
a training result; no calibration or held-out detection result is claimed.

The [machine-readable fitting receipt](evaluation-panels/native-mouth-head-fit-result-v125.json)
binds the private learned parameters, central-frame fitting labels, coverage,
resource journal and frozen procedure. The
[procedure](native-mouth-head-fit-v125-method.md) was published before execution
at commit ba7554e. The input bank and replay results were published at 3823907.

| Measurement | Actual result |
| --- | ---: |
| Fixed fitting sources retained | 18 |
| Untouched original points for genuine-reference projections | 1,347 |
| Supported labelled fitting centres | 8,082 |
| Positive neural-edit central pictures | 720 |
| Negative central pictures, including ordinary edits and untouched context | 7,362 |
| Visual-only dimensions / iterations | 16 / 11 |
| Concatenation dimensions / iterations | 32 / 118 |
| Explicit-interaction dimensions / iterations | 64 / 127 |
| New pretrained-network inputs | 0 |

Training used source/class/condition-balanced weights and each head's own
weighted feature scaling. The genuine-reference projections used only original
fitting data. Every saved supervised row belongs to the fitting role; calibration,
selection and confirmation groups supplied no training vectors or labels.
The actual learned artifact and coverage/label file hashes were checked against
the completion receipt after the guarded worker exited.

The worker finished in 10.016 seconds with exit code 0, no resource guard failure and
zero sampling errors. Maximum sampled descendant RSS was 836,321,280 bytes;
minimum sampled available RAM was 6,926,712,832 bytes. These are samples, not
guaranteed peaks. The unchanged data-only limits remained in force.

The three heads return raw signed margins, not AI probabilities. They are not
registered for customer analysis or incorporated into overall confidence.
Next work must freeze and perform separate calibration for each arm, compare
false alarms, missed edits, localization and unassessed scope on locked groups,
then test fresh placement/generator transfer and the full local application
flow. Current checks and protected audio rollback remain unchanged.

Private learned reference parameters, coefficients, source identities and labels
remain on Seagate. Public artifacts contain methods, tests, aggregate counts and
whole-artifact commitments. No outside reviewer, outreach, GitHub workflow,
real payment or GCP deployment was involved.
