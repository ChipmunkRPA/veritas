# Short speech edits under competing voices

This study tests whether a narrow edit can be localized from one submitted
recording by requiring the same timing in a raw and a noise-reduced
reading. The previous unrelated-reader test found one useful location in
32 loud-background edits using the current two-second scan. The shorter
check found none and added ordinary alerts. [Previous held comparison](audio-short-edit-union-holdout-2026-11-15.md)

## Comparison

The candidate follows the separation of boundaries and segment content in
recent partial speech work. It compares the mean edit response inside a
300–1,050 ms interval with the stronger response in the two surrounding
speech intervals. It adds a contribution from the weaker of the two
boundary readings. The shared proposal uses the lower score at the same
sample range across the original and enhanced waveforms. Both inputs use
the same six-second clock and original speech admission. The study
measures the raw, enhanced and shared proposals independently and also
tests adding the shared result when the existing check has no result.
[Boundary and segment study](https://arxiv.org/abs/2604.02913) ·
[Artifact-controlled dataset study](https://arxiv.org/abs/2512.13012)

The [frozen procedure](evaluation-panels/audio-standalone-dual-view-plan-2026-11-18.json)
selects 28 SINE readers from verified archived publisher prefixes after
excluding reader IDs named in seven earlier study stages. Twelve readers
and four AMI excerpts set the candidate points from ordinary files before
held scoring. Sixteen separate readers and four further excerpts supply
the test cases. Each held reader contributes the unchanged foreground,
one labeled synthetic cut-and-paste edit and an ordinary splice made from
the same source recording. The paired versions use the same recorded
competing speech at foreground-to-room ratios of −5 and +3 dB.
The four held room excerpts also enter alone, for 100 held deliveries.
Publisher identity and exact input commitments establish the split.
Model-pretraining identity overlap is unknown.
[Source reservation](evaluation-panels/audio-standalone-dual-view-sources-2026-11-18.json)

The test uses a useful-location rule of at least 50% edit coverage and
0.30 temporal intersection over union. A genuine file flagged at the
same edit position counts as a paired collision. At −5 dB the frozen
admission rule requires at least three additional useful locations across
three readers with no paired ordinary collision. It allows two additional
ordinary flags and requires zero added flags on the isolated-room checks.
The tracked intervals and data are separate from the overall AI score.

The SINE release carries Apache 2.0 terms and the AMI audio carries
CC BY 4.0 attribution terms. The local focused speech head uses the pinned
encoder's CC BY-SA 3.0 terms and the enhancer uses its pinned MIT release.
Only scalar scores, intervals and hashes are retained here.
[SINE terms](https://huggingface.co/datasets/PeacefulData/SINE#license) ·
[AMI terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) ·
[Enhancer license](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/LICENSE) ·
[Encoder provenance](audio-wavlm-v13-protocol.md) ·
[Current two-second comparison assets](audio-specialists-v15-protocol.md)

## Held results

| Check | Located at −5 dB | Ordinary flags at −5 dB | Located at +3 dB | Ordinary flags at +3 dB |
| --- | ---: | ---: | ---: | ---: |
| Current two-second scan | 0/16 | 1/32 | 3/16 | 3/32 |
| Narrow comparison on the original | 0/16 | 3/32 | 0/16 | 3/32 |
| Narrow comparison on the enhanced reading | 0/16 | 0/32 | 0/16 | 0/32 |
| Shared interval | 0/16 | 0/32 | 0/16 | 4/32 |
| Current scan with shared-interval fallback | 0/16 | 1/32 | 3/16 | 7/32 |

All foreground deliveries had comparable speech. Two of four isolated
rooms had comparable speech, and every check flagged zero isolated rooms.
The fallback added one misplaced edit range at each level. At +3 dB it
also added four ordinary flags from three readers. All three correctly
located current results cleared the matched ordinary comparisons.

The shorter interval grid could fit every held edit with the known
positions supplied. The two-second grid could fit four of 16. Before
applying any cutoff, the strongest shared proposal usefully overlapped
only two of 16 edits at each level. The narrower geometry increased
the available locations, while the observed score ranked other time
ranges above most annotated edits. The tested method stays in evaluation.
[Paired outcomes and measurements](evaluation-panels/audio-standalone-dual-view-2026-11-18.json.gz)

## Replay

The numeric reviewer requires Python's standard library. Native replay
requires the exact previously approved local assets and room excerpts.
The source restoration reads the reserved SINE archive prefixes from
one or more cache directories. A separate restoration reproduced all
56 source WAVs byte for byte. A second native candidate run on those
restored files reproduced the same held and calibration measurements.
An independent recount matched the baseline and fallback totals and all
100 held input hashes. The earlier procedure documents
the publisher download and byte checks.
[SINE source preparation](audio-sine-resynthesis-transfer-2026-11-14.md#replay) ·
[Pinned room excerpts](audio-sine-resynthesis-transfer-2026-11-14.md)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VERITAS_MODELS_DIR="$PWD/models"
base=docs/evaluation-panels/audio-standalone-dual-view
module=scripts.audio_standalone_dual_view
plan="$base-plan-2026-11-18.json"
sources="$base-sources-2026-11-18.json"
receipt="$base-2026-11-18.json.gz"
python3 -B -S -m "$module" review "$plan" "$sources" "$receipt"
python3 -B -S -m "$module" restore "$plan" "$sources" "$run_sources" "$sine_prefix_cache"
backend/.venv/bin/python -B -m "$module" candidate \
  "$plan" "$sources" "$run_sources" "$ami_backgrounds" "$voice_weights" "$new_candidate" "$enhancer"
backend/.venv/bin/python -B -m "$module" baseline \
  "$plan" "$sources" "$run_sources" "$ami_backgrounds" "$voice_weights" "$new_baseline"
python3 -B -S -m "$module" combine \
  "$plan" "$sources" "$new_candidate/candidate.json.gz" "$new_baseline" "$new_receipt"
```
