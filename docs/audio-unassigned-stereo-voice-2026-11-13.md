# Score both stereo separation outputs under competing speech

The earlier stereo test chose the more centered of two separated signals before running the voice check. That choice selected the wrong signal on all eight edits with an off-center target in a third room. This follow-up scores both anonymous signals and tests whether retaining the larger response or requiring both responses can recover useful edit locations. The original stereo downmix and the earlier selection rule provide separate baselines. Each score is a response to a processed signal and does not identify its speaker. [Earlier stereo study](audio-blind-spatial-transfer-2026-11-12.md) · [Third-room acoustic results](audio-blind-huddersfield-transfer-2026-11-12.md)

Research on mixture-derived speaker candidates motivates retaining both alternatives. Recent work on target speech reports that even enrollment can confuse sources, and independent research shows that speaker characteristics can influence detector responses. This experiment uses the existing blind separator, voice model, and speech gate with their pinned local files. It does not use those papers' models or implementation. [Mixture-derived speaker candidates](https://arxiv.org/abs/2604.03219) · [Target confusion under overlap](https://arxiv.org/abs/2609.30631) · [Speaker sensitivity in deepfake detectors](https://arxiv.org/abs/2607.21820)

## Frozen comparison

The 160 six-second files come from retained source-separated room studies. The Surrey arm contributes six edited examples and 20 ordinary views, of which only the ordinary views set the operating points. The BBC room supplies six edited and 20 ordinary views. Each of three separately reported Huddersfield layouts supplies eight edited and 28 ordinary views. The three Huddersfield layouts reuse the same target groups and include different playback layouts. These groups differ from the Surrey and BBC groups. Each room study includes untouched speech, ordinary same- and different-reader splices, background-only controls, and synthetic edits across clone, expressive and stock families. The source and room recordings were already used in other project studies. [Source and license records](audio-measured-spatial-transfer-2026-11-12.md#sources-and-fixed-comparison) · [Huddersfield layouts and terms](audio-blind-huddersfield-transfer-2026-11-12.md#inputs-and-scope)

The frozen two-second windows start at 1 and 3 seconds. Each submitted or processed view keeps a window only when that view passes the same speech check. The separated views must also be unclipped. The previous balance rule chooses an output from the whole clip. The other two methods combine per-window responses across the anonymous outputs. The higher-response method uses the largest available response. The agreement method uses the lower response when both are available. Each method's point is the highest eligible genuine-file response in Surrey, with strict excess. The comparison also replays cases within shared time coverage and reports retrospective equal false-group budgets. A useful location must overlap half the actual edited time and reach 0.30 temporal intersection over union.

## Results

Each cell gives useful locations followed by flagged ordinary source groups at the frozen points. One group can supply multiple conventional variants.

| Held layout | Original downmix | Earlier selection rule | Higher response | Both outputs must support |
|---|---:|---:|---:|---:|
| BBC, 6 edits | 0 / 2 | 2 / 2 | 2 / 2 | 0 / 3 |
| Huddersfield balanced target, 8 edits | 0 / 4 | 0 / 2 | 0 / 2 | 1 / 4 |
| Huddersfield lateral target, 8 edits | 0 / 3 | 0 / 0 | 1 / 2 | 4 / 6 |
| Huddersfield second competitor, 8 edits | 3 / 3 | 0 / 2 | 0 / 3 | 2 / 4 |

The higher-response rule's one additional lateral edit came with one newly flagged genuine group relative to the original downmix. It reproduced the two BBC locations already found by the earlier balance rule. It missed all three original useful locations in the two-competitor layout and flagged one background-only recording there. The agreement rule recovered four lateral edits with five newly flagged genuine groups at its frozen point. At a retrospective budget of one false group it found two lateral edits, compared with zero for the other methods. That result uses the held data to set the budget and needs a new panel at a frozen operating point. With the second competitor, the agreement rule added one location and lost two of the original three. Restricting all comparisons to shared time coverage did not change the useful-location counts.

Keep the original voice path. The higher-response and agreement methods offer a way to recover some off-center edits, and performance depends on who else is speaking and how the microphone captures them. A further test needs new speakers, rooms and generators, with the original analysis retained alongside the candidate and the decision rule fixed before scoring. No candidate entered serving or overall confidence.

## Reproduce

The CPU trial used 818 distinct voice forwards. A second native run changed the FFT implementation and reproduced all 960 original and separated window hashes and their speech admissions. The public standard-library reviewer checks the counts, frozen point selection, exact location arithmetic, and retained abstentions from anonymous scalar rows. It does not rerun voice inference. The ledger keeps raw audio and per-window hashes private. [Audit receipt](evaluation-panels/audio-unassigned-stereo-voice-2026-11-13.json)

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_OFFLINE=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze score audit; do
  backend/.venv/bin/python -B -m scripts.audio_unassigned_stereo_voice "$phase" "$pilot" "$third_room" "$run" || exit 1
done
python3 -B -IS backend/scripts/review_audio_unassigned_stereo_voice.py "$receipt" --from-run "$run"
python3 -B -IS backend/scripts/review_audio_unassigned_stereo_voice.py docs/evaluation-panels/audio-unassigned-stereo-voice-2026-11-13.json
```

Use the retained verified output directories from the two linked room studies as `pilot` and `third_room`. The runner checks the earlier published roster and freeze commitments before reading audio.
