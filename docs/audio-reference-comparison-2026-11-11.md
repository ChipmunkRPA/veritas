# Compare an audio recording with an earlier copy

On a local development instance, set `VERITAS_AUDIO_REFERENCE_COMPARISON_ENABLED=true`. Sign in, open a completed audio analysis, and choose **Compare with an earlier audio recording**. The service compares the retained analyzed file against a temporary upload. It creates no analysis, payment order, or retained copy of the new upload.

Each file may contain up to 30 seconds of decoded audio and occupy at most 48 MiB. Both first audio streams must yield continuous signed 16-bit, 16 kHz mono samples. For matching start ticks and sample counts, the tool compares the same sample positions. For supported small trims, it compares only the shared span using the two file clocks.

The report groups adjacent 20 ms bins containing differing samples. It records the exact number of differing samples in each bin run and the peak and RMS of analyzed minus earlier samples. The three longest runs have aligned waveforms and playable excerpts covering up to their first two seconds. The two excerpts share one amplitude scale. The difference plot uses a labeled separate scale so quiet changes remain visible. The PDF records the source clocks, file hashes, times, counts, and waveforms. The JSON includes the same measurements, source ticks, checksums, and playable WAV bytes. Encoding, gain changes, and ordinary edits can produce differences. This comparison contributes no AI confidence.

## Differences after a volume adjustment

The comparison can also examine two lossless recordings that are already mono at 16 kHz and at least two seconds long. It requires sound activity in both the first and last half second. A file with full-scale samples or a potentially clipped fit receives only the raw comparison. Each end supplies the range of constant gains compatible with its signed 16-bit samples under nearest-sample rounding. The two ranges must overlap within 0.25× to 4× and exclude unity. The same rational gain bounds and the original source ticks and sample hashes for both end windows are recorded in the export.

The check counts a sample when its quantization interval excludes every value predicted by the supported gains. It reports zero residual differences only when one gain fits all the samples. Each range uses the same 20 ms bins as the raw comparison. The residual waveform shows submitted samples minus the earlier samples adjusted by the midpoint of the supported gain range and rounded to the nearest even sample. The two playable excerpts contain the unmodified decoded samples. When the fit is unsupported, including changes that invalidate either end window, the ordinary raw comparison remains available.

## Compare copies with a small trim

When two 16-bit lossless 16 kHz mono files have different sample counts, the comparison can recover a constant offset with at most 250 ms outside the overlap in either file. The supported codecs are signed 16-bit PCM and 16-bit FLAC. At one-fifth and four-fifths of the shorter length, each 1,024-sample window from the earlier file must contain sufficient activity and varied samples, and must match exactly once within the allowed search span of the analyzed file. The two windows must support the same offset and have different sample hashes. The shared interval, distinct source ticks, anchor hashes and exact excerpt samples are in the JSON. The browser and PDF show each file’s elapsed time for the shared interval and the selected waveforms. The check abstains when it cannot establish that alignment.

When exact matches are unavailable, the comparison can account for a trim and a constant volume change together. The same two windows must uniquely support the same offset. The correlation step proposes candidate offsets and signed PCM quantization bounds verify both windows. The bounded gains from the two windows must agree and leave headroom against clipping. The comparison shows raw differences and, separately, the samples that still differ after the supported volume change. The browser, PDF and JSON retain distinct timelines for both views. Lossy encodes, time-varying gain and unsupported offsets can remain unassessed.

## Local verification

Eight distinct retained genuine speech sources supplied three-second mono controls. Four comparisons inserted scaled recorded room audio at samples `[22400,35200)`. Four substituted known synthetic speech at `[10240,19840)`. The comparison recovered all eight exact ranges. An independent sample calculation reproduced the counts, source ticks, difference levels, and excerpt samples. All eight lossless FLAC conversions matched the original decoded values. A 0.7 gain changed all eight copies, with differences covering every bin on seven. The original comparison left all eight 20 ms trims unassessed. [Source preparation and rights](audio-paired-stereo-overlap-2026-11-11.md#matched-inputs)

Authenticated app tests verify the JSON and PDF against the same decoded intervals. A second fixture preserves a nonzero 16 kHz source origin while displaying elapsed seconds relative to the first sample. Tests also verify that access checks apply and the comparison does not add objects, analyses, jobs, or payment records.

### Changes in the recorded stereo channels

Version 5 also compares the left and right channels of the first audio stream separately when both files have two channels and identical continuous clocks. It uses the original channel order and decoded 16 kHz signed 16-bit PCM. For each affected channel, the app shows the 20 ms bins with changes, the exact first and last differing sample inside each range, and the source-clock ticks. It provides playable excerpts from both copies, their waveforms, and a difference waveform for up to three ranges per channel. PDF and JSON preserve the same coordinates. A mono counterpart, trim, incompatible clock, or unavailable channel capture leaves the stereo check unassessed. Ordinary mixes and encoding can produce differences. The check measures changes between two supplied copies and leaves the AI confidence score unchanged.

Using the 12 retained six-second speech groups with competing room speech, an engineered control applied each generated or ordinary replacement to the left channel and the equal-and-opposite change to the right channel. The mono mix matched in all 48 pairs. Channel comparison recovered all 12 generated replacements, all 24 ordinary replacements, and zero differences in all 12 untouched pairs. A second panel used 26 pairs from the held measured-room stereo mixtures. Channel comparison reproduced the sample-level differences on all 18 edits, including the room response tails, and found zero differences on six untouched copies and two isolated background controls. A lossless FLAC conversion matched both channels. The controlled cancellation tests validate the sample measurement and do not represent prevalence or detection accuracy on random uploads. [Competing speech panel](audio-remote-neighbor-replication-2026-11-12.md) · [Measured-room stereo panel](audio-measured-spatial-transfer-2026-11-12.md)

Local Chrome verified six waveforms, four playable excerpts, and API-equivalent PDF and JSON for the 0.320–0.960 s example. An independent calculation checked the two source hashes, each range and its changed samples, and WAV readback. Offline JSON replay reproduced the exact artifacts. The request created no new job, payment, analysis, or retained evidence object. The separate stereo reader also verified FLAC, 44.1 kHz AAC, and 48 kHz Opus deliveries. FLAC preserved the original one range per channel. The lossy AAC and Opus pairs yielded 19 and eight ranges per channel, respectively, which shows why these raw differences do not establish an edit boundary after re-encoding.

### Volume-adjusted copies with recorded competing speech

The local panel generated 52 new signed 16-bit WAV pairs using retained six-second PCM from 12 held speech groups and four room-only controls. The three authored edit variants for each speech group share the unchanged host and background outside known edit bounds. They include one generated speech insert from one of three synthesis families, one ordinary same-reader insert, and one ordinary other-reader insert. The speech mixes use recorded overlapping talkers from four rooms at −5 dB foreground-to-background ratio over the clip. [Source preparation and rights](audio-remote-neighbor-replication-2026-11-12.md#held-sources-and-frozen-comparison)

| Delivered pairs | Total | Gain fit available | Residual found at known edit |
| --- | ---: | ---: | ---: |
| Unedited speech, gain only | 12 | 12 | 0 |
| Generated speech insert plus gain | 12 | 8 | 8 |
| Ordinary same-reader insert plus gain | 12 | 8 | 8 |
| Ordinary other-reader insert plus gain | 12 | 8 | 8 |
| Room-only recording, gain only | 4 | 2 | 0 |

The raw comparison marked every 20 ms bin in all 52 pairs. All confirmed residual samples were inside the authored edits. The four edge edits in each edited arm interfered with an end window, and two room controls lacked enough active anchor samples. Independent integer arithmetic verified all 3,648,000 admitted samples, every raw bin count in the 52 pairs, and all 48 adjusted excerpt WAVs. Both kinds of ordinary edits produced the same observable type of residual as the generated inserts. The result describes where the two supplied copies differ after a constant volume adjustment.

To reconstruct the inputs, rebuild the [native held roster](audio-remote-neighbor-replication-2026-11-12.md#replay). For each arm take its original mono PCM and the untouched clip from the same group. Multiply each source by 7/8 for headroom and round to signed 16-bit samples. On the submitted file apply gains 5/8, 7/10 and 9/8 in repeating order across the sorted speech groups, and 7/10 on the room-only controls. Use nearest-even rounding for both steps and write WAV at 16 kHz. Run the comparison command below on the resulting pairs.

### Small trims with recorded competing speech

The same 12 speech groups supplied new WAV variants with a controlled trim. Each untouched copy aligned and had zero differences across the shared span. The check found the known splice in 10 of 12 generated inserts, 10 of 12 same-reader splices and 10 of 12 other-reader splices. The remaining edits prevented an exact anchor match. Every internal cut, different-source pair and larger trim stayed unassessed, 12 of each. Native sample calculations verified all admitted offsets and changed-sample counts, and matched the playable excerpts against both source files. These are shared-source localization tests under loud background speech.

To repeat, use the same native roster and omit the room-only records. Number the speech groups from zero in roster order. For group `i`, remove `101 + 47*i` samples from the beginning and `153 + 41*i` from the end of each submitted file. Use the same group’s untouched mix as the earlier file. For the controls, use the untouched mix with samples `[48000,48241)` removed, the next group’s untouched mix with the scheduled trim, and the same mix with its first 4,001 samples removed.

### Trim and constant volume together

On the same source roster, apply the scheduled trim followed by gains `5/8`, `7/10`, and `9/8` in repeating group order. Use exact rational multiplication, round to the nearest even integer and clamp to the 16-bit range. The combined check aligned 10 of 12 untouched copies and reported zero residual differences in all 10. The other two would clip. It aligned and found the known splice in 8 of 12 generated, 8 of 12 same-reader and 8 of 12 other-reader edits. The earlier comparison left all these trim-plus-gain pairs unassessed. All 48 incompatible controls remained unassessed. They consisted of 12 internal cuts, 12 different-source pairs, 12 large edge trims and 12 time-varying gain pairs. For the last control, apply an additional `9/10` gain before sample 48,000 in the untrimmed source.

Independent sample readback verified both supporting intervals and every displayed audio excerpt. The submitted synthetic inserts contained 94,443 differing samples after subtracting the known gain. The report counted 94,429 of those samples as incompatible with every supported gain. The app leaves a sample out when it could match within the range of gains. A separate Chrome check downloaded PDF and JSON with identical source coordinates and two playable excerpts for the adjusted difference. Offline JSON replay reproduced the comparison and asset bytes.

## Replay without the app

Use the project Python environment, FFmpeg, and both original files. The first command writes a directory of measurements and excerpts. The second verifies a browser JSON export, including the exact bytes of its embedded media. It also accepts a directory created by the first command.

```sh
cd backend
python -m scripts.compare_audio_references earlier.wav analyzed.wav evidence-folder
python -m scripts.compare_audio_references earlier.wav analyzed.wav chipmunk-verify-audio-comparison.json --verify
```

Replay uses the app's measurement implementation and supports all five export versions. Record the FFmpeg version if runs produce different decoded bytes. Keep private audio and exports in your local workspace.

For a measured version 5 stereo section, a separate reader needs only Python 3.11 or newer, FFmpeg and FFprobe. From the repository root, give it both original files and the JSON downloaded from the browser. It checks the original hashes, decoded channel bytes and clocks, every displayed interval, waveform and WAV against the originals. It verifies the channel comparison only.

```sh
python3 -B -IS backend/scripts/verify_stereo_reference.py earlier.wav analyzed.wav chipmunk-verify-audio-comparison.json
```
