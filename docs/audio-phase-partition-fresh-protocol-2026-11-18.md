# Fresh noisy-speech phrase localization protocol

This protocol was fixed before scoring the reserved speech. The study asks
whether two measurements of one noisy recording can improve the location of
a short synthetic replacement over the current two-second speech check.
The statistical assessment remains separate from these candidate locations.
[Earlier ordinary-control and diagnostic results](audio-phase-partition-diagnostic-2026-11-18.md)

## Exact matched inputs

The reserve contains 24 utterances from eight readers, with exact published
word alignments at 16 kHz. Each utterance has two parallel synthetic versions,
one from MeloTTS and one from ChatterboxTTS. The native original supplies the
12-second context. Each synthetic phrase is fitted to the same original
time interval. Each generator has a matched ordinary round-trip resampling
control using its corresponding native phrase duration. The source publisher
provides the parallel recordings and alignment files. [PhonemeDF paper](https://aclanthology.org/2026.lrec-1.469/) · [Publisher dataset](https://zenodo.org/records/19522979)

The 24 utterances each contribute five clean, six traffic-noise, and six
competing-speech deliveries. The noisy conditions include the background
alone. Interference is set to −10 dB foreground-to-background ratio within
the original phrase. All matched roles share the same comparison interval
and gain. The competing-speech schedule uses two recordings with annotated
overlap. At least half of the phrase overlaps genuine background speech in
21 of 24 scheduled cases. Familiar background labels and visible waveforms
carry no AI-authorship weight.

## Decision rule and independent review

The analysis compares the original mixture, a projected speech view, and
the same view when its response exceeds both the original mixture's
response and any measured remainder response. The new locations must be
supported by two neighboring windows. Each fixed point comes from the
previous ordinary-control selections. The new evaluation holds those
points constant across all three conditions.

The prospective comparison adds the contrast-gated locations to the
current check. Advancement requires at least three newly located edits
across at least three readers and both generators in competing speech.
Every newly located competing-speech edit needs a clean same-time result
on its matched ordinary and background controls. Each condition also needs
at least 80% assessed coverage of edited and ordinary files, zero additional
flags on isolated backgrounds, and at most two newly flagged groups of
ordinary recordings.
These are evaluation criteria for this panel. The result would authorize a
separate app replay before any change to the served check.

The local reviewer independently rebuilt the points from the earlier
portable receipt and agreed with the scorer on two authored 408-row
fixtures, including a same-time ordinary alert. After the real comparison
closes, a separate diagnostic can measure how much of each known source
remains in the selected and remainder waveforms. Recent separation research
motivates checking the remainder directly. Its reported score measures
separation on two-speaker mixtures, while our outcome measures phrase
localization and ordinary alerts. The study uses the already reviewed
enhancement checkpoint. [November speech-separation study](https://arxiv.org/html/2611.09358v1)

| Frozen input | SHA-256 |
| --- | --- |
| Reserve and word alignment selection | `5965ed2cb76c48257a46f670e8012ab9ed6456eb142b23e447d69aa5c7885931` |
| Background recording and timing lock | `50f22b7a38ea30365a599cdf8da760361d10ad4262a7c8fc17dc282ff942bfa6` |
| Input renderer | `3fbb36acb6aea11c6229bf7f165c0bb53b22624a73dd327c8baa6a27854d0836` |
| Fresh comparison scorer | `9a13dc6567aca9507421041e3beff25d42d8de03d68c020225c4723a1ffce81a` |
