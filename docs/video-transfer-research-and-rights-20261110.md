# Recent video evaluation methods and source admission

The next HD study should preserve separate visual and audio edit clocks,
include untouched and ordinary controls from the same source, and measure
false alarms on the unchanged stream. These tests apply to the proposed
[in-envelope video study](video-in-envelope-transfer-design.md).

The October 2026 [AV-FABLE preprint](https://arxiv.org/html/2610.33413v1)
uses pristine/manipulated twins that pass through the same export process.
It evaluates stream-specific time ranges and checks for shortcuts using file
metadata. Its soundtrack-only test found that a cross-modal readout could
raise visual alarms even when pictures were unchanged. Apply those ideas
locally: compare separate and combined readouts on exact same-input controls,
measure excess flagged time on the untouched track, and test whether a
metadata-only arm can predict condition. For long files, report both errors
per clip and time falsely flagged. A low alarm rate can coexist with missed
edits. The paper's license does not establish rights to the source videos,
weights, or constructed media; none were imported for this evaluation.

[TalkingHeadBench](https://openaccess.thecvf.com/content/WACV2026/papers/Xiong_TalkingHeadBench_A_Multi-Modal_Benchmark__Analysis_of_Talking-Head_DeepFake_Detection_WACV_2026_paper.pdf)
separates held-out identities, held-out generators, and both shifts together.
Its reported practical operating points and background saliency shifts argue
for keeping ordinary background and compression controls in the next test.
The [dataset card](https://huggingface.co/datasets/luchaoqi/TalkingHeadBench/blob/main/README.md)
lists an MIT metadata tag, requires compliance with upstream licenses, and
describes a 2026 audio re-extraction after synchronization problems. The
[paper supplement](https://openaccess.thecvf.com/content/WACV2026/supplemental/Xiong_TalkingHeadBench_A_Multi-Modal_WACV_2026_supplemental.pdf)
describes a different release license and restricted underlying assets. The
historical fetch plan's MIT tag is not sufficient to admit that material into
a commercial test. Treat any future use as a new per-asset rights and timing
review. The archived benchmark records remain historical.

[DeepSpeak v2](https://www.ischool.berkeley.edu/research/publications/2025/deepspeak-dataset)
and [DeepSpeak-Agentic](https://huggingface.co/datasets/faridlab/deepspeak-agentic)
are methodology references. The former offers research and noncommercial use;
the latter has separate licensing for nonacademic institutions. The Agentic
protocol reinforces splitting by recording session and controlling codec and
filename cues. Neither is admitted here. [FaceStar](https://github.com/facebookresearch/facestar#license)
is noncommercial. [SpeakingFaces](https://issai.nu.edu.kz/download-speaking-faces/)
is labeled CC BY 4.0, but its [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8156799/)
reports a native visible-video height of 512 pixels. It cannot fill this HD
cohort by upscaling.

[FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) lists CC BY
source videos and resolution metadata. Access requires a gated agreement, and
the publisher's description alone does not verify each underlying recording.
Commons has individual license and source pages, such as [this native 720p
interview listing](https://commons.wikimedia.org/wiki/File:Interview_WMF.ogv).
These are screening leads. Admit only after verifying creator authority,
attribution, picture and audio rights, restrictions on derivative use, and the
rights relevant to the intended use of people's likenesses. Creative Commons
notes these may be separate from copyright in its [CC BY deed](https://creativecommons.org/licenses/by/4.0/).
No assets were downloaded, no access agreements were accepted, and nobody was
contacted for this review.
