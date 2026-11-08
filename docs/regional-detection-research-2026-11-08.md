# Regional detection research checked on 8 November 2026

These primary-source findings guide the pending learning experiment. They do
not establish that Chipmunk Verify reproduces a paper's results or qualifies a
new model. No paper checkpoint or new dataset media is imported by this note.

## Background shortcuts: Inpainting Exchange

[INP-X v2, 4 November 2026](https://arxiv.org/html/2602.00192v2), reports that
restoring original background pixels can sharply reduce detection of an
otherwise unchanged generated region. Its controlled comparisons distinguish
generated content from a whole-image autoencoder artifact. Localization and
mixed-distribution training help, but the authors still report weak transfer
to unseen editors. These are their results, not measurements of our app.

The application to our experiment is to require actual positive changes inside
the declared source mask, verify unchanged background pixels, and evaluate
localization and ordinary-edit false alarms separately. Our frozen v21 pair
qualification already checks that containment. A passing pair check is still
not successful detection. Future comparisons should preserve mask-size strata
and separate compression effects from the inserted-region target.

The [official repository at bf5c461](https://github.com/emirhanbilgic/INP-X/tree/bf5c461d54ef3b5afd45ac97cf32fc2864c218e4)
advertises MIT in its README, but its complete tree has no LICENSE file or
released training/testing modules. The README leaves that release pending and
links a separate notebook. We do not treat the badge as a fully verified
commercial detector release, copy its snippet, or import its benchmark media.

## Sequential edits and mask semantics: DiffSeg30k

[DiffSeg30k](https://arxiv.org/html/2511.19111v2) evaluates local edits from eight
generators, including sequential edits, compression and unseen generators.
Its real-source and AI-source image subsets are distinct. The relevant lessons
are source-connected splitting, preservation of edit history, separate distorted
formats and reporting by source type. We should not collapse multiple edited
versions of one source into independent examples.

The [publisher's data card](https://huggingface.co/datasets/Chaos2629/Diffseg30k/blob/main/README.md)
declares Apache-2.0 and documents mask values for editing generators. This is a
promising later data route, not automatic clearance for all underlying photos
or generator outputs. It needs release-specific provenance/permissions and
source/role checks before acquisition. No media or weights have been downloaded.

One particularly important interpretation is that a zero-valued background in
an edit mask does not authenticate those pixels: an AI-source image may already
be generated before a later local edit. This is our inference from the paper's
source categories and card's mask semantics. The experiment must distinguish
"not labeled as this edit" from "proven human/camera origin."

## Consumer phone edits: methodology only

[ConsumerGenEdit / Edits in the Wild](https://aclanthology.org/2026.emnlp-main.1661/)
covers edits on Apple, Google and Samsung devices, paired sources, several edit
operations and learned-compression variants. Its
[official project page](https://shieldlab.github.io/On-Device-AI-Edit-Forensics/)
distinguishes reference-assisted from reference-free evaluation. A comparison
given the original photo is a different product capability from analyzing only
an uploaded image; results must not be transferred between them.

The [dataset page](https://shieldlab.github.io/On-Device-AI-Edit-Forensics/dataset.html)
restricts use to non-commercial research and prohibits redistribution. We do not
download or train/test the commercial project on that media. The reusable ideas
are the operation taxonomy, chained-device stress tests, realistic compression
and separate human-reference assessment. A commercial study would need its own
permitted sources and labels. No contact or access request has been sent.

The publisher PDF could not be retrieved by the browsing tool because of its
size. The claims above use the accessible official abstract, project, data and
code pages; this note does not claim a complete PDF/code audit.

## Current action

Finish and independently check the source-bound v21 pairs, then run the frozen
v22 source graph and an authored gradient probe. The planned experiment needs
matched ordinary controls, held-out source components, its own calibration,
localization measurements and an honest unseen-source/generator comparison.
Neither model execution, source qualification nor a paper's reported accuracy
is itself a customer confidence score. All new study roles and admission rules
must be fixed before project fitting. Existing checks remain in service.
