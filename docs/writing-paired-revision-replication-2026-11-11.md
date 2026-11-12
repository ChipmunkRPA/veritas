# Replicating the paired writing check

The earlier two-version check improved localization on one generator family using dated F1000Research revisions. This study tested two different Apache-2.0 generators on those same nine article groups and prepared source-disjoint pilot pairs from PeerJ and Copernicus. The F1000Research results measure generator transfer on exposed texts. The new publisher pilot contains one independent work per publisher. [Earlier study and exact draft comparison](writing-reference-revision-transfer-2026-11-11.md)

## Frozen comparison

The current rule keeps the first six positive native inputs at its existing raw comparison point. Same-window agreement additionally requires the second detector to be positive on the same input. The paired check compares the earlier and later native contexts with exact unchanged neighboring sentences. It marks the later center sentence only when both model-specific AI-like odds at least double. The combined arm takes the paired center and agreement windows whose exact input changed. A useful generated location requires at least 0.30 non-whitespace character IoU and at least 0.50 recall against the known replacement. An adequate additional generator lane must admit at least six groups, add useful locations in at least two groups, preserve existing useful locations, and add at most one human false-flag group over same-window agreement. The exact draft comparator separately tests wording changes in both versions.

Each generator made one rewrite attempt for each earlier center. The source and asset freeze fixed the exact prompt, seeds and identities before generation. Valid rewrites had to complete in one sentence, differ lexically from their source and from the publisher’s human-edited center, and preserve four exact flanking sentences. Both inference snapshots ran offline with one CPU thread. On F1000Research, prior checkpoint responses were reused only when the native input hash matched byte for byte. [Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306) · [SmolLM2 checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

## Results on the exposed source groups

| Case group | Admitted | Current useful | Same-window useful | Paired useful | Combined useful |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen rewrites | 7/9 | 0/7 | 0/7 | 2/7 | 2/7 |
| SmolLM2 rewrites | 6/9 | 1/6 | 1/6 | 1/6 | 1/6 |

The Qwen lane met the frozen generator replication criterion. The paired check found two exact centers and dropped the wrong-only location that appeared under same-window agreement. The combined arm retained that wrong-only location. The SmolLM2 lane met the size requirement and preserved its one useful current location. It produced no additional useful locations. Across the earlier and additional generator lanes, the paired check localized 7 of 24 generated edits and the current check localized 2 of 24. All 24 edits draw on nine article groups, and several share the same starting text.

The current check flagged two of nine human-edited files. Same-window, paired, and combined flagged zero of nine. The current and same-window checks respectively flagged three and one of the earlier human files. Before-versus-self is an unchanged-scope control. The existing app’s exact comparator found 46 hunks within the registered center on both sides of all 22 edited pairs. That includes nine natural human revisions and thirteen generated replacements.

The diagnostic checked 111 distinct native inputs per checkpoint, using 72 previously committed exact responses and 39 newly scored inputs per checkpoint. The new inputs required 20 CPU forwards in total. An independent replay checked the exact contexts, cached responses, source and generator bindings, and all outcome counts without loading the model weights.

## Publisher source separation

Two fixed Crossref member queries located versions explicitly linked by the publisher. The PeerJ lane uses version 2 linked to its exact version 1. The Copernicus lane uses a discussion preprint linked to a final revised paper. Both require the original author list, a dated revision before 2020, and a per-version CC BY 3.0 or 4.0 term. The study uses the publishers’ deposited abstracts. The PeerJ primary XML and PDF returned HTTP 403. The first Crossref counterpart attempt returned HTTP 429 and the frozen roster resumed after a pause. The Copernicus preprint XML sampled in source discovery had an empty body. [PeerJ version record](https://api.crossref.org/works/10.7287/peerj.preprints.2225v2) · [Copernicus preprint record](https://api.crossref.org/works/10.5194/tc-2017-280) · [Copernicus publication policy](https://publications.copernicus.org/for_authors/publication_policy.html)

Each fixed catalog contributed 48 candidates. The source replay checked exact offsets, publisher rights and dated author lists, rejected known overlaps with the F1000Research development sources and six project panels, and accepted one pair per publisher. Forty-one PeerJ and 39 Copernicus records lacked the exact edit geometry. Six PeerJ and eight Copernicus records failed metadata or parse checks. The resulting two-group pilot gives no estimate of publisher-level false alarms or generalization.

Both generators attempted both source groups. The pilot admitted one Qwen rewrite per publisher and one SmolLM2 rewrite on Copernicus. The paired check localized the PeerJ Qwen edit at its exact center. Current and same-window agreement missed it, and all three rules missed both generated edits on the Copernicus work. All three rules left both natural later revisions unflagged. The app’s exact draft comparator placed ten change hunks inside the registered centers of all five edited pairs. The independent audit covered seven files and 25 exact inputs per checkpoint. This pilot ran 14 new CPU forwards with no cached responses. [Audited results](evaluation-panels/writing-paired-revision-replication-2026-11-11.json)

The paired response transferred to Qwen on the exposed F1000Research sources and one new PeerJ work. The SmolLM2 lane added no useful locations on the exposed groups. Larger publisher samples and more varied natural revisions are still needed before adding a paired model result to user reports or confidence. The exact draft comparator remains available as source evidence when the user supplies both versions.

## Replay

Use the saved publisher catalog, counterpart records, and source receipt to reproduce qualification with the six committed history files and the prior F1000Research manifest. The diagnostic `source` is the completed F1000Research run. The holdout `source` is the separate qualified publisher directory. Use the existing local checkpoint snapshots and the saved receipts for a completed run.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
backend/.venv/bin/python -B -m scripts.writing_publisher_revision_holdout verify "$holdout" --exposed "$f1000/originals.json" "${history_args[@]}"
backend/.venv/bin/python -B -m scripts.writing_paired_revision_replication verify "$run" "$source" --scope "$scope" --current "$current" --second "$second"
backend/.venv/bin/python -B -m scripts.review_writing_paired_revision_replication "$run" "$source"
```

For a fresh replay of inference, run `freeze`, generate once per family in the offline MLX environment, then run `freeze-scoring`, `current`, `second`, and `report`. For the holdout generation, the MLX environment also needs the local XML parser used by the publisher source module. Model and publisher source bytes stay local.
