# Test whether screenshot edit locations survive framing changes

The preceding photo-only routing trial found nine useful sharp locations in full screenshots and one in their exact photo crops. Eight of the nine full-image locations used a measured side near the photo frame. This experiment holds every photo pixel fixed and changes the surrounding frame to test whether those locations depend on the original framing. [Previous screenshot routing study](image-viewport-routing-transfer-2026-11-12.md) · [Reproducible scalar results](evaluation-panels/image-viewport-frame-perturbation-2026-11-12.json)

## Frozen conditions

Use all 12 archived Chromium screenshot twins from the preceding comparison. The nine previously useful sharp cases are an exposed discovery stratum. Separately construct matched ordinary and edited photo panels from six source-disjoint, already qualified archival groups, with three SDXL and three Kandinsky sources. Those six source photos were outside the previous viewport trial and have earlier project exposure. Their outer screen panels are simple authored raster fixtures. [Qualified photo source terms](image-background-patch-transfer-2026-11-11.md)

For each twin, hold the photo and every changed pixel at the same native coordinates. Change the 16 pixels immediately outside its rectangle in one of three ways. Add a dark rim, add a light rim, or extend clean photo texture by 16 pixels using reflection at the boundary. The reflected apron is identical in both members of each pair. It moves the external texture transition while preserving the original photo and edit mask. The procedure verifies the full RGB difference mask after every intervention.

The existing sharp boundary check uses the same operating point, window sizes and scoring on all variants. It selects a new region on each image. Independently recompute the four side responses at the initial edited image's selected coordinates in both members of each pair. This records whether the same side evidence persists even if the selected rectangle changes. A useful location needs at least 0.30 intersection over union and 0.50 recall against the exact native edited pixels. A substantially overlapping selected location reaches 0.50 IoU with the initial selected box. The experiment uses no new model weights and changes no serving behavior.

## Results

| Frame | Discovery useful edits /12 | Discovery ordinary flags /12 | Transfer useful edits /6 | Transfer ordinary flags /6 |
| --- | ---: | ---: | ---: | ---: |
| Original frame | 9 | 10 | 4 | 5 |
| Dark rim | 9 | 10 | 4 | 6 |
| Light rim | 9 | 10 | 4 | 5 |
| Reflected photo apron | 10 | 10 | 6 | 6 |

All nine useful discovery regions and all four useful transfer regions were still selected as useful on each of the three interventions. Every retained location reached 0.50 IoU with its original selected rectangle. In the discovery group the exact box was unchanged in seven dark, eight light and eight apron cases. In transfer the corresponding counts were three, four and three.

Each intervention changed 11 of 36 fixed side pixel inputs on the previously useful discovery boxes and five of 16 on the transfer boxes. At the original coordinates, dark and light rims preserved all 13 edited flags. The texture apron preserved 12 of 13. The detector recovered the remaining case with a shifted useful box. None of those 13 fixed boxes flagged on the corresponding ordinary image under any frame. The ordinary whole-image flag counts remain high, so stable locations on these edits give no basis for enabling this as a customer filter.

This result narrows the prior viewport observation. The exact crop removed much of the whole-screenshot check's useful behavior. Moving or recoloring the frame while preserving the original image extent left the useful selections stable on this panel. The audit measures these particular authored interventions and source pairs.

## Reproduce

The freeze is `f505dab1951a47498d44b9216df685b0e40bde0250ef7ddcc4c8432e326a9a12`. The local run checked 18 source groups, 72 clean/edited variant pairs and 144 full native checks. A separate invocation reproduced every case, side-input commitment and aggregate. Its canonical record digest, excluding elapsed time, is `0465028d5ebbc63ebcc18b9a64f100a99c9c7e6aeb8ae74a1b377b86c5759b3f`. The portable record includes four-side scalar measurements and whether each fixed side's pixels differ from the initial framing. Photo pixels, masks and individual image hashes stay local.

```sh
python3 -I -S backend/scripts/image_viewport_frame_perturbation.py review \
  docs/evaluation-panels/image-viewport-frame-perturbation-2026-11-12.json
```

Use the earlier viewport run directory, its two existing qualified photo archives and a new private output directory.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for stage in freeze run; do
  backend/.venv/bin/python -B -m scripts.image_viewport_frame_perturbation "$stage" \
    "$previous" "$earlier" "$later" "$run" || exit 1
done
```
