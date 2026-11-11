# Native GRID pilot clip assembly

Assemble each of the eleven fixed v143 cases for each pilot clip. Verify the
completed v143 plan and the published v144, v146, v147 and v148 procedures at
admission. Require the actual matching producer profiles, receipts and private
journals for each clip before its worker starts. A missing procedure or producer
receipt leaves that stage pending. Preserve every unavailable condition with
its reason. A neural variant requires all twenty of its own producer pictures
and masks.

Load all seventy-five original PNGs from the saved native bank. Verify file
SHA-256, plain RGB bytes and observed integer PTS. Convert the observed rational
video and audio clocks exactly to the millisecond positions supported by v119.
Keep the source time base, PTS, waveform length and separate audio origin in
private metadata. Record an unavailable clock when v119 cannot represent it
exactly. Keep the v119 constructed terminal picture duration explicit. It is
the upper median picture interval used at muxing. The decode check verifies
picture PTS and PCM, with no terminal packet-duration measurement. The v151
scorer requires separately observed support to score a terminal-cell window.

For each neural condition, verify the saved crop, native composite and mask
files against its producer commitments. Reconstruct the composite using the
same bound original picture, inverse affine, exact source-mouth alpha and
planned opacity. The binary mask must equal every changed native pixel.
Check that changed pixels stay inside the corresponding alpha. Record zero
effects and distinguish verified neural construction from observed pixel change.

For the resample and brightness conditions use the corresponding branch's
cached **original** crop, inverse affine and native alpha. LivePortrait resizes
the 256-pixel original crop to 512 using linear interpolation before applying
the recorded 512 affine inverse. MuseTalk uses its original 256 affine inverse.
Apply `np.rint(resampled_rgb * 1.1)` with clipping to uint8 for brightness before
alpha composition. Use half opacity at the first and last edit pictures and
full opacity at the interior pictures. The genuine cut copies the actual
picture eight positions earlier at full opacity and retains the output PTS.
Verify the earlier source clock is exactly 320 ms before each output clock.

Verify the actual v147 full WAV and float32 PCM payloads and compare them against
the independently validated source PCM. The delayed case uses the complete
v147 delayed waveform. Verify its copied interval and unchanged outside values
against v143. Other cases use the complete original waveform. Save the output
clock mapping and the separate sample-value provenance for each ready case.

Reuse the actual v119 qualification and its installed isolated PyAV runtime.
Use its unchanged FFV1 and mono 16 kHz signed-16 PCM Matroska mux. Verify every
emitted picture and sample through the v119 decode round trip. Inspect the
actual streams and tags. Numeric filenames come from a deterministic
permutation bound to the private source commitment. Case labels and edit masks
belong only in the private journal and artifacts on Seagate. A file with a
failed verification leaves a failed stage and its existing artifacts for review.

Run one exclusive `clip-OO/bind-00` stage at a time through the unchanged v108
owned monitor. Use 900 seconds, 1.5 GiB sampled descendant RSS, 3 GiB initial
RAM and project/archive disk, and 1.5 GiB running reserves. Require successful
prior clip completion for the fixed order. Sampled memory is a lower bound.
Keep failed attempts and their partial artifacts. The worker measures no
achieved motion, synchronization or detection performance. Publish aggregate
counts and whole-artifact commitments. Clips, masks, source clocks, labels and
individual fingerprints remain private.

## Consumer API

Import `scripts.grid_native_clips_v149`. Call `load_completed(ordinal)` with the
original v143 pilot ordinal. It binds the actual successful stage and returns
eleven ordered private case records. The consumer verifies payload bytes
inside its own guarded worker.

- The trial is `backend/data/independent-evaluation/grid-native-clips-v149`.
  Each `clip-OO/bind-00` has the actual profile and result receipt.
- The private cache is `grid-native-clips-v149/clip-OO` on Seagate. Its
  `input-binding.json` binds all actual per-clip producer metadata. Its
  `cases.jsonl` retains every case and the numeric filename mapping.
- Case schema `chipmunk.grid-native-clip-case/149` includes `asset_index`
  (original v143 ordinal), `source_slot`, `component_index`, `study_role`,
  `condition`, `status`, `reason`, `waveform_id`, and construction provenance.
- A ready case has `file`, `file_sha256`, `expected` (v119 output PTS, RGB,
  signed-16 PCM, audio origin, length and constructed terminal picture duration),
  `review` and `uniform_lossless_header_verified`. The original clock is at
  `source_video_time_base` and `source_video_pts`.
- `edit_proofs` has the twenty visual replacement positions for visual cases.
  Each record has native `frame_index`, `pts`, `mask_file`, `mask_sha256`,
  `mask_raw_sha256` and `changed_pixels`. Original and audio-only cases retain
  empty lists. `changed_native_pixels`, `changed_video_frames` and
  `changed_audio_samples` report actual observed differences.
- `planned_neural_provenance` identifies the planned operation.
  `neural_provenance_verified` requires complete verified neural output and
  `observed_neural_pixel_change` records whether any native pixels changed.
  `audio_output_clock_mapping` and `audio_value_mapping` retain separate clock
  and sample-origin evidence. The `waveform_id` selects the matching v147 state.
- `source_time_bounds_seconds` and `before_during_after` retain the planned
  edit context where applicable. Unavailable cases have an explicit reason
  and omit the completed payload fields.
