"""Run public ledger checks and save a privacy-safe receipt for independent review."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PANELS = "docs/evaluation-panels"
SCRIPTS = "backend/scripts"
SCHEMA = "chipmunk.public-evaluation-replay/1"
SCOPE = "Arithmetic replay of published evaluation ledgers"
SHA256 = re.compile(r"[0-9a-f]{64}")
CHECKS = {
    "image": ("review_image_thumbnail_extent_transfer.py", ("image-thumbnail-extent-transfer-2026-11-12.json",), ()),
    "video": ("review_video_motion_late_transfer.py", ("video-motion-late-transfer-2026-11-12.json",),
              (f"{SCRIPTS}/review_video_lossy_sequence_transfer.py",)),
    "audio": ("audio_vocal_event_mixtures.py", ("audio-vocal-event-mixtures-2026-11-12.json",),
              (f"{PANELS}/audio-vocal-event-source-roster-2026-11-12.json",
               f"{PANELS}/audio-remote-neighbor-replication-2026-11-12.json")),
    "writing": ("review_writing_elife_exact_sentence_confirmation.py", ("writing-elife-exact-sentence-confirmation-2026-11-12.json",),
                (f"{PANELS}/writing-multiblock-independent-2026-11-12.json",)),
}
LOCALIZATION = {
    "image": ("review_image_boundary_multiple_transfer.py", "image-boundary-multiple-transfer-2026-11-12.json"),
    "video": ("review_video_reference_soundtrack_panel.py", "video-paired-soundtrack-transfer-2026-11-12.json"),
    "audio": ("review_audio_partialedit_dense_transfer.py", "audio-partialedit-dense-transfer-2026-11-12.json"),
    "writing": ("review_writing_source_conditioned.py", "writing-source-conditioned-2026-11-12.json"),
}
TRANSFERS = {
    "image": ("review_image_png_layout_guard.py", "image-png-layout-guard-2026-11-12.json", (), (
        f"{SCRIPTS}/image_png_layout_guard_transfer.py", f"{SCRIPTS}/capture_image_png_layout_controls.cjs",
        f"{PANELS}/image-v7-smaug-genuine-operating-point.json", f"{PANELS}/image-whole-png-scope-probe-2026-11-12.json",
        f"{PANELS}/image-whole-png-transfer-2026-11-12.json", f"{PANELS}/image-whole-fresh-sources-2026-11-12.json",
    )),
    "video": ("review_grid_phone_viseme_timing.py", "grid-phone-viseme-timing-2026-11-12.json", (), ()),
    "audio": ("audio_short_coarse_transfer.py", "audio-short-coarse-transfer-2026-11-12.json", ("review",), (
        f"{SCRIPTS}/review_audio_reduced_background_localization.py", f"{PANELS}/audio-noise-window-transfer-2026-11-12.json",
    )),
    "writing": ("writing_local_lexical_transfer.py", "writing-local-lexical-transfer-2026-11-12.json", ("review",), (
        f"{PANELS}/writing-elife-exact-sentence-confirmation-2026-11-12.json",
        f"{PANELS}/writing-multiblock-independent-2026-11-12.json",
    )),
}
SOURCE_EVIDENCE = {
    "image": ("review_image_png_exif_preview_transfer.py", (
        "image-png-exif-preview-2026-11-12.json", "image-png-exif-preview-cases-2026-11-12.json",
    ), (), ()),
    "video": ("review_video_face_paired_edge_transfer.py", (
        "video-face-paired-edge-hevc-confirmation-2026-11-12.json", "--confirm-against", "video-face-paired-edge-transfer-2026-11-12.json",
    ), (), ()),
    "audio": ("audio_dense_evidence_confirmation.py", ("audio-dense-evidence-confirmation-2026-11-12.json",), ("review",), (
        f"{SCRIPTS}/audio_vocal_event_mixtures.py", f"{SCRIPTS}/review_audio_reduced_background_localization.py",
        f"{PANELS}/audio-partial-publisher-transfer-2026-11-12.json",
        f"{PANELS}/audio-reduced-background-localization-2026-11-12.json", f"{PANELS}/audio-noise-window-transfer-2026-11-12.json",
    )),
    "writing": ("writing_mixed_pdf_page_transfer.py", ("writing-mixed-pdf-page-transfer-2026-11-12.json",), ("review",), (
        f"{PANELS}/writing-agreement-document-format-transfer-2026-11-12.json",
    )),
}
RECENT_CHECKS = {
    "image": ("image_gainmap_cross_camera.py", "image-gainmap-cross-camera-2026-11-12.json", ("review",), (
        f"{SCRIPTS}/image_gainmap_iso.py", f"{SCRIPTS}/image_gainmap_localization_transfer.py",
        f"{PANELS}/image-gainmap-localization-2026-11-12.json",
    )),
    "video": ("review_video_face_unrouted_short_transfer.py", "video-face-unrouted-short-transfer-2026-11-12.json", (), ()),
    "audio": ("audio_transient_reduction_transfer.py", "audio-transient-reduction-transfer-2026-11-12.json", ("review",), (
        f"{SCRIPTS}/audio_subband_partial_transfer.py", f"{SCRIPTS}/audio_focused_window_transfer.py",
        f"{SCRIPTS}/audio_dense_evidence_confirmation.py", f"{SCRIPTS}/audio_vocal_event_mixtures.py",
        f"{SCRIPTS}/review_audio_reduced_background_localization.py",
        f"{PANELS}/audio-noise-window-transfer-2026-11-12.json", f"{PANELS}/audio-focused-window-transfer-2026-11-12.json",
    )),
    "writing": ("review_writing_paired_center_transfer.py", "writing-paired-center-transfer-2026-11-12.json", (), ()),
}
EVIDENCE_FOLLOWUPS = {
    "image": ("image_gainmap_perimeter_transfer.py", "image-gainmap-perimeter-transfer-2026-11-12.json", ("review",), (
        f"{SCRIPTS}/image_gainmap_cross_camera.py", f"{SCRIPTS}/image_gainmap_iso.py",
        f"{SCRIPTS}/image_gainmap_localization_transfer.py",
        f"{PANELS}/image-gainmap-cross-camera-2026-11-12.json", f"{PANELS}/image-gainmap-localization-2026-11-12.json",
    )),
    "video": ("review_video_mouth_sequence_transfer.py", "video-mouth-sequence-reuse-transfer-2026-11-12.json", (), ()),
    "audio": ("audio_reflected_local_transfer.py", "audio-reflected-local-transfer-2026-11-12.json", ("review",), (
        f"{SCRIPTS}/audio_transient_reduction_transfer.py", f"{SCRIPTS}/audio_subband_partial_transfer.py",
        f"{SCRIPTS}/audio_focused_window_transfer.py", f"{SCRIPTS}/audio_dense_evidence_confirmation.py",
        f"{SCRIPTS}/audio_vocal_event_mixtures.py", f"{SCRIPTS}/review_audio_reduced_background_localization.py",
        f"{PANELS}/audio-noise-window-transfer-2026-11-12.json", f"{PANELS}/audio-focused-window-transfer-2026-11-12.json",
    )),
    "writing": ("review_email_body_alternatives.py", "email-body-alternatives-transfer-2026-11-12.json", (), ()),
}
SHORT_EDITS = {
    "image": ("review_image_thumbnail_extent_transfer.py", "image-thumbnail-extent-transfer-2026-11-12.json", (), ()),
    "video": ("review_video_lipsync_repeat_audio.py", "video-lipsync-repeat-audio-2026-11-12.json", (), ()),
    "audio": ("review_audio_modulation_partial_transfer.py", "audio-modulation-partial-transfer-2026-11-12.json", (), ()),
    "writing": ("review_writing_two_context_transfer.py", "writing-two-context-transfer-2026-11-12.json", (), ()),
}
DIFFICULT_EDITS = {
    "image": ("review_image_masked_context_public.py", "image-masked-context-transfer-2026-11-13.json", (), ()),
    "video": ("review_video_mouth_chroma_coupling.py", "video-mouth-chroma-coupling-2026-11-13.json", (), (
        f"{SCRIPTS}/review_video_mouth_contour_phase.py",
    )),
    "audio": ("review_audio_unassigned_stereo_voice.py", "audio-unassigned-stereo-voice-2026-11-13.json", (), ()),
    "writing": ("review_writing_position_qwen.py", "writing-position-qwen-confirmation-2026-11-12.json", (), (
        f"{SCRIPTS}/review_writing_position_confirmation.py",
        f"{PANELS}/writing-position-qwen-admission-2026-11-12.json",
        f"{PANELS}/writing-two-context-transfer-2026-11-12.json",
    )),
}
LOCALIZATION_LIMITS = {
    "image": ("review_image_regional_tile_rescue_main.py", "image-regional-tile-rescue-complete-2026-11-13.json", (), ()),
    "video": ("review_video_mouth_luma_condition.py", "video-mouth-luma-condition-h264-2026-11-13.json", (), (
        f"{SCRIPTS}/review_video_mouth_track_cycle.py",
    )),
    "audio": ("review_audio_room_stress_transfer.py", "audio-room-stress-transfer-2026-11-13.json", (), ()),
    "writing": ("writing_entropy_conditioned_transfer.py", "writing-entropy-conditioned-2026-11-13.json", ("verify",), (
        f"{SCRIPTS}/review_writing_long_position.py",
        f"{PANELS}/writing-long-position-transfer-2026-11-12.json",
        f"{PANELS}/writing-f1000-position-transfer-2026-11-13.json",
        f"{PANELS}/writing-elife-long-position-transfer-2026-11-13.json",
        f"{PANELS}/writing-f1000-revision-spans-2026-11-13.json",
    )),
}
SPECIALIST_TRANSFER = {
    "image": ("review_image_flame_jpeg_transfer.py", ("image-flame-jpeg-transfer-2026-11-13.json",), ()),
    "video": ("review_video_mouth_codec_prediction.py", ("video-mouth-codec-prediction-2026-11-13.json",), ()),
    "audio": ("review_audio_wpe_room_transfer.py", (
        "audio-wpe-room-transfer-2026-11-13.json", "--ancestor", "audio-room-stress-transfer-2026-11-13.json",
    ), ()),
    "writing": ("review_writing_revision_trajectory_transfer.py", ("writing-revision-trajectory-2026-11-13.json",), (
        f"{SCRIPTS}/writing_revision_trajectory_transfer.py",
        f"{PANELS}/writing-multiblock-independent-2026-11-12.json",
    )),
}
LOCAL_EVIDENCE_2026_11_13 = {
    "image": ("review_image_structure_jpeg_totals.py", ("image-reference-structure-jpeg-held-2026-11-13.json",), (), ()),
    "video": ("review_video_scene_return_transfer.py", ("video-scene-return-transfer-2026-11-13.json",), (), ()),
    "audio": ("review_audio_odss_matched_room.py", ("audio-odss-matched-room-2026-11-13.json",), (), ()),
    "writing": ("review_writing_revision_worker_transfer.py", (
        "writing-revision-worker-held-batch0-2026-11-13.json", "writing-revision-trajectory-2026-11-13.json",
    ), ("review",), (f"{SCRIPTS}/writing_revision_worker_priority.py", f"{SCRIPTS}/writing_revision_worker_transfer.py",
                    f"{SCRIPTS}/writing_revision_trajectory_transfer.py", f"{SCRIPTS}/review_writing_revision_trajectory_transfer.py")),
}
EVIDENCE_TRANSFER_2026_11_13 = {
    "image": LOCAL_EVIDENCE_2026_11_13["image"],
    "video": ("review_video_paired_native_mouth_transfer.py", ("video-paired-native-mouth-transfer-2026-11-13.json",), (), ()),
    "audio": ("review_audio_flank_background_transfer.py", (
        "audio-flank-background-transfer-2026-11-13.json", "audio-odss-matched-room-2026-11-13.json",
        "audio-phonemedf-matched-transfer-2026-11-13.json",
    ), (), (f"{SCRIPTS}/review_audio_odss_matched_room.py", f"{SCRIPTS}/review_audio_phonemedf_matched_transfer.py")),
    "writing": ("review_writing_revision_worker_transfer.py", (
        "writing-revision-worker-complete-2026-11-13.json", "writing-revision-trajectory-2026-11-13.json",
    ), *LOCAL_EVIDENCE_2026_11_13["writing"][2:]),
}
EVIDENCE_EXTENT_2026_11_13 = {
    "image": ("review_image_native_stencil_interpolation_transfer.py", (
        "image-native-stencil-interpolation-measurements-2026-11-13.json",
        "image-native-stencil-interpolation-cc0-roster-2026-11-13.json",
        "image-native-stencil-interpolation-transfer-2026-11-13.json",
    ), (), ()),
    "video": ("review_video_reference_extent.py", (
        "video-reference-extent-transfer-2026-11-13.json", "video-reference-extent-controls-2026-11-13.json",
    ), (), ()),
    "audio": ("review_audio_odss_matched_room.py", ("audio-odss-matched-room-2026-11-13.json",), (), ()),
    "writing": ("replay_document_word_tables.py", (
        "writing-uswds-revision-transfer-2026-11-13.json", "document-word-table-comparison-2026-11-13.json",
    ), ("review",), (f"{SCRIPTS}/writing_uswds_revision_transfer.py",)),
}
LOCAL_EVIDENCE_EXPANSION = {
    "image": ("review_image_native_cfa_dimensions_transfer.py", (
        "image-native-cfa-dimensions-measurements-2026-11-13.json",
        "image-native-cfa-dimensions-cc0-roster-2026-11-13.json", "image-native-cfa-dimensions-transfer-2026-11-13.json",
    ), (), ()),
    "video": ("review_video_reference_robust_transfer.py", ("video-paired-local-area-transfer-2026-11-13.json",), (), ()),
    "audio": ("review_audio_phoneme_short_edit_intake.py", ("audio-phoneme-short-edit-intake-2026-11-13.json",), (), ()),
    "writing": ("replay_document_word_table_alignment.py", (
        "document-word-table-alignment-cbo-roster-2026-11-13.json", "document-word-table-alignment-cbo-2026-11-13.json",
    ), ("review",), ()),
}
SHORT_EDIT_TRANSFER_2026_11_13 = {
    "image": ("review_image_native_generated_grid_transfer.py", (
        "image-native-generated-grid-measurements-2026-11-13.json",
        "image-native-generated-grid-cc0-roster-2026-11-13.json", "image-native-generated-grid-transfer-2026-11-13.json",
    ), (), ()),
    "video": ("review_video_mouth_photometric_transfer.py", ("video-mouth-photometric-transfer-2026-11-13.json",), (), ()),
    "audio": ("review_audio_phoneme_short_edit_score.py", (
        "audio-phoneme-short-edit-score-2026-11-13.json", "audio-phoneme-short-edit-intake-2026-11-13.json",
    ), (), ()),
    "writing": ("review_writing_forum_revision_transfer.py", ("writing-forum-revision-transfer-2026-11-13.json",), (), (
        f"{SCRIPTS}/writing_uswds_revision_transfer.py",
    )),
}
DELIVERY_AND_LOCALIZATION = {
    "image": ("review_image_native_rotated_stencil_transfer.py", (
        "image-native-rotated-stencil-measurements-2026-11-13.json",
        "image-native-rotated-stencil-cc0-roster-2026-11-13.json", "image-native-rotated-stencil-transfer-2026-11-13.json",
    ), (), ()),
    "video": ("review_video_trimmed_reencode_transfer.py", ("video-trimmed-reencode-transfer-2026-11-13.json",), (), ()),
    "audio": ("review_audio_paired_speech_continuity_transfer.py", (
        "audio-paired-speech-continuity-transfer-2026-11-13.json",
    ), (), ()),
    "writing": ("writing_interval_transfer.py", ("writing-interval-transfer-2026-11-13.json",), ("review",), (
        f"{SCRIPTS}/writing_revision_trajectory_transfer.py", f"{SCRIPTS}/writing_revision_worker_priority.py",
        f"{SCRIPTS}/writing_revision_worker_transfer.py", f"{SCRIPTS}/review_writing_revision_worker_transfer.py",
        f"{SCRIPTS}/video_short_return_encoded_app.py",
        f"{PANELS}/writing-revision-trajectory-2026-11-13.json", f"{PANELS}/writing-revision-worker-complete-2026-11-13.json",
    )),
}
REFERENCE_AND_ROTATION = {
    "image": ("review_image_native_rotation_location.py", (
        "image-native-rotation-location-2026-11-13.json", "image-native-stencil-interpolation-cc0-roster-2026-11-13.json",
    ), (), ()),
    "video": ("review_video_scene_return_webm_transfer.py", (
        "video-scene-return-webm-transfer-2026-11-13.json", "video-scene-return-transfer-2026-11-13.json",
    ), (), ()),
    "audio": ("review_audio_clean_reference_transfer.py", ("audio-clean-reference-transfer-2026-11-13.json",), (), (
        f"{SCRIPTS}/review_audio_paired_speech_continuity_transfer.py",
    )),
    "writing": ("review_writing_gao_archive_transfer.py", ("writing-gao-archive-transfer-2026-11-13.json",), (), ()),
}
SINGLE_UPLOAD_TRANSFERS = {
    "image": ("review_image_bayer_prediction_transfer.py", ("image-bayer-prediction-transfer-2026-11-13.json",), (), (
        f"{PANELS}/image-native-generated-grid-cc0-roster-2026-11-13.json",
        f"{PANELS}/image-native-generated-grid-measurements-2026-11-13.json",
        f"{PANELS}/image-native-stencil-interpolation-cc0-roster-2026-11-13.json",
        f"{PANELS}/image-native-stencil-interpolation-measurements-2026-11-13.json",
    )),
    "video": ("review_video_brief_face_120_carrier.py", ("video-brief-face-120-carrier-2026-11-13.json",), (), ()),
    "audio": ("review_audio_overlap_pitch_transfer.py", ("audio-overlap-pitch-transfer-2026-11-13.json",), (), ()),
    "writing": ("writing_typography_transfer.py", ("writing-typography-transfer-2026-11-13.json",), ("review",), ()),
}
PROCESSING_AND_SHORT_EDITS = {
    "image": ("review_image_quantized_phase_transfer.py", (
        "image-quantized-phase-transfer-2026-11-13.json", "image-quantized-phase-serving-2026-11-13.json",
    ), (), tuple(f"{PANELS}/image-native-{study}-{source}-2026-11-13.json"
                 for study in ("generated-grid", "rotated-stencil", "stencil-interpolation")
                 for source in ("cc0-roster", "measurements"))),
    "video": ("review_video_reference_chroma_transfer.py", ("video-reference-chroma-transfer-2026-11-13.json",), (), ()),
    "audio": ("review_audio_local_context_transfer.py", ("audio-local-context-transfer-2026-11-13.json",), (), ()),
    "writing": ("writing_temperature_transfer.py", ("writing-temperature-transfer-2026-11-13.json",), ("review",), (
        f"{SCRIPTS}/writing_interval_transfer.py", f"{SCRIPTS}/writing_revision_trajectory_transfer.py",
    )),
}
DELIVERY_AND_REFERENCE_TRANSFERS = {
    "image": ("review_image_regional_jpeg90_carrier.py", (
        "image-regional-jpeg90-tandem-2026-11-13.json", "image-regional-jpeg90-direct-2026-11-13.json",
    ), (), ()),
    "video": ("review_video_reference_boundary_transfer.py", ("video-reference-measured-interval-confirmation-2026-11-13.json",), (), (
        f"{SCRIPTS}/review_video_reference_chroma_transfer.py", f"{SCRIPTS}/verify_native_video_interval.py",
    )),
    "audio": ("review_audio_clean_reference_duration.py", ("audio-clean-reference-duration-2026-11-13.json",), (), ()),
    "writing": ("review_writing_gao_exact_excerpt.py", (
        "writing-temperature-transfer-2026-11-13.json", "writing-gao-exact-excerpt-2026-11-13.json",
    ), (), tuple(f"{SCRIPTS}/{file}.py" for file in (
        "writing_temperature_transfer", "writing_interval_transfer", "writing_revision_trajectory_transfer",
    ))),
}
SHORT_EDIT_CHECKS = {
    "image": ("review_image_regional_jpeg90_quality_confirmation.py", ("image-regional-jpeg90-quality-confirmation-2026-11-13.json",), (), (
        f"{SCRIPTS}/review_image_regional_jpeg90_quality_agreement.py", f"{SCRIPTS}/review_image_regional_jpeg90_carrier.py",
    )),
    "video": ("review_video_lipsync_repeat_motion.py", ("video-lipsync-repeat-picture-guard-2026-11-13.json",), (), ()),
    "audio": ("review_audio_formant_local_transfer.py", ("audio-formant-local-transfer-2026-11-13.json",), (), ()),
    "writing": ("writing_word_edit_transfer.py", ("writing-word-edit-transfer-2026-11-13.json",), ("review",), tuple(
        f"{SCRIPTS}/{name}.py" for name in (
            "writing_interval_transfer", "writing_revision_trajectory_transfer", "writing_temperature_transfer", "writing_focus_v3",
        )
    )),
}
EDIT_POSITION_TRANSFERS = {
    "image": ("image_jpeg_detail_loss_confirmation.py", ("image-jpeg-detail-loss-2026-11-13.json",), ("review",), (
        f"{SCRIPTS}/image_reference_same_jpeg_confirmation.py", f"{SCRIPTS}/image_reference_same_jpeg_transfer.py",
        f"{SCRIPTS}/image_jpeg90_paired_corroboration.py",
    )),
    "video": ("video_temporal_multiple_inserts.py", ("video-temporal-multiple-inserts-held-2026-11-13.json",), ("review",), (
        f"{PANELS}/video-temporal-multiple-inserts-plan-2026-11-13.json",
        f"{PANELS}/video-temporal-insert-confirmation-plan-2026-11-13.json",
        f"{PANELS}/video-temporal-insert-confirmation-2026-11-13.json",
        f"{PANELS}/video-temporal-insert-native-2026-11-13.json",
        f"{PANELS}/video-temporal-insert-source-intake-2026-11-13.json",
        f"{SCRIPTS}/video_temporal_insert_confirmation.py", f"{SCRIPTS}/video_temporal_insert_native.py",
        f"{SCRIPTS}/review_video_temporal_insert_sources.py", "backend/app/evidence/video_reference_insertion.py",
    )),
    "audio": ("review_audio_random_clock_stock.py", ("audio-random-clock-stock-2026-11-13.json",), (), (
        f"{SCRIPTS}/review_audio_stock_voice_transfer.py",
    )),
    "writing": ("writing_edit_intensity_transfer.py", ("writing-edit-intensity-transfer-2026-11-13.json",), ("review",), (
        f"{SCRIPTS}/writing_revision_trajectory_transfer.py", f"{SCRIPTS}/review_writing_long_position.py",
        f"{SCRIPTS}/writing_interval_transfer.py", f"{SCRIPTS}/writing_temperature_transfer.py",
    )),
}
SUITES = ("overview", "localization", "transfer", "source-evidence", "recent-checks", "evidence-followups", "short-edits",
          "difficult-edits", "localization-limits", "specialist-transfer", "local-evidence-2026-11-13",
          "evidence-transfer-2026-11-13", "evidence-extent-2026-11-13", "local-evidence-expansion",
          "short-edit-transfer-2026-11-13", "delivery-and-localization", "reference-and-rotation", "single-upload-transfers",
          "processing-and-short-edits", "delivery-and-reference-transfers", "short-edit-checks", "edit-position-transfers")
ISOLATED = (
    "import runpy,sys; "
    "script=sys.argv.pop(1); "
    "sys.argv[0]=script; "
    "sys.path[:0]=['backend','backend/scripts']; "
    "runpy.run_path(script,run_name='__main__')"
)


def hash_file(relative: str) -> str | None:
    path = ROOT / relative
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def revision() -> str | None:
    if not (ROOT / ".git").exists():
        return None
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                   check=True, timeout=3, text=True)
    except (OSError, subprocess.SubprocessError):
        return None
    head = completed.stdout.strip()
    return head if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", head) else None


def replay_inputs(modality: str, suite: str) -> tuple[str, list[str], tuple[str, ...]]:
    if suite == "localization":
        script, panel = LOCALIZATION[modality]
        panel_names, dependencies = (panel,), ()
        prefix = ()
    elif suite == "transfer":
        script, panel, prefix, dependencies = TRANSFERS[modality]
        panel_names = (panel,)
    elif suite == "recent-checks":
        script, panel, prefix, dependencies = RECENT_CHECKS[modality]
        panel_names = (panel,)
    elif suite == "evidence-followups":
        script, panel, prefix, dependencies = EVIDENCE_FOLLOWUPS[modality]
        panel_names = (panel,)
    elif suite == "short-edits":
        script, panel, prefix, dependencies = SHORT_EDITS[modality]
        panel_names = (panel,)
    elif suite == "difficult-edits":
        script, panel, prefix, dependencies = DIFFICULT_EDITS[modality]
        panel_names = (panel,)
    elif suite == "localization-limits":
        script, panel, prefix, dependencies = LOCALIZATION_LIMITS[modality]
        panel_names = (panel,)
    elif suite == "specialist-transfer":
        script, panel_names, dependencies = SPECIALIST_TRANSFER[modality]
        prefix = ()
    elif suite == "local-evidence-2026-11-13":
        script, panel_names, prefix, dependencies = LOCAL_EVIDENCE_2026_11_13[modality]
    elif suite == "evidence-transfer-2026-11-13":
        script, panel_names, prefix, dependencies = EVIDENCE_TRANSFER_2026_11_13[modality]
    elif suite == "evidence-extent-2026-11-13":
        script, panel_names, prefix, dependencies = EVIDENCE_EXTENT_2026_11_13[modality]
    elif suite == "local-evidence-expansion":
        script, panel_names, prefix, dependencies = LOCAL_EVIDENCE_EXPANSION[modality]
    elif suite == "short-edit-transfer-2026-11-13":
        script, panel_names, prefix, dependencies = SHORT_EDIT_TRANSFER_2026_11_13[modality]
    elif suite == "delivery-and-localization":
        script, panel_names, prefix, dependencies = DELIVERY_AND_LOCALIZATION[modality]
    elif suite == "reference-and-rotation":
        script, panel_names, prefix, dependencies = REFERENCE_AND_ROTATION[modality]
    elif suite == "single-upload-transfers":
        script, panel_names, prefix, dependencies = SINGLE_UPLOAD_TRANSFERS[modality]
    elif suite == "processing-and-short-edits":
        script, panel_names, prefix, dependencies = PROCESSING_AND_SHORT_EDITS[modality]
    elif suite == "delivery-and-reference-transfers":
        script, panel_names, prefix, dependencies = DELIVERY_AND_REFERENCE_TRANSFERS[modality]
    elif suite == "short-edit-checks":
        script, panel_names, prefix, dependencies = SHORT_EDIT_CHECKS[modality]
    elif suite == "edit-position-transfers":
        script, panel_names, prefix, dependencies = EDIT_POSITION_TRANSFERS[modality]
    elif suite == "source-evidence":
        script, panel_names, prefix, dependencies = SOURCE_EVIDENCE[modality]
    else:
        script, panel_names, dependencies = CHECKS[modality]
        prefix = ("review",) if modality == "audio" else ()
    entry = f"{SCRIPTS}/{script}"
    panel_arguments = [name if name.startswith("--") else f"{PANELS}/{name}" for name in panel_names]
    inputs = tuple(name for name in panel_arguments if not name.startswith("--"))
    return entry, [*prefix, *panel_arguments], (entry, *inputs, *dependencies)


def replay(modality: str, suite: str = "overview") -> dict:
    entry, arguments, inputs = replay_inputs(modality, suite)
    fingerprints = {path: hash_file(path) for path in inputs}
    record = {"files": fingerprints, "entry": entry, "arguments": arguments}
    if None in fingerprints.values():
        return record | {"status": "missing_input"}
    try:
        completed = subprocess.run([sys.executable, "-B", "-I", "-S", "-c", ISOLATED, entry, *arguments],
                                   cwd=ROOT, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        return record | {"status": "timed_out"}
    if fingerprints != {path: hash_file(path) for path in fingerprints}:
        return record | {"status": "inputs_changed"}
    return record | {"status": "matched" if completed.returncode == 0 else "rejected",
                     "exit_code": completed.returncode, "output_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                     "diagnostic_sha256": hashlib.sha256(completed.stderr).hexdigest()}


def load_receipt(path: Path) -> dict:
    with path.open("rb") as source:
        data = source.read(32_769)
    if len(data) > 32_768:
        raise ValueError("Replay receipt exceeds 32 KiB")
    receipt = json.loads(data)
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA or receipt.get("scope") != SCOPE:
        raise ValueError("Unknown replay receipt format")
    suite = receipt.get("suite", "overview")
    checks = receipt.get("checks")
    if (suite not in SUITES or not isinstance(checks, dict) or not checks
            or not checks.keys() <= CHECKS.keys() or not is_sha256(receipt.get("runner_sha256"))):
        raise ValueError("Invalid replay suite, checks or runner commitment")
    for modality, check in checks.items():
        entry, arguments, inputs = replay_inputs(modality, suite)
        if not isinstance(check, dict) or check.get("entry") != entry or check.get("arguments") != arguments:
            raise ValueError(f"Invalid {modality} replay command")
        files = check.get("files")
        if (not isinstance(files, dict) or files.keys() != set(inputs)
                or any(value is not None and not is_sha256(value) for value in files.values())):
            raise ValueError(f"Invalid {modality} input commitments")
        if check.get("status") == "matched":
            if (None in files.values() or type(check.get("exit_code")) is not int or check["exit_code"] != 0
                    or not is_sha256(check.get("output_sha256")) or not is_sha256(check.get("diagnostic_sha256"))):
                raise ValueError(f"Invalid successful {modality} result")
        elif check.get("status") not in ("missing_input", "timed_out", "inputs_changed", "rejected"):
            raise ValueError(f"Invalid {modality} result status")
    return receipt


def is_sha256(value) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def compare(left: dict, right: dict) -> bool:
    if left.get("suite", "overview") != right.get("suite", "overview"):
        print("Replay suites differ.")
        return False
    if left["runner_sha256"] != right["runner_sha256"]:
        print("Runner versions differ. Checking the recorded reader inputs and outputs.")
    first, second = left["checks"], right["checks"]
    all_matched = True
    for modality in sorted(first.keys() | second.keys()):
        a, b = first.get(modality), second.get(modality)
        if a is None or b is None:
            outcome = "missing from one receipt"
        elif a["status"] != "matched" or b["status"] != "matched":
            outcome = "one or both replays failed"
        elif a["files"] != b["files"]:
            outcome = "input commitments differ"
        elif a["output_sha256"] != b["output_sha256"]:
            outcome = "output commitments differ"
        elif a["diagnostic_sha256"] != b["diagnostic_sha256"]:
            outcome = "output matches, diagnostics differ"
        else:
            outcome = "matched"
        print(f"{modality}: {outcome}")
        all_matched &= outcome == "matched"
    return all_matched


def issue_excerpt(receipt: dict) -> str:
    suite = receipt.get("suite", "overview")
    revision = receipt.get("source_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision):
        revision = "unrecorded"
    environment = receipt.get("environment")
    if not isinstance(environment, dict):
        environment = {}
    values = [environment.get(key) for key in ("python", "os", "architecture")]
    python, system, architecture = [value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,32}", value)
                                    else "unrecorded" for value in values]
    lines = ["### Public evidence replay", "", f"Suite `{suite}` at revision `{revision}`.",
             f"Python `{python}` on `{system}` / `{architecture}`.",
             f"Runner SHA-256 `{receipt['runner_sha256']}`.", "",
             "| Media | Status | Reader output SHA-256 |", "| --- | --- | --- |"]
    for name, check in sorted(receipt["checks"].items()):
        output = f"`{check['output_sha256']}`" if check["status"] == "matched" else "Unavailable"
        lines.append(f"| {name.capitalize()} | `{check['status']}` | {output} |")
    return "\n".join(lines + ["", "Attach or paste the JSON receipt to preserve the exact input and diagnostic hashes."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, help="New JSON receipt path, or the first receipt in a comparison")
    parser.add_argument("--compare", type=Path, help="Compare two saved receipts without running their commands")
    parser.add_argument("--format-issue", action="store_true", help="Print a validated replay receipt as a GitHub issue excerpt")
    parser.add_argument("--modality", action="append", choices=CHECKS, help="Select a modality (default: all four)")
    parser.add_argument("--suite", choices=SUITES, default="overview",
                        help="Select the published study set (default: overview)")
    args = parser.parse_args()
    if args.format_issue:
        if args.compare is not None or args.modality or args.suite != "overview":
            parser.error("Issue formatting uses the suite and modalities recorded in the receipt")
        try:
            receipt = load_receipt(args.receipt)
            print(issue_excerpt(receipt))
        except (OSError, ValueError, RecursionError):
            parser.error("Could not read a valid public replay receipt")
        return
    if args.compare is not None:
        if args.modality or args.suite != "overview":
            parser.error("A comparison uses the suites and modalities recorded in the receipts")
        try:
            matched = compare(load_receipt(args.receipt), load_receipt(args.compare))
        except (OSError, ValueError, RecursionError):
            parser.error("Could not read valid public replay receipts")
        raise SystemExit(0 if matched else 1)
    if args.receipt.exists():
        parser.error("Choose a new receipt path")
    modalities = tuple(dict.fromkeys(args.modality or CHECKS))
    checks = {name: replay(name, args.suite) for name in modalities}
    receipt = {
        "schema": SCHEMA,
        "scope": SCOPE,
        "source_revision": revision(),
        "environment": {"python": platform.python_version(), "os": platform.system(), "architecture": platform.machine()},
        "runner_sha256": hash_file(f"{SCRIPTS}/replay_public_evaluations.py"),
        "checks": checks,
    }
    if args.suite != "overview":
        receipt["suite"] = args.suite
    with args.receipt.open("x", encoding="utf-8") as output:
        json.dump(receipt, output, indent=2, sort_keys=True)
        output.write("\n")
    for name, result in checks.items():
        print(f"{name}: {result['status']}")
    if any(result["status"] != "matched" for result in checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
