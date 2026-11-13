"""Application settings. All tunables live here; nothing is hard-coded in detectors.

Environment variables override defaults (prefix VERITAS_). A .env file in the backend
directory is honoured for local development.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.version import VERSION

BACKEND_DIR = Path(__file__).resolve().parents[2]


MIN_KEY_CHARS = 32  # of the bootstrap key and of the signing key in prod (D-59)
GENERATE_KEY = "generate one with `python -c 'import secrets; print(secrets.token_urlsafe(32))'`"
# Values that documents, samples and habit suggest. Compared without case, separators and repetition
_PLACEHOLDER_WORDS = ("changeme", "replaceme", "placeholder", "yourapikey", "yourkeyhere", "apikey", "secret", "password",
                      "example", "testkey", "veritas", "todo", "xxx")


def is_placeholder(key: str) -> bool:
    """A key made of one repeated character, of a counting run, or of placeholder words only (D-59)."""
    plain = "".join(ch for ch in key.lower() if ch.isalnum())
    if len(set(plain)) <= 2:
        return True
    if plain in "0123456789" * 8 or plain in "abcdefghijklmnopqrstuvwxyz" * 4:
        return True
    rest = plain
    for word in _PLACEHOLDER_WORDS:
        rest = rest.replace(word, "")
    return not rest.strip("0123456789")


class Settings(BaseSettings):
    # hide_input_in_errors: a refused setting is named, its value (a key, a database password) is never printed
    model_config = SettingsConfigDict(env_prefix="VERITAS_", env_file=BACKEND_DIR / ".env", extra="ignore", hide_input_in_errors=True)

    # --- service ---
    app_name: str = "Chipmunk Verify"  # the service's name, as the API's description and the v1 report builder print it
    # The version the probes, the metrics, manifests, reports, packages, webhooks and backups state (D-58). Not a field:
    # VERITAS_VERSION and a `.env` line are not read, and the configuration hash of a result does not hold it
    version: ClassVar[str] = VERSION
    build_commit: str | None = None  # VERITAS_BUILD_COMMIT baked at image build (ARCHITECTURE §6.6); default: the checkout's HEAD
    environment: Literal["dev", "prod"] = "dev"  # prod: an API key is required, VERITAS_OFFLINE defaults to true, the API's documentation pages are off and logs are JSON (the image sets prod)
    log_level: str = "INFO"  # level of the log lines of the API, the worker and the command line (DEBUG, INFO, WARNING, ERROR)
    frontend_dir: Path | None = None  # optional built SPA; default: repository frontend/dist when present
    api_key: str | None = Field(default=None, description="If set, clients must send X-API-Key")
    allow_anonymous: bool = False  # prod refuses to start without an API key unless this is explicitly true
    # The client address of the limiters and of the sign-in events. False: the socket peer (uvicorn runs with
    # --no-proxy-headers, so no header changes it). True: taken from X-Forwarded-For, which only your own proxy may set
    trust_proxy: bool = False
    # How many proxies of yours append to X-Forwarded-For before the API: 1 for the bundled nginx alone, 2 with a TLS proxy or
    # a tunnel in front of it. The client is that many entries from the right; entries further left are the client's own words
    trusted_proxy_hops: int = Field(default=1, ge=1, le=8)
    download_token_ttl_s: int = 600  # signed, short-lived tokens for browser downloads (no API key in URLs); capped at 600
    # HMAC key for resource tokens and review signatures (§5.3). Default in prod: the generated data/keys/signing.key, never a
    # derivation of the API key (D-59). Default in dev: HKDF from api_key, else data/keys/signing.key
    signing_key: str | None = None
    # --- principals and sessions (C-1, §5.3) ---
    single_user: bool | None = None  # unauthenticated loopback requests act as local-examiner (D-09); default: true without api_key
    session_ttl_s: int = 12 * 3600  # browser session lifetime
    login_rate_per_minute: int = 10  # sign-in attempts per client
    credential_failures_per_minute: int = 20  # wrong API keys and dead sessions per client before 429 (SEC-07)
    # Hosted Checkout: explicit opt-in, never falls back to free submissions when enabled but unconfigured.
    billing_mode: Literal["disabled", "test", "live"] = "disabled"
    stripe_secret_key: SecretStr | None = None
    stripe_webhook_secret: SecretStr | None = None
    billing_app_origin: str = "http://127.0.0.1:5230"
    api_key_pepper: str | None = None  # HMAC pepper for principal API keys; default: data/keys/api_key_pepper.key (0600)
    job_timeout_s: int = 900  # per-analysis wall-clock budget (cooperative deadline + watchdog)
    max_requeue_attempts: int = 3  # poison-input guard on crash recovery
    max_upload_concurrency: int = 4  # uploads taken in at once; a further one waits up to 30 s for a slot, then gets 503
    # origins a browser may call the API from (JSON list); the Compose file names the bundled web server's
    cors_origins: list[str] = ["http://localhost:5230", "http://127.0.0.1:5230", "http://localhost:8080", "http://localhost:8160"]
    rate_limit_per_minute: int = 300  # the UI polls running analyses every 1.5-2 s
    # v1 routes are still served (2.1.1 included; API-03) with `Deprecation: true` and this `Sunset` HTTP-date (RFC 8594; ARCHITECTURE §5.1)
    v1_sunset: str = "Fri, 31 Dec 2027 23:59:59 GMT"
    # --- batches and webhooks (X-1; API-01, API-02) ---
    batch_max_items: int = 1000  # files per batch: multipart files, archive members or files below a local_path
    batch_max_bytes: int = 20 * 1024 ** 3  # bytes per batch (staged uploads, the archive, and its declared uncompressed size)
    batch_zip_max_ratio: int = 200  # a member above 64 KB expanding more than this many times its compressed size is refused
    batch_local_roots: list[Path] = []  # VERITAS_BATCH_LOCAL_ROOTS (JSON list): directories a local_path batch may read
    webhooks_enabled: bool = False  # VERITAS_WEBHOOKS_ENABLED: the only outbound network path (PL-01); off by default
    # Deliver to loopback, private, link-local or otherwise non-global addresses (an on-premises receiver). Off by default: a webhook
    # URL must not become a probe of the internal network (Q-2 W-1).
    webhook_allow_private_networks: bool = False
    webhook_timeout_s: float = 10.0  # seconds one delivery attempt may take
    webhook_retry_base_s: float = 30.0  # attempt n+1 after base × 2^(n-1) seconds
    webhook_retry_window_s: float = 24 * 3600.0  # no attempt later than this after the first

    # --- storage ---
    data_dir: Path = BACKEND_DIR / "data"  # the data directory: the SQLite database, originals and artefacts, reports, keys and temporary copies
    min_free_bytes: int = Field(default=1024 ** 3, ge=0)  # N-13: below this much free space on the data volume intake answers 507, readiness is 503 and the worker claims nothing; 0 = no check
    database_url: str | None = None  # default: sqlite in data_dir
    # Postgres only (D-72): the connection pool of each process, and the name its sessions carry in pg_stat_activity (give the
    # API and a standalone worker different names to tell them apart). SQLite ignores the three.
    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=10, ge=0)  # Postgres only: connections a process may open beyond its pool
    db_application_name: str = Field(default="veritas", min_length=1, max_length=63)  # Postgres only: the name of the process's sessions in pg_stat_activity
    # D-55: an unsigned quick analysis (implicit case, or no case) whose newest revision is older than this is purged by
    # `python -m app.cli retention purge`. A named case is purged through its own retention date only; held, retained, signed
    # and in-progress records are never purged by age.
    retention_days: int = 30
    tmp_sweep_min_age_s: int = 6 * 3600  # D-48: data/tmp copies of finished analyses older than this are swept when the worker starts

    # --- compute ---
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"  # where models run; auto: CUDA when present, else MPS, else CPU
    worker_threads: int = 1  # GPU-bound; keep 1 unless multiple accelerators
    run_worker: bool = True  # false on API-only containers; run `python -m app.engine.worker` instead (ARCHITECTURE §3.4)
    job_lease_s: float = 60.0  # a leased job whose worker stops heartbeating is re-queued after this
    job_heartbeat_s: float = 10.0  # also how quickly a running job notices a cancellation request
    job_poll_s: float = 0.5  # idle worker poll interval (submit() wakes in-process workers immediately)
    worker_heartbeat_s: float = Field(default=10.0, gt=0.0)  # N-13: a worker writes its row in `worker_heartbeats` this often
    worker_stale_s: float = Field(default=45.0, gt=0.0)  # N-13: a worker whose row is older than this is counted as gone
    pipeline: Literal["v1", "v2"] = "v2"  # images (P-5), documents and video (P-6) run on the staged DAG; VERITAS_PIPELINE=v1 keeps v0.3
    detector_set: Literal["manifest", "fake"] = "manifest"  # VERITAS_DETECTOR_SET=fake: model-free fakes (tests, e2e-fake)
    fake_delay_s: float = Field(default=0.0, ge=0.0, le=4.0)  # each standard fake sleeps this long first (e2e-fake watches a running analysis)
    sse_poll_s: float = 0.5  # SSE polls analysis_events at this interval (works across processes and SQLite)
    sse_heartbeat_s: float = 15.0  # comment line so proxies keep the stream open
    sse_max_s: float = 3600.0  # a stream closes after this; clients reconnect with Last-Event-ID
    sse_max_per_principal: int = 8  # open event streams per principal (Q-2 S-1); a further stream gets 429
    sse_max_connections: int = 200  # open event streams in this process; a further stream gets 503
    max_json_bytes: int = 2 * 1024 * 1024  # request body cap for every route that takes no upload (Q-2 L-2), counted while streaming
    hf_home: Path | None = None  # HuggingFace cache; default ~/.cache/huggingface
    models_dir: Path | None = None  # VERITAS_MODELS_DIR: non-HF and local weights under <dir>/weights (images: /models); default data_dir

    # --- detector engine (v2) ---
    models_manifest: Path | None = None  # default: app/models.yaml
    # Commercial deployments have no default exception for weights with unresolved permissions.
    # Explicit research policies remain available for separately authorised evaluation.
    licence_policy: str = "commercial"
    default_profile: Literal["auto", "quick", "full", "deep"] = "auto"  # auto: full on accelerators, quick on CPU
    model_memory_budget_mb: str = "auto"  # auto: MPS min(0.5·RAM, RAM−20 GB), CUDA 0.85·VRAM, CPU 0.5·RAM (D-21)
    # VERITAS_OFFLINE: no network fetches; HF hub and transformers read the local cache only. Unset: true in prod, false in dev
    # (PL-01, ARCHITECTURE §8.5); `models fetch` and a models-init container set VERITAS_OFFLINE=false explicitly.
    offline: bool | None = None
    audio_waveform_repeat_review: bool = False  # development-only repeat evidence, with no voice confidence weight

    @model_validator(mode="after")
    def _apply_offline_env(self):
        """Offline mode must reach huggingface_hub / transformers before they are imported (ARCHITECTURE §8.5)."""
        if self.offline is None:
            self.offline = self.environment == "prod"
        if self.offline:
            import os

            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return self

    # --- reports (ARCHITECTURE §6.1) ---
    report_engine: Literal["auto", "weasyprint", "reportlab"] = "auto"  # auto: WeasyPrint when Pango loads; reportlab: v1 PDF only
    # macOS: dyld does not search Homebrew's prefix for bare library names, so the Pango libraries are retried here
    report_library_dirs: list[Path] = [Path("/opt/homebrew/lib"), Path("/usr/local/lib")]

    # --- upload limits ---
    max_image_bytes: int = 60 * 1024 * 1024  # D-48: full-resolution camera originals (the v2.1 field test refused a 26 MB one)
    max_video_bytes: int = 500 * 1024 * 1024  # largest video upload
    max_audio_bytes: int = 150 * 1024 * 1024  # A-1: 10 min of 48 kHz stereo 16-bit WAV is ~110 MB (the pipeline analyses ≤ 600 s)
    max_document_bytes: int = 25 * 1024 * 1024  # largest document upload
    max_document_chars: int = 120_000  # ~20k words; bounded with the 1 500 scored-unit cap
    document_decode_timeout_s: float = 120.0  # hard limit of reading and segmenting one document, in child processes that are killed (SEC-04, S2-01)

    # --- image detector ---
    image_classifiers: list[str] = [  # HuggingFace image-classification checkpoints, each "repo_id|positive_label_substring|weight"
        # HuggingFace image-classification checkpoints: "repo_id|positive_label_substring|weight"
        # weights set from docs/eval_image.md (public-benchmark AUC / FPR); see docs/evaluation.md
        "haywoodsloan/ai-image-detector-deploy|artificial|1.0",
        "Ateeqq/ai-vs-human-image-detector|ai|0.7",  # AUC 0.94 on benchmarks; over-fires on phone computational photography and video frames (field test)
        # Organika/sdxl-detector removed in v2 (detector survey: PR-AUC at chance on marketplace photos)
        "dima806/deepfake_vs_real_image_detection|fake|1.0",  # face-oriented (GAN portraits); AUC 1.00 whole-image
        "Wvolf/ViT_Deepfake_Detection|fake|0.8",  # face-oriented (GAN portraits); AUC 1.00 whole-image
        # "prithivMLmods/Deep-Fake-Detector-v2-Model|deepfake|1.0",  # disabled: inverted on public benchmarks (AUC 0.06-0.12)
    ]
    univfd_enabled: bool = True  # use the UnivFD linear probe (CLIP features) in the image scoring
    univfd_weight: float = 1.1  # strong on GAN/SD; weaker on FLUX/MJ v6 (field test)
    univfd_fc_url: str = (  # where the UnivFD linear-probe weights are fetched from (pinned in weights_sha256)
        "https://github.com/WisconsinAIVision/UniversalFakeDetect/raw/main/pretrained_weights/fc_weights.pth"
    )
    yunet_url: str = (  # where the YuNet face detector (ONNX) is fetched from (pinned in weights_sha256)
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx"
    )
    dfdc_enabled: bool = True  # use the DFDC face-swap classifier on detected faces
    dfdc_weight: float = 2.0  # its weight in the v0.3 image scoring
    # one or more of the author's five released checkpoints; predictions are averaged
    dfdc_urls: list[str] = [
        "https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19",
        "https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23",
    ]
    face_min_size: int = 48  # a detected face whose box is smaller than this many pixels is left out
    gradcam_enabled: bool = True  # v0.3 image pipeline: draw Grad-CAM heat maps for the ViT classifiers

    # --- metadata (I-3) ---
    exiftool_path: str = "exiftool"  # ExifTool ≥ 12.24 on PATH or an absolute path (D-07); Pillow fallback without it

    # --- video detector ---
    video_max_frames: int = 40  # v0.3 video pipeline: most frames sampled from a video (4 per second up to this)
    # per-model weight multipliers applied only when scoring video frames (compression changes behaviour)
    video_weight_overrides: dict[str, float] = {
        "ai-vs-human-image-detector": 0.3,  # fires on compressed frames
        "deepfake_vs_real_image_detection": 0.25,  # GAN-portrait detectors are not swap detectors
        "ViT_Deepfake_Detection": 0.25,
    }
    # face classifiers lose reliability when the face crop must be upscaled to the model input
    face_upscaled_weight: float = 0.4
    dfdc_min_face_px: int = 120  # below this the DFDC crop is heavily upscaled; weight × 0.4
    video_min_frames: int = 16  # v0.3 video pipeline: fewest frames sampled from a video
    video_face_track_iou: float = 0.3  # overlap (IoU) from which face boxes of consecutive frames are one track

    # Optional native PNG region evidence; old checks stay available for comparison/rollback.
    image_region_mode: Literal["legacy", "candidate"] = "legacy"
    image_region_flip_review: bool = False  # Development-only repeatability filter for sensitive PNG regions.
    image_sharp_boundary_enabled: bool = False
    image_reference_comparison_enabled: bool = False  # Development-only comparison of an uploaded earlier copy.
    image_reference_jpeg_png_envelope: bool = False  # Optional earlier standard JPEG to cropped PNG color differences.
    audio_reference_comparison_enabled: bool = False  # Authenticated comparison of a short earlier recording.
    video_reference_comparison_enabled: bool = False  # Authenticated development-only comparison of a short earlier clip.
    video_reference_timing_model_path: Path | None = None  # Optional pinned local landmark task for a marked earlier closure.
    document_revision_comparison_enabled: bool = False  # Authenticated development-only earlier-draft comparison.
    document_archive_comparison_enabled: bool = False  # Authenticated development-only public archive comparison.
    image_affine_copy_review: bool = False  # Development-only dense masks for same-image geometric copies.
    image_jpeg_block_copy_review: bool = False  # Development-only exact JPEG brightness-block fallback.
    image_jpeg_offgrid_copy_review: bool = False  # Development-only texture and perimeter matching for selected JPEGs.
    image_embedded_pair_review: bool = False  # Development-only exact differences between two matching PNG areas.
    image_png_exif_preview_review: bool = False  # Optional native PNG comparison with its saved Exif thumbnail.
    image_whole_review_python: Path | None = None  # Optional local Python environment for the supplementary image check.
    image_whole_png_review_enabled: bool = False  # Also review eligible upright RGB PNGs in development.

    # --- audio (video sound track) ---
    audio_enabled: bool = True  # score the sound track of a video with audio_classifiers
    # Local rollout switch; the legacy weights/calibration remain installed for rollback.
    audio_detector_mode: Literal["legacy", "candidate"] = "legacy"
    audio_background_review_model: Path | None = None  # Local pinned enhancement assets.
    audio_background_review_packages: Path | None = None  # Local Apple Silicon dependency overlay.
    video_face_review_mode: Literal["legacy", "sequence", "appearance"] = "legacy"
    video_face_boundary_refinement: bool = False  # Refine timing on an existing short appearance review range.
    video_face_short_return_review: bool = False  # Add a distinct, source-bound range to eligible appearance reviews.
    video_face_second_return_review: bool = False  # Check a second disjoint range on tested short H.264 profiles.
    video_face_detail_repeat_review: bool = False  # Repeat a selected coarse appearance range at lower image detail.
    video_whole_face_review: bool = False  # Add six native face observations for local review without changing the estimate.
    audio_weight: float = 0.8  # docs/eval_audio.md: fused AUC 0.92 on ASVspoof-DF + in-the-wild, FPR ~10 % in the wild
    audio_classifiers: list[str] = [  # HuggingFace audio-classification checkpoints, each "repo_id|positive_label_substring|weight"
        "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification|fake|1.0",  # AUC 0.917 overall / 0.974 ASVspoof-DF
        "mo-thecreator/Deepfake-audio-detection|fake|0.5",  # AUC 0.855; FPR 26 % → half weight
        # "MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection|spoof|1.0",  # disabled: AUC 0.55, 77 % FPR in the wild
    ]

    # --- text detector ---
    text_detector_mode: Literal["legacy", "candidate"] = "legacy"
    text_isolated_paragraph_review: bool = False
    text_paragraph_evidence_priority: bool = False
    text_matched_context_review: bool = False
    text_checkpoint_agreement_review: bool = False
    text_classifier_id: str = "desklib/ai-text-detector-v1.01"  # base checkpoint (HF Hub)
    # fine-tuned on 2025-26 LLM output (scripts/finetune_text_detector.py); used automatically when present
    text_classifier_finetuned_path: Path = BACKEND_DIR / "data" / "weights" / "desklib-ft-v2"
    text_classifier_weight: float = 2.0  # primary; validated on frontier-model prose (docs/eval_text_ood.md)
    text_secondary_classifier_id: str | None = "Hello-SimpleAI/chatgpt-detector-roberta"  # a second text classifier (HF Hub); unset: none
    text_secondary_weight: float = 0.3  # ChatGPT-2022 era; votes 'human' on modern text
    binoculars_enabled: bool = True  # run the Binoculars zero-shot check on text
    binoculars_observer_id: str = "Qwen/Qwen2.5-1.5B"  # its observer model (HF Hub)
    binoculars_performer_id: str = "Qwen/Qwen2.5-1.5B-Instruct"  # its performer model (HF Hub)
    binoculars_weight: float = 0.6  # zero-shot corroboration; weak on frontier-model prose (docs/evaluation.md)
    # Binoculars score: lower => more likely machine. Thresholds are model-pair specific and are
    # set by scripts/calibrate_binoculars.py. Defaults below are for the Qwen2.5-1.5B pair.
    binoculars_threshold: float = 0.9505
    binoculars_scale: float = 26.0  # logistic slope around the threshold (scripts/eval_text.py, HC3 n=300)
    stylometry_weight: float = 0.5  # weight of the stylometry signal in the v0.3 text scoring
    text_window_sentences: int = 3  # context window for per-sentence classifier scoring
    text_min_words_for_verdict: int = 40  # v0.3 text pipeline: a text of fewer than three times this many words gets the short-document note and low confidence

    # --- supply chain: pinned SHA-256 of non-HF weight files (verified on download and on load) ---
    weights_sha256: dict[str, str] = {  # file name -> pinned SHA-256 of each non-HF weight file
        "univfd_fc_weights.pth": "477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847",
        "face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19": "e2f0066a8ff036add2f5b418d2eba1051c2d2788f84168f00a85207f3b666e04",
        "final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23": "849036dc211387420412feb68c7451e98a948072b94e213e1104e2f2bf7791ad",
    }

    # --- fusion ---
    prob_floor: float = 0.02  # v0.3 fusion: the lowest probability it reports
    prob_ceiling: float = 0.98  # v0.3 fusion: the highest probability it reports
    # K-3: the release fuser version per kind, e.g. {"image": "2.1.1"}; unset kinds serve their newest installed artefact. A
    # refused release artefact means the D-46 fallback, never an older artefact (those load only through a sha256 pin)
    fuser_versions: dict[str, str] = {}
    # K-3, D-57: the cloud preflight (`infra/gcp/preflight.py`) refuses, in prod, a fuser that no review sidecar names by its
    # sha256 (`python -m app.cli fusers review`); true accepts it with a printed warning. Local and dev serving never check it
    accept_pending_fuser_review: bool = False
    # D-57: fuser artefacts `fuser-<modality>-<semver>.json` (+ .boot.npy, + .review.json). Set: only this directory is read.
    # Unset: data_dir/fusion and the release directory packaged with the source, listed together (the newest version serves;
    # within one version the packaged file)
    fuser_dir: Path | None = None
    prior_ai: float | None = Field(default=None, gt=0, lt=1)  # VERITAS_PRIOR_AI: deployment base rate, adds logit(π) (ML §4.4). It also decides which results are decisive (D-77 reads the published estimate): a prior under one half takes AI headlines away, one over it clean headlines (at 0.1 an AI code needs an estimate of 94.2 or more on the scale of the levels, at 0.9 a clean code one of 5.5 or less); under a self-declared marker the confidence is then low wherever the rule took the statistical code; the second condition of D-76 does not depend on it. An inconclusive result can then stand with its estimate in the part of the scale of the other answer (at 0.1 a picture with a face to which the levels give the AI answer at 70 is published at 21): its reason, the note under its scale and what the result means name that part, and it says that the estimate is adjusted for the configured share and the calibrated levels are not (docs/limitations.md)
    h3_influence_bound: float = Field(default=0.05, ge=0, le=0.5)  # |Δs| cap for non-override provenance states (DC-06, D-33)

    @model_validator(mode="after")
    def _prod_requires_auth(self):
        if self.environment != "prod":
            return self
        if not self.api_key and not self.allow_anonymous:
            raise ValueError("VERITAS_API_KEY must be set in prod (or VERITAS_ALLOW_ANONYMOUS=true to run an open instance deliberately)")
        # D-59: a key that can be guessed is no key. The values are never printed
        if self.api_key and (len(self.api_key) < MIN_KEY_CHARS or is_placeholder(self.api_key)):
            raise ValueError(f"VERITAS_API_KEY must be at least {MIN_KEY_CHARS} characters in prod and not a placeholder; {GENERATE_KEY}")
        if self.signing_key is not None:
            if len(self.signing_key) < MIN_KEY_CHARS or is_placeholder(self.signing_key):
                raise ValueError(f"VERITAS_SIGNING_KEY must be at least {MIN_KEY_CHARS} characters in prod and not a placeholder; "
                                 f"{GENERATE_KEY}, or leave it unset to use the generated data/keys/signing.key")
            if self.signing_key == self.api_key:
                raise ValueError("VERITAS_SIGNING_KEY must differ from VERITAS_API_KEY in prod")
        return self

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def weights_dir(self) -> Path:
        return self.models_dir / "weights" if self.models_dir is not None else self.data_dir / "weights"

    @property
    def fusion_dir(self) -> Path:
        return self.fuser_dir or self.data_dir / "fusion"

    @property
    def objects_dir(self) -> Path:
        """Content-addressed artefact store `objects/sha256/aa/bb/<sha>` (ARCHITECTURE §4.7)."""
        return self.data_dir / "objects"

    @property
    def sqlalchemy_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'veritas.db').as_posix()}"

    def ensure_dirs(self) -> None:
        make_data_dir(self.data_dir)
        for d in (self.uploads_dir, self.artifacts_dir, self.reports_dir, self.weights_dir, self.objects_dir):
            d.mkdir(parents=True, exist_ok=True)
        ensure_tmpdir()


DATA_DIR_MODE = 0o700


def make_data_dir(path: Path) -> None:
    """Make the data directory when it is missing, for the service's user alone (N-14b; the part of SEC-17 that needs
    code). The database and every evidence file lie under it, and with the default mode (0755, the database 0644) the
    other users of a shared host can read both. A directory that exists is left as it is: the runbook says `chmod 700`
    for one that an earlier version made."""
    path.mkdir(mode=DATA_DIR_MODE, parents=True, exist_ok=True)


def ensure_tmpdir() -> Path | None:
    """Make the directory `TMPDIR` names (Compose: `/data/tmp`, on the data volume; BUILD_SPEC N-14, OPS-09) and have
    `tempfile` ask again. Python falls back to `/tmp` without a word when the named directory is missing, and it keeps its
    first answer: uploads and report work files would then spool into the container, not onto the volume. A directory that
    cannot be made is reported once on stderr (the logging of the process may not be set up yet) and the fallback stays."""
    import os
    import sys
    import tempfile

    named = os.environ.get("TMPDIR")
    if not named:
        return None
    path = Path(named)
    if path.is_dir():
        return path  # `tempfile` found it, or will
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"TMPDIR {named} cannot be created ({type(e).__name__}); temporary files go to {tempfile.gettempdir()}", file=sys.stderr)
        return None
    tempfile.tempdir = None
    return path


settings = Settings()


def text_classifier_source() -> str:
    """Local fine-tuned checkpoint if it exists, else the base HF checkpoint."""
    p = settings.text_classifier_finetuned_path
    return str(p) if (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists() else settings.text_classifier_id
