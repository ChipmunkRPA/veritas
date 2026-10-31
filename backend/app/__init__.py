import os

# Set before torch is imported anywhere (ARCHITECTURE §2.5): CPU fallback for missing MPS ops, and fail fast
# instead of thrashing swap on the shared host. Explicit environment values win.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
# PyTorch requires low ≤ high. Its default low ratio (1.4 on unified memory) is invalid once the high ratio is lowered,
# and every MPS allocation then fails with "invalid low watermark ratio 1.4". Derive the low ratio when it is unset.
if "PYTORCH_MPS_LOW_WATERMARK_RATIO" not in os.environ:
    try:
        _high = float(os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"])
    except ValueError:
        _high = 0.0
    if _high > 0:  # 0.0 disables the high limit, and with it the ordering check
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = f"{_high * 0.85:.3f}"
