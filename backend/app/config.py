"""
Central configuration for the ASL inference backend.
Edit the paths below to point at your local checkpoint / model files.
Everything can also be overridden with environment variables of the
same name (e.g. `DYNAMIC_CHECKPOINT_PATH=/models/foo.pt uvicorn ...`).
"""
import os

# ── model checkpoints ─────────────────────────────────────────────
DYNAMIC_CHECKPOINT_PATH = os.environ.get(
    "DYNAMIC_CHECKPOINT_PATH", "./checkpoints/final_dynamic_model.pt"
)
STATIC_CHECKPOINT_PATH = os.environ.get(
    "STATIC_CHECKPOINT_PATH", "./checkpoints/final_static_model.pth"
)
STATIC_LABEL_ENCODER_PATH = os.environ.get(
    "STATIC_LABEL_ENCODER_PATH", "./checkpoints/label_encoder.joblib"
)

# ── feature / model dims (must match training) ────────────────────
FIXED_FRAMES = 60
RAW_FEAT_DIM = 155
MOTION_FEAT_DIM = 308
POSE_IDXS = [0, 11, 12, 13, 14, 15, 16, 23, 24]

# ── thresholds ──────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45
SMOOTHING_WINDOW = 5

STATIC_CLASSES_FALLBACK = [str(d) for d in range(10)] + [
    chr(c) for c in range(ord("A"), ord("Z") + 1)
]
STATIC_CONFIDENCE_THRESHOLD = 0.60
STATIC_SMOOTHING_WINDOW = 8

PRED_COOLDOWN = 0.08          # seconds, dynamic model (lowered from 0.15 for snappier updates)
STATIC_PRED_COOLDOWN = 0.03   # seconds, static model (lowered from 0.10)

NO_COMMIT_LABELS = {"uncertain", "collecting…", "no hand", "…"}

# ── CORS ────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
