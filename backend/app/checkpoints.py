import torch

from . import config
from .model_defs import ASLv3Classifier, ASLStaticModel


def load_dynamic_checkpoint(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    classes = ckpt["classes"]

    feat_dim = cfg.get("feat_dim", config.MOTION_FEAT_DIM)
    if feat_dim != config.MOTION_FEAT_DIM:
        print(
            f"[WARNING] Checkpoint feat_dim={feat_dim} != expected "
            f"{config.MOTION_FEAT_DIM}. Expects a V3 motion-aware "
            f"checkpoint (feat_dim=308)."
        )

    model = ASLv3Classifier(
        feat_dim=feat_dim,
        num_classes=len(classes),
        gru1_hidden=cfg.get("gru1_hidden", 256),
        gru2_hidden=cfg.get("gru2_hidden", 128),
        dropout_gru=cfg.get("dropout_gru", 0.35),
        dropout_cls1=cfg.get("dropout_cls1", 0.40),
        dropout_cls2=cfg.get("dropout_cls2", 0.30),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    arch = ckpt.get("architecture", "ASLv3Classifier")
    print(f"Loaded {arch} | feat_dim={feat_dim} | {len(classes)} classes | device={device}")
    return model, classes


def load_static_classes(encoder_path: str) -> list:
    try:
        import joblib

        encoder = joblib.load(encoder_path)
        classes = list(encoder.classes_)
        print(f"Loaded static label encoder | {len(classes)} classes")
        return classes
    except Exception as e:
        print(
            f"[WARNING] Could not load static label encoder ({e}). "
            f"Falling back to hardcoded 0-9/A-Z class list."
        )
        return config.STATIC_CLASSES_FALLBACK


def load_static_checkpoint(ckpt_path: str, num_classes: int, device):
    model = ASLStaticModel(num_classes=num_classes)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"Loaded ASLStaticModel | {num_classes} classes | device={device}")
    return model
