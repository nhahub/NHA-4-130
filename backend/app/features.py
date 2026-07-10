"""
Feature-space math, ported verbatim from the original camera-inference
script. Landmark EXTRACTION now happens in the browser (MediaPipe Tasks
Vision JS produces the same 21-point hand / 9-point pose landmarks), so
this module starts from an already-extracted RAW 155-dim vector per
frame (or a 63-dim vector for a single static frame) and only handles:

  mirror -> motion-feature expansion -> normalisation -> model forward
"""
import numpy as np
import torch

POS_COLS_M = slice(0, 153)
VEL_COLS_M = slice(153, 306)
LH_FLAG_M = 306
RH_FLAG_M = 307


def add_motion_features(seq: np.ndarray) -> np.ndarray:
    """RAW (T, 155) -> (T, 308) with explicit velocity channels."""
    coords = seq[:, :153]
    flags = seq[:, 153:155]

    velocity = np.zeros_like(coords)
    velocity[1:] = coords[1:] - coords[:-1]

    out = np.concatenate([coords, velocity, flags], axis=1)
    return out.astype(np.float32)


def normalise_coords(seq_308: np.ndarray) -> np.ndarray:
    """Wrist-relative anchoring + independent z-score of pos/vel blocks."""
    out = seq_308.copy()

    lh = out[:, 0:63].reshape(-1, 21, 3)
    lh = lh - lh[:, 0:1, :]
    out[:, 0:63] = lh.reshape(-1, 63)

    rh = out[:, 63:126].reshape(-1, 21, 3)
    rh = rh - rh[:, 0:1, :]
    out[:, 63:126] = rh.reshape(-1, 63)

    pos = out[:, POS_COLS_M]
    pos_mean, pos_std = pos.mean(), pos.std()
    if pos_std > 1e-6:
        out[:, POS_COLS_M] = (pos - pos_mean) / pos_std

    vel = out[:, VEL_COLS_M]
    vel_mean, vel_std = vel.mean(), vel.std()
    if vel_std > 1e-6:
        out[:, VEL_COLS_M] = (vel - vel_mean) / vel_std

    return out


def mirror_sequence(seq_155: np.ndarray) -> np.ndarray:
    """Mirror a RAW (T, 155) sequence — applied BEFORE motion expansion."""
    out = seq_155.copy()
    lh, rh = seq_155[:, :63].copy(), seq_155[:, 63:126].copy()
    lf, rf = seq_155[:, 153].copy(), seq_155[:, 154].copy()

    out[:, :63] = rh
    out[:, 63:126] = lh
    out[:, 153] = rf
    out[:, 154] = lf

    for sl in [slice(0, 63), slice(63, 126)]:
        block = out[:, sl].reshape(seq_155.shape[0], 21, 3)
        block[:, :, 0] *= -1
        out[:, sl] = block.reshape(seq_155.shape[0], -1)

    pose = out[:, 126:153].reshape(seq_155.shape[0], 9, 3)
    pose[:, :, 0] *= -1
    out[:, 126:153] = pose.reshape(seq_155.shape[0], -1)
    return out


def prepare_sequence(raw_155: np.ndarray, mirror: bool) -> np.ndarray:
    seq = raw_155
    if mirror:
        seq = mirror_sequence(seq)
    seq = add_motion_features(seq)
    seq = normalise_coords(seq)
    return seq


@torch.no_grad()
def predict_with_mirror(seq_155_np: np.ndarray, model, device) -> torch.Tensor:
    """Average original + mirrored softmax (hand-agnostic)."""
    orig_308 = prepare_sequence(seq_155_np.copy(), mirror=False)
    mir_308 = prepare_sequence(seq_155_np.copy(), mirror=True)

    orig = torch.from_numpy(orig_308).unsqueeze(0).to(device)
    mir = torch.from_numpy(mir_308).unsqueeze(0).to(device)

    p_orig = torch.softmax(model(orig), dim=-1).squeeze(0)
    p_mir = torch.softmax(model(mir), dim=-1).squeeze(0)
    return (p_orig + p_mir) / 2.0


@torch.no_grad()
def predict_static(feat_63: np.ndarray, model, device) -> torch.Tensor:
    """Run the static CNN on one (63,) landmark vector, return softmax probs."""
    x = torch.from_numpy(feat_63.astype(np.float32)).reshape(1, 1, 63).to(device)
    logits = model(x)
    return torch.softmax(logits, dim=-1).squeeze(0)
