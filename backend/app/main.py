import time
import json
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import config
from .checkpoints import load_dynamic_checkpoint, load_static_checkpoint, load_static_classes
from .features import predict_with_mirror, predict_static
from .session import Session
from . import nlp_bridge

app = FastAPI(title="ASL Real-Time Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {"device": None, "dynamic_model": None, "dynamic_classes": None,
         "static_model": None, "static_classes": None}


@app.on_event("startup")
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    STATE["device"] = device
    print(f"Device: {device}")

    try:
        model, classes = load_dynamic_checkpoint(config.DYNAMIC_CHECKPOINT_PATH, device)
        STATE["dynamic_model"], STATE["dynamic_classes"] = model, classes
    except Exception as e:
        print(f"[WARNING] Could not load dynamic checkpoint: {e}")

    try:
        static_classes = load_static_classes(config.STATIC_LABEL_ENCODER_PATH)
        static_model = load_static_checkpoint(
            config.STATIC_CHECKPOINT_PATH, len(static_classes), device
        )
        STATE["static_model"], STATE["static_classes"] = static_model, static_classes
    except Exception as e:
        print(f"[WARNING] Could not load static checkpoint: {e}")


@app.get("/api/health")
def health():
    return {
        "dynamic_model_loaded": STATE["dynamic_model"] is not None,
        "static_model_loaded": STATE["static_model"] is not None,
        "dynamic_classes": STATE["dynamic_classes"],
        "static_classes": STATE["static_classes"],
        "nlp_available": nlp_bridge.NLP_AVAILABLE,
        "device": str(STATE["device"]),
    }


def top5_payload(probs: np.ndarray, classes: list) -> dict:
    if probs is None or probs.sum() == 0:
        return {"classes": [], "scores": []}
    idx = np.argsort(probs)[::-1][:5]
    return {"classes": [classes[i] for i in idx], "scores": [float(probs[i]) for i in idx]}


def parse_binary_frame(data: bytes):
    """Inverse of the frontend's packFrame(): 1 byte type, 1 byte has_raw63
    flag, 155 float32 (raw155), optionally 63 float32 (raw63)."""
    has63 = data[1] == 1
    raw155 = np.frombuffer(data, dtype="<f4", count=config.RAW_FEAT_DIM, offset=2)
    raw63 = None
    if has63:
        raw63 = np.frombuffer(
            data, dtype="<f4", count=63, offset=2 + config.RAW_FEAT_DIM * 4
        )
    return raw155, raw63


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = Session()
    dynamic_probs_display = np.zeros(len(STATE["dynamic_classes"] or []), dtype=np.float32)
    static_probs_display = np.zeros(len(STATE["static_classes"] or []), dtype=np.float32)

    try:
        while True:
            packet = await websocket.receive()

            # ── binary message: a landmark frame ──────────────────────
            if packet.get("bytes") is not None:
                now = time.time()
                raw155, raw63 = parse_binary_frame(packet["bytes"])

                # dynamic buffer always fills, same as the original script
                feat = np.nan_to_num(np.array(raw155, dtype=np.float32))
                session.frame_buffer.append(feat)
                if len(session.frame_buffer) > config.FIXED_FRAMES:
                    session.frame_buffer.pop(0)

                # STATIC branch
                if session.mode == "STATIC" and STATE["static_model"] is None:
                    if not session.__dict__.get("_warned_no_static_model"):
                        print("[WARNING] Switched to STATIC mode but no static model is "
                              "loaded — check /api/health and STATIC_CHECKPOINT_PATH / "
                              "STATIC_LABEL_ENCODER_PATH. Predictions will stay stuck on '…'.")
                        session._warned_no_static_model = True

                if (session.mode == "STATIC" and STATE["static_model"] is not None
                        and now - session.last_static_pred_time > config.STATIC_PRED_COOLDOWN):
                    session.last_static_pred_time = now
                    if raw63 is not None:
                        feat63 = np.nan_to_num(np.array(raw63, dtype=np.float32))
                        s_probs = predict_static(feat63, STATE["static_model"], STATE["device"])
                        static_probs_display = s_probs.cpu().numpy()

                        s_conf = float(s_probs.max().item())
                        s_idx = int(s_probs.argmax().item())
                        classes = STATE["static_classes"]
                        s_label = classes[s_idx] if s_conf >= config.STATIC_CONFIDENCE_THRESHOLD else "uncertain"

                        session.static_pred_history.append(s_label)
                        import collections as _c
                        counts = _c.Counter(session.static_pred_history)
                        session.current_static_label = counts.most_common(1)[0][0]
                        session.current_static_conf = s_conf

                        n = session.__dict__.get("_static_pred_count", 0)
                        if n < 5:
                            print(f"[static] raw_top1={classes[s_idx]} conf={s_conf:.2f} "
                                  f"-> smoothed={session.current_static_label}")
                            session._static_pred_count = n + 1
                    else:
                        session.current_static_label = "no hand"
                        session.current_static_conf = 0.0

                # DYNAMIC branch
                if (session.mode == "DYNAMIC" and STATE["dynamic_model"] is not None
                        and len(session.frame_buffer) == config.FIXED_FRAMES
                        and now - session.last_dynamic_pred_time > config.PRED_COOLDOWN):
                    session.last_dynamic_pred_time = now

                    seq_155 = np.array(session.frame_buffer, dtype=np.float32)
                    seq_155 = np.nan_to_num(seq_155)

                    probs = predict_with_mirror(seq_155, STATE["dynamic_model"], STATE["device"])
                    dynamic_probs_display = probs.cpu().numpy()

                    conf = float(probs.max().item())
                    idx = int(probs.argmax().item())
                    classes = STATE["dynamic_classes"]
                    label = classes[idx] if conf >= config.CONFIDENCE_THRESHOLD else "uncertain"

                    session.pred_history.append(label)
                    import collections as _c
                    counts = _c.Counter(session.pred_history)
                    session.current_dynamic_label = counts.most_common(1)[0][0]
                    session.current_dynamic_conf = conf

                session.sync_current()

                if session.mode == "STATIC":
                    probs_display, display_classes = static_probs_display, STATE["static_classes"] or []
                else:
                    probs_display, display_classes = dynamic_probs_display, STATE["dynamic_classes"] or []

                await websocket.send_json({
                    "type": "prediction",
                    "state": session.snapshot(),
                    "top5": top5_payload(probs_display, display_classes),
                })
                continue

            # ── text message: a control action ────────────────────────
            if packet.get("text") is not None:
                msg = json.loads(packet["text"])
                msg_type = msg.get("type")

                if msg_type != "control":
                    await websocket.send_json({"type": "error", "reason": f"unknown message type '{msg_type}'"})
                    continue

                action = msg.get("action")
                result = None
                if action == "toggle_mode":
                    result = {"mode": session.toggle_mode()}
                elif action == "add":
                    result = session.add_current_prediction()
                elif action == "word":
                    result = session.finish_word()
                elif action == "undo":
                    result = session.undo()
                elif action == "toggle_llm":
                    result = {"llm_enabled": session.toggle_llm()}
                elif action == "generate":
                    result = session.generate_sentence()
                elif action == "clear":
                    session.clear()
                    result = {"ok": True}
                elif action == "reset":
                    session.reset_buffer()
                    dynamic_probs_display[:] = 0
                    static_probs_display[:] = 0
                    result = {"ok": True}
                else:
                    result = {"ok": False, "reason": f"unknown action '{action}'"}

                session.sync_current()
                await websocket.send_json({
                    "type": "control_result",
                    "action": action,
                    "result": result,
                    "state": session.snapshot(),
                })

    except WebSocketDisconnect:
        pass


# Serve the frontend (mounted last so it doesn't shadow /api or /ws routes).
_frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
