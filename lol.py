import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
import joblib
import time

# ======= MEDIAPIPE TASKS =======
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions

# ================= LOAD =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = joblib.load("D:/DEPI/Final/project/asl_dataset/label_encoder.joblib")

# ================= MODEL =================
class ASLModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1   = nn.Conv1d(1, 64, 3, padding=1)
        self.bn1     = nn.BatchNorm1d(64)
        self.conv2   = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2     = nn.BatchNorm1d(128)
        self.pool    = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1     = nn.LazyLinear(128)
        self.fc2     = nn.Linear(128, len(encoder.classes_))

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

model = ASLModel(num_classes=len(encoder.classes_))
model.load_state_dict(torch.load(
    "D:/DEPI/Final/project/asl_dataset/best_model.pth",
    map_location=device
))
model.to(device)
model.eval()

# ================= MEDIAPIPE =================
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=1
)
detector = HandLandmarker.create_from_options(options)

# ================= GESTURE HELPERS =================

def is_finger_extended(lms, tip, pip):
    """Returns True if a finger is extended (tip above pip in y-axis)."""
    return lms[tip].y < lms[pip].y

def is_thumbs_up(lms):
    """
    Thumb tip clearly above wrist, all other fingers curled.
    Works for either hand shown to camera.
    """
    thumb_up   = lms[4].y < lms[2].y          # thumb tip above thumb base
    wrist_y    = lms[0].y
    thumb_high = lms[4].y < wrist_y - 0.1     # thumb well above wrist

    # other fingers curled: tip below their pip joint
    index_curled  = lms[8].y  > lms[6].y
    middle_curled = lms[12].y > lms[10].y
    ring_curled   = lms[16].y > lms[14].y
    pinky_curled  = lms[20].y > lms[18].y

    return thumb_up and thumb_high and index_curled and middle_curled and ring_curled and pinky_curled

def is_thumbs_down(lms):
    """
    Thumb tip clearly below wrist, all other fingers curled.
    """
    thumb_down = lms[4].y > lms[2].y          # thumb tip below thumb base
    wrist_y    = lms[0].y
    thumb_low  = lms[4].y > wrist_y + 0.1     # thumb well below wrist

    index_curled  = lms[8].y  > lms[6].y
    middle_curled = lms[12].y > lms[10].y
    ring_curled   = lms[16].y > lms[14].y
    pinky_curled  = lms[20].y > lms[18].y

    return thumb_down and thumb_low and index_curled and middle_curled and ring_curled and pinky_curled

# ================= HOLD-TO-CONFIRM SETTINGS =================
HOLD_SECONDS      = 1.5    # seconds to hold a letter gesture before it's added
COOLDOWN_SECONDS  = 1.0    # wait after adding before accepting next gesture

# ================= UI =================
st.title("🖐 ASL Sentence Builder")

# Sentence display at top
if "sentence" not in st.session_state:
    st.session_state.sentence = ""

sentence_placeholder = st.empty()
sentence_placeholder.markdown(
    f"<div style='font-size:28px; letter-spacing:3px; padding:10px; "
    f"border:2px solid #444; border-radius:8px; min-height:50px;'>"
    f"{st.session_state.sentence or '&nbsp;'}</div>",
    unsafe_allow_html=True
)

st.markdown("""
**Controls:**  
👍 Thumbs Up → `Space`  
👎 Thumbs Down → `Delete last character`  
✋ ASL Letter (hold 1.5s) → adds letter to sentence
""")

run = st.checkbox("Start Camera")
frame_placeholder = st.empty()
status_placeholder = st.empty()

# ================= STATE =================
smooth_conf   = 0.0
alpha         = 0.2
stable_label  = "Waiting..."

hold_label    = None       # which label is currently being held
hold_start    = None       # when the hold started
last_added_t  = 0.0       # timestamp of last addition (cooldown)

cap = cv2.VideoCapture(0)

# ================= LOOP =================
while run:

    ret, frame = cap.read()
    if not ret:
        st.write("Camera error")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape

    # ===== BOX =====
    box_size = 300
    x1 = w // 2 - box_size // 2
    y1 = h // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    # ===== MEDIAPIPE =====
    small    = cv2.resize(frame, (320, 240))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
    result   = detector.detect(mp_image)

    inside_box    = False
    current_label = None    # what we detect this frame
    action_taken  = None    # "letter", "space", or "delete"

    now = time.time()
    in_cooldown = (now - last_added_t) < COOLDOWN_SECONDS

    if result.hand_landmarks:
        hand      = result.hand_landmarks[0]
        landmarks = []
        inside_box = True

        # --- draw dots ---
        for lm in hand:
            px = int(lm.x * w)
            py = int(lm.y * h)
            cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
            landmarks.extend([lm.x, lm.y, lm.z])

        # --- box check: only key points (wrist + 5 fingertips) ---
        key_indices = [0, 4, 8, 12, 16, 20]
        lm_arr = np.array(landmarks).reshape(21, 3)
        inside_box = all(
            x1 < int(lm_arr[i, 0] * w) < x2 and
            y1 < int(lm_arr[i, 1] * h) < y2
            for i in key_indices
        )

        if inside_box and not in_cooldown:

            # ---- check special gestures FIRST ----
            lm_obj = hand   # list of NormalizedLandmark objects

            if is_thumbs_up(lm_obj):
                current_label = "SPACE"

            elif is_thumbs_down(lm_obj):
                current_label = "DELETE"

            else:
                # ---- ASL letter prediction ----
                if len(landmarks) == 63:
                    lms = lm_arr.copy()
                    lms = lms - lms[0]
                    scale = np.max(np.abs(lms))

                    if scale > 1e-6:
                        lms = lms / scale
                        inp = lms.flatten().reshape(1, 1, 63).astype("float32")
                        x_t = torch.tensor(inp).to(device)

                        with torch.no_grad():
                            output = model(x_t)
                            probs  = torch.softmax(output, dim=1)
                            conf, pred = torch.max(probs, dim=1)

                        conf = conf.item()
                        pred = pred.item()
                        smooth_conf = alpha * conf + (1 - alpha) * smooth_conf

                        if smooth_conf > 0.6:
                            current_label = encoder.inverse_transform([pred])[0]

            # ================= HOLD LOGIC =================
            if current_label is not None:

                # start holding new gesture
                if hold_label != current_label:
                    hold_label = current_label
                    hold_start = now

                # check if held long enough
                elif hold_start is not None and (now - hold_start) >= HOLD_SECONDS:

                    if current_label == "SPACE":
                        st.session_state.sentence += " "
                        action_taken = "space"

                    elif current_label == "DELETE":
                        st.session_state.sentence = st.session_state.sentence[:-1]
                        action_taken = "delete"
                    elif current_label == "CLEAR":
                        st.session_state.sentence = ""
                        action_taken = "clear"    

                    else:
                        # letter
                        st.session_state.sentence += current_label
                        action_taken = "letter"

                    last_added_t = now
                    hold_label = None
                    hold_start = None

        else:
            # reset hold if hand disappears or leaves box
            hold_label = None
            hold_start = None

    # ================= DRAW BOX =================
    color = (0, 255, 0) if inside_box else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # ================= DISPLAY TEXT =================
    cv2.putText(frame, f"Gesture: {current_label}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Sentence: {st.session_state.sentence[-40:]}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # ================= STREAMLIT RENDER =================
    frame_placeholder.image(frame, channels="BGR")

    sentence_placeholder.markdown(
        f"<div style='font-size:28px; letter-spacing:3px; padding:10px; "
        f"border:2px solid #444; border-radius:8px; min-height:50px;'>"
        f"{st.session_state.sentence or '&nbsp;'}</div>",
        unsafe_allow_html=True
    )

    status_placeholder.markdown(
        f"**Current:** `{current_label}` | **Action:** `{action_taken}`"
    )

    time.sleep(0.02)

cap.release()