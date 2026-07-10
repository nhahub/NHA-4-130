import {
  HandLandmarker,
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ── configuration ──────────────────────────────────────────────────
const WS_URL = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
const HAND_MODEL_URL = "./models/hand_landmarker.task";
const POSE_MODEL_URL = "./models/pose_landmarker_lite.task";
const POSE_IDXS = [0, 11, 12, 13, 14, 15, 16, 23, 24];
const RAW_DIM = 155;
const STATIC_DIM = 63;
// No artificial send throttle: every detected frame's features are sent
// immediately. The backend's own cooldowns (STATIC_PRED_COOLDOWN /
// PRED_COOLDOWN) control how often the (slower) models actually run —
// sending features every frame just keeps the rolling buffer and the
// skeleton overlay perfectly in sync with the camera, with no added lag.
//
// Frames are sent as a compact binary WebSocket message instead of JSON
// (skips JSON.stringify/json.loads on ~150-200 floats every frame):
//   byte 0        : message type — 1 = frame
//   byte 1        : has_raw63 flag (0/1)
//   bytes 2..622  : 155 x float32 (raw155), little-endian
//   bytes 622..874: 63 x float32 (raw63), only present if has_raw63
function packFrame(raw155, raw63) {
  const has63 = raw63 !== null && raw63 !== undefined;
  const buf = new ArrayBuffer(2 + RAW_DIM * 4 + (has63 ? STATIC_DIM * 4 : 0));
  const view = new DataView(buf);
  view.setUint8(0, 1);
  view.setUint8(1, has63 ? 1 : 0);
  let off = 2;
  for (let i = 0; i < RAW_DIM; i++, off += 4) view.setFloat32(off, raw155[i], true);
  if (has63) for (let i = 0; i < STATIC_DIM; i++, off += 4) view.setFloat32(off, raw63[i], true);
  return buf;
}

// ── DOM ─────────────────────────────────────────────────────────────
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");

const wsStatusEl = document.getElementById("wsStatus");
const wsDot = document.getElementById("wsDot");
const camDot = document.getElementById("camDot");
const modelDot = document.getElementById("modelDot");

const fpsBadge = document.getElementById("fpsBadge");
const modeBadge = document.getElementById("modeBadge");
const labelMain = document.getElementById("labelMain");
const labelConf = document.getElementById("labelConf");
const bufferFill = document.getElementById("bufferFill");
const bufferTxt = document.getElementById("bufferTxt");
const handsTxt = document.getElementById("handsTxt");
const top5El = document.getElementById("top5");
const glossBox = document.getElementById("glossBox");
const sentenceBox = document.getElementById("sentenceBox");
const sentenceMeta = document.getElementById("sentenceMeta");

const btnMode = document.getElementById("btnMode");
const btnAdd = document.getElementById("btnAdd");
const btnWord = document.getElementById("btnWord");
const btnUndo = document.getElementById("btnUndo");
const btnReset = document.getElementById("btnReset");
const btnToggleLandmarks = document.getElementById("btnToggleLandmarks");
const btnLLM = document.getElementById("btnLLM");
const btnGenerate = document.getElementById("btnGenerate");
const btnClear = document.getElementById("btnClear");
const btnSpeak = document.getElementById("btnSpeak");
const btnAutoSpeak = document.getElementById("btnAutoSpeak");

// ── state ───────────────────────────────────────────────────────────
let handLandmarker, poseLandmarker;
let ws;
let lastMode = "DYNAMIC";
let lastSentence = "";
let autoSpeak = false;
let showLandmarks = true;
let lastHandFlags = { lh: false, rh: false };

// ═════════════════════════════════════════════════════════════════
// 1. Feature extraction — mirrors extract_frame_features /
//    extract_static_features / normalize_hand_local from the python
//    reference implementation, operating on MediaPipe Tasks JS output.
// ═════════════════════════════════════════════════════════════════
function dist3(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function normalizeHandLocal(pts) {
  // pts: array of 21 [x,y,z], already anchor/scale-adjusted
  const wrist = pts[0];
  const centered = pts.map((p) => [p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]]);
  const mcpIdx = [5, 9, 13, 17];
  let sum = 0;
  for (const i of mcpIdx) sum += Math.hypot(centered[i][0], centered[i][1], centered[i][2]);
  const palmSize = sum / mcpIdx.length + 1e-6;
  return centered.map((c) => [c[0] / palmSize, c[1] / palmSize, c[2] / palmSize]);
}

function extractFrameFeatures(handResult, poseResult) {
  let lh = Array.from({ length: 21 }, () => [0, 0, 0]);
  let rh = Array.from({ length: 21 }, () => [0, 0, 0]);
  let poseFeats = Array.from({ length: POSE_IDXS.length }, () => [0, 0, 0]);
  let lhPresent = 0.0, rhPresent = 0.0;

  let anchor = [0, 0, 0];
  let scale = 1.0;

  const poseLms = poseResult && poseResult.landmarks && poseResult.landmarks[0];
  if (poseLms) {
    const lSh = [poseLms[11].x, poseLms[11].y, poseLms[11].z];
    const rSh = [poseLms[12].x, poseLms[12].y, poseLms[12].z];
    anchor = [(lSh[0] + rSh[0]) / 2, (lSh[1] + rSh[1]) / 2, (lSh[2] + rSh[2]) / 2];
    scale = dist3(lSh, rSh) + 1e-6;
    poseFeats = POSE_IDXS.map((idx) => {
      const p = poseLms[idx];
      return [(p.x - anchor[0]) / scale, (p.y - anchor[1]) / scale, (p.z - anchor[2]) / scale];
    });
  }

  const handLms = handResult && handResult.landmarks;
  const handedness = handResult && handResult.handedness;
  let sawLeft = false, sawRight = false;
  if (handLms && handLms.length) {
    handLms.forEach((pts, i) => {
      const label = handedness && handedness[i] && handedness[i][0]
        ? handedness[i][0].categoryName
        : "Right";
      let arr = pts.map((lm) => [
        (lm.x - anchor[0]) / scale,
        (lm.y - anchor[1]) / scale,
        (lm.z - anchor[2]) / scale,
      ]);
      arr = normalizeHandLocal(arr);
      if (label === "Left") { rh = arr; rhPresent = 1.0; sawRight = true; }   // camera-left = signer-right
      else { lh = arr; lhPresent = 1.0; sawLeft = true; }
    });
  }
  lastHandFlags = { lh: sawLeft, rh: sawRight };

  const flat = (arr) => arr.flat();
  return [...flat(lh), ...flat(rh), ...flat(poseFeats), lhPresent, rhPresent]; // 155-dim
}

function extractStaticFeatures(handResult) {
  if (!handResult || !handResult.landmarks || !handResult.landmarks.length) return null;
  const hand = handResult.landmarks[0];
  let pts = hand.map((lm) => [lm.x, lm.y, lm.z]);
  const wrist = pts[0];
  pts = pts.map((p) => [p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]]);
  let maxAbs = 0;
  for (const p of pts) for (const v of p) maxAbs = Math.max(maxAbs, Math.abs(v));
  if (maxAbs < 1e-6) return null;
  pts = pts.map((p) => p.map((v) => v / maxAbs));
  return pts.flat(); // 63-dim
}

// ═════════════════════════════════════════════════════════════════
// 2. MediaPipe setup
// ═════════════════════════════════════════════════════════════════
async function setupMediaPipe() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  poseLandmarker = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  modelDot.className = "dot ok";
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((res) => (video.onloadedmetadata = res));
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  camDot.className = "dot ok";
}

// ═════════════════════════════════════════════════════════════════
// 3. WebSocket
// ═════════════════════════════════════════════════════════════════
function connectWS() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    wsStatusEl.textContent = "connected";
    wsDot.className = "dot ok";
  };
  ws.onclose = () => {
    wsStatusEl.textContent = "disconnected — retrying…";
    wsDot.className = "dot bad";
    setTimeout(connectWS, 1500);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);
    if (msg.type === "prediction") {
      updateHUD(msg.state, msg.top5);
    } else if (msg.type === "control_result") {
      updateHUD(msg.state, null);
      if (msg.action === "generate" && msg.result && msg.result.ok && autoSpeak) {
        speak(msg.result.sentence);
      }
    }
  };
}

function sendControl(action) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "control", action }));
  }
}

// ═════════════════════════════════════════════════════════════════
// 4. HUD rendering
// ═════════════════════════════════════════════════════════════════
function updateHUD(state, top5) {
  if (!state) return;
  lastMode = state.mode;

  modeBadge.textContent = state.mode === "STATIC" ? "STATIC (letter/digit)" : "DYNAMIC (word)";
  modeBadge.className = "badge " + (state.mode === "STATIC" ? "mode-static" : "mode-dynamic");

  const uncertain = ["uncertain", "no hand", "collecting…", "…"].includes(state.label);
  labelMain.textContent = String(state.label).toUpperCase();
  labelMain.className = "label-main" + (uncertain ? " uncertain" : "");
  labelConf.textContent = uncertain ? "" : `${(state.confidence * 100).toFixed(0)}% confidence`;

  const pct = Math.min(100, (state.buffer_fill / state.buffer_size) * 100);
  bufferFill.style.width = pct + "%";
  bufferTxt.textContent = `buffer ${state.buffer_fill}/${state.buffer_size}`;

  handsTxt.textContent = `L-hand: ${lastHandFlags.lh ? "✓" : "✗"}   R-hand: ${lastHandFlags.rh ? "✓" : "✗"}`;

  if (top5 && top5.classes && top5.classes.length) {
    top5El.innerHTML = top5.classes
      .map((c, i) => {
        const s = top5.scores[i];
        return `<div class="top5-row">
          <div class="top5-label">${c}</div>
          <div class="top5-track"><div class="top5-fill" style="width:${(s * 100).toFixed(1)}%"></div></div>
          <div class="top5-score">${(s * 100).toFixed(0)}%</div>
        </div>`;
      })
      .join("");
  }

  glossBox.innerHTML =
    (state.gloss_buffer && state.gloss_buffer.length
      ? state.gloss_buffer.join(" ")
      : '<span class="empty">(empty)</span>') +
    (state.spelling_buffer && state.spelling_buffer.length
      ? ` <span class="spelling">[${state.spelling_buffer.join("-")}]</span>`
      : "");

  sentenceBox.innerHTML = state.sentence
    ? state.sentence
    : '<span class="empty">Press "Generate sentence" once you have a few words.</span>';
  sentenceMeta.textContent = state.sentence_info || "";

  if (state.sentence && state.sentence !== lastSentence) {
    lastSentence = state.sentence;
  }

  btnLLM.textContent = `LLM: ${state.llm_enabled ? "on" : "off"}`;
  btnLLM.className = state.llm_enabled ? "on" : "off";
}

// ═════════════════════════════════════════════════════════════════
// 5. Text-to-speech (Web Speech API — entirely client-side)
// ═════════════════════════════════════════════════════════════════
function speak(text) {
  if (!text || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 1.0;
  utter.pitch = 1.0;
  window.speechSynthesis.speak(utter);
}

btnSpeak.addEventListener("click", () => speak(lastSentence));
btnAutoSpeak.addEventListener("click", () => {
  autoSpeak = !autoSpeak;
  btnAutoSpeak.textContent = `Auto-speak: ${autoSpeak ? "on" : "off"}`;
  btnAutoSpeak.className = "wide " + (autoSpeak ? "on" : "off");
});

// ═════════════════════════════════════════════════════════════════
// 6. Controls (buttons + keyboard, mirroring the desktop app's keys)
// ═════════════════════════════════════════════════════════════════
btnMode.addEventListener("click", () => sendControl("toggle_mode"));
btnAdd.addEventListener("click", () => sendControl("add"));
btnWord.addEventListener("click", () => sendControl("word"));
btnUndo.addEventListener("click", () => sendControl("undo"));
btnReset.addEventListener("click", () => sendControl("reset"));
btnToggleLandmarks.addEventListener("click", () => toggleLandmarks());

function toggleLandmarks() {
  showLandmarks = !showLandmarks;
  btnToggleLandmarks.innerHTML = `Landmarks: ${showLandmarks ? "on" : "off"} <kbd>V</kbd>`;
  btnToggleLandmarks.className = showLandmarks ? "on" : "off";
  if (!showLandmarks) ctx.clearRect(0, 0, overlay.width, overlay.height);
}
btnLLM.addEventListener("click", () => sendControl("toggle_llm"));
btnGenerate.addEventListener("click", () => sendControl("generate"));
btnClear.addEventListener("click", () => sendControl("clear"));

window.addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();
  if (k === "v") { toggleLandmarks(); return; }
  const map = { c: "toggle_mode", a: "add", w: "word", b: "undo", r: "reset", l: "toggle_llm", g: "generate", x: "clear" };
  if (map[k]) sendControl(map[k]);
});

// ═════════════════════════════════════════════════════════════════
// 7. Main detection loop
// ═════════════════════════════════════════════════════════════════
const fpsHistory = [];
const hasVFC = typeof video.requestVideoFrameCallback === "function";

function drawSkeleton(handResult) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!showLandmarks || !handResult || !handResult.landmarks) return;
  const CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17],
  ];
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.lineWidth = 2;
  for (const pts of handResult.landmarks) {
    const px = pts.map((lm) => [lm.x * overlay.width, lm.y * overlay.height]);
    for (const [a, b] of CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(px[a][0], px[a][1]);
      ctx.lineTo(px[b][0], px[b][1]);
      ctx.stroke();
    }
    for (const [x, y] of px) {
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function processFrame() {
  const t0 = performance.now();
  const ts = performance.now();

  if (video.readyState >= 2) {
    const handResult = handLandmarker.detectForVideo(video, ts);
    const poseResult = poseLandmarker.detectForVideo(video, ts);
    drawSkeleton(handResult);

    // Send every processed frame — no artificial throttle. The backend's
    // own cooldowns gate how often the (slower) models actually run.
    if (ws && ws.readyState === WebSocket.OPEN) {
      const raw155 = extractFrameFeatures(handResult, poseResult);
      const raw63 = lastMode === "STATIC" ? extractStaticFeatures(handResult) : null;
      ws.send(packFrame(raw155, raw63));
    }
  }

  fpsHistory.push(performance.now() - t0);
  if (fpsHistory.length > 30) fpsHistory.shift();
  const avg = fpsHistory.reduce((a, b) => a + b, 0) / fpsHistory.length;
  fpsBadge.textContent = `FPS: ${(1000 / Math.max(avg, 1)).toFixed(0)}`;
}

function loop() {
  // requestVideoFrameCallback fires exactly once per *actual new* decoded
  // camera frame (no duplicate processing of the same frame, no drift),
  // which is lower-latency than requestAnimationFrame for this purpose.
  // Falls back to rAF on browsers that don't support it (e.g. Firefox).
  if (hasVFC) {
    video.requestVideoFrameCallback(() => {
      processFrame();
      loop();
    });
  } else {
    processFrame();
    requestAnimationFrame(loop);
  }
}

// ═════════════════════════════════════════════════════════════════
// boot
// ═════════════════════════════════════════════════════════════════
(async function main() {
  try {
    await Promise.all([setupMediaPipe(), setupCamera()]);
    connectWS();
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    wsStatusEl.textContent = "startup error — see console";
    alert("Startup failed: " + err.message + "\nCheck camera permissions and that /models/*.task files exist.");
  }
})();
