import collections
import time
from typing import List, Optional

import numpy as np

from . import config
from . import nlp_bridge


class Session:
    """Holds all per-client state that used to be local variables inside
    the original script's `run_inference()` while-loop."""

    def __init__(self):
        # dynamic-model state (RAW 155-dim vectors, one per frame)
        self.frame_buffer: List[np.ndarray] = []
        self.pred_history = collections.deque(maxlen=config.SMOOTHING_WINDOW)
        self.current_dynamic_label = "collecting…"
        self.current_dynamic_conf = 0.0

        # static-model state
        self.static_pred_history = collections.deque(maxlen=config.STATIC_SMOOTHING_WINDOW)
        self.current_static_label = "…"
        self.current_static_conf = 0.0
        self.last_static_pred_time = 0.0
        self.last_dynamic_pred_time = 0.0

        # mode: "STATIC" | "DYNAMIC"
        self.mode = "DYNAMIC"

        # NLP gloss pipeline state
        self.gloss_buffer: List[str] = []
        self.spelling_buffer: List[str] = []
        self.generated_sentence = ""
        self.generated_info = ""
        self.llm_enabled = nlp_bridge.NLP_AVAILABLE

        self.current_label = "collecting…"
        self.current_conf = 0.0

    # ── mode / buffer controls, mirroring the C/A/W/B/R/X keys ──────
    def toggle_mode(self):
        self.mode = "DYNAMIC" if self.mode == "STATIC" else "STATIC"
        return self.mode

    def add_current_prediction(self):
        if self.current_label in config.NO_COMMIT_LABELS:
            return {"ok": False, "reason": "no confident prediction right now"}
        if self.mode == "DYNAMIC":
            self.gloss_buffer.append(self.current_label.lower())
        else:
            self.spelling_buffer.append(self.current_label.upper())
        return {"ok": True}

    def finish_word(self):
        if not self.spelling_buffer:
            return {"ok": False, "reason": "spelling buffer is empty"}
        word = "".join(self.spelling_buffer).lower()
        self.gloss_buffer.append(word)
        self.spelling_buffer.clear()
        return {"ok": True, "word": word}

    def undo(self):
        if self.spelling_buffer:
            removed = self.spelling_buffer.pop()
            return {"ok": True, "removed": removed, "scope": "letter"}
        if self.gloss_buffer:
            removed = self.gloss_buffer.pop()
            return {"ok": True, "removed": removed, "scope": "word"}
        return {"ok": False, "reason": "nothing to remove"}

    def toggle_llm(self):
        self.llm_enabled = not self.llm_enabled
        return self.llm_enabled

    def clear(self):
        self.gloss_buffer.clear()
        self.spelling_buffer.clear()
        self.generated_sentence = ""
        self.generated_info = ""

    def reset_buffer(self):
        self.frame_buffer.clear()
        self.pred_history.clear()
        self.static_pred_history.clear()
        self.current_dynamic_label = "collecting…"
        self.current_static_label = "…"

    def generate_sentence(self):
        if not self.gloss_buffer:
            return {"ok": False, "reason": "gloss buffer is empty"}

        oov_result = nlp_bridge.process_oov(self.gloss_buffer)
        words_for_nlp = oov_result.mapped_words

        text_result = nlp_bridge.gloss_to_text(words_for_nlp, self.llm_enabled)

        self.generated_sentence = text_result.sentence
        self.generated_info = f"OOV:{oov_result.method} | NLP:{text_result.method} ({text_result.confidence:.0%})"
        return {
            "ok": True,
            "gloss": list(self.gloss_buffer),
            "sentence": self.generated_sentence,
            "info": self.generated_info,
            "warnings": text_result.warnings,
        }

    # ── per-frame prediction state pick ──────────────────────────
    def sync_current(self):
        if self.mode == "STATIC":
            self.current_label = self.current_static_label
            self.current_conf = self.current_static_conf
        else:
            self.current_label = self.current_dynamic_label
            self.current_conf = self.current_dynamic_conf

    def snapshot(self) -> dict:
        return {
            "mode": self.mode,
            "label": self.current_label,
            "confidence": self.current_conf,
            "buffer_fill": len(self.frame_buffer),
            "buffer_size": config.FIXED_FRAMES,
            "gloss_buffer": list(self.gloss_buffer),
            "spelling_buffer": list(self.spelling_buffer),
            "sentence": self.generated_sentence,
            "sentence_info": self.generated_info,
            "llm_enabled": self.llm_enabled,
        }
