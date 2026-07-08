"""
gloss_to_text_ollama.py
========================
calls a LOCAL model through Ollama instead of the paid Anthropic API. No API key, no billing, no internet
required after the model is downloaded once.

-------------------------------------------------------------------------
SETUP (one-time)
-------------------------------------------------------------------------
1. Install Ollama (runs models locally on your own machine/GPU/CPU):
   - Windows / Mac: download from https://ollama.com/download
   - Linux:
       curl -fsSL https://ollama.com/install.sh | sh

2. Pull a small, free, instruction-following model. For a constrained
   grammar-fixing task like this, a small model is enough. Pick ONE:

       ollama pull llama3.1:8b       # good quality, ~4.7GB, needs ~8GB RAM
       ollama pull qwen2.5:7b        # good at following strict instructions
       ollama pull phi3:mini         # smallest/fastest, ~2.3GB, weaker

   (If your laptop is low on RAM, use phi3:mini or qwen2.5:3b.)

3. Start the Ollama server (it usually auto-starts after install; if not):
       ollama serve

4. Install the Python client:
       pip install ollama

5. Run this file:
       python3 gloss_to_text_ollama.py

That's it — everything runs on your machine, completely free, no API key.
-------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# 1. VOCABULARY (same as before — from your TARGET_FOLDERS list)
# ---------------------------------------------------------------------------

TARGET_FOLDERS = sorted(list(set([
    "if", "conversation", "to", "sad", "say", "or", "for", "alarm", "adult",
    "after", "all day", "angry", "baby", "banana", "beard", "before", "blue",
    "book", "bread", "want", "buy", "cat", "camera", "sick", "headache",
    "go", "brother", "born", "egypt", "also", "apple", "candy", "child",
    "choose", "coffee", "clock", "forget", "how", "hungry", "in", "know",
    "listen", "me", "milk", "mustache", "please", "play", "saturday",
    "sunday", "monday", "tuesday", "wednesday", "thursday", "friday",
    "will", "meet", "always", "finally", "project", "help", "word", "no",
    "need", "more", "next", "week", "who", "I", "again", "and", "asl",
    "ball", "because", "boy", "but", "bye", "can", "car", "day", "deaf",
    "do", "drink", "eat", "enjoy", "family", "friend", "from", "fun",
    "happy", "have", "hello", "like", "love", "my", "name", "not", "now",
    "people", "phone", "sorry", "talk", "thank", "tired", "understand",
    "use", "where", "with", "yes", "your",
])))

VOCAB_SET = {w.lower() for w in TARGET_FOLDERS}


# ---------------------------------------------------------------------------
# 2. RESULT CONTAINER
# ---------------------------------------------------------------------------

@dataclass
class GlossToTextResult:
    gloss: List[str]
    sentence: str
    method: str                     # "llm_local" or "rule_based_fallback"
    confidence: float = 1.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "gloss": self.gloss,
            "sentence": self.sentence,
            "method": self.method,
            "confidence": self.confidence,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# 3. INPUT VALIDATION
# ---------------------------------------------------------------------------

def validate_gloss(gloss: List[str]) -> List[str]:
    warnings = []
    if not gloss:
        warnings.append("Empty gloss sequence.")
        return warnings
    for w in gloss:
        if w.lower() not in VOCAB_SET:
            warnings.append(f"Word '{w}' is not in the trained vocabulary (possible model hallucination or OOV sign).")
    return warnings


# ---------------------------------------------------------------------------
# 4. PROMPT DESIGN (identical rules to the Claude version)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You convert ASL gloss word sequences into ONE fluent, natural English sentence.

STRICT RULES:
1. Output EXACTLY one sentence. Nothing before or after it. No preamble, no explanation, no quotes.
2. You may ONLY add small grammatical function words: articles (a/an/the), auxiliary verbs (is/am/are/do/does/did/will), prepositions (to/at/in/on), and conjunctions needed for grammar.
3. You must NOT add any new content words: no new nouns, verbs, adjectives, names, numbers, times, or facts that are not implied by the given gloss words.
4. You must NOT remove or replace any content word's core meaning. Every content word in the gloss must be reflected in the sentence.
5. Emotion / sentiment words in the gloss (e.g. sad, angry, happy, love, like, sorry, tired, sick, hungry, thank, please, fun, enjoy) must be preserved with their exact meaning -- do not soften, intensify, or swap them for a synonym.
6. Reorder words only as needed for correct English grammar (ASL gloss order often differs from English order).
7. If the gloss is a question (contains words like "who", "where", "how", "if"), output a question with correct punctuation.
8. Keep the tone neutral and literal. Do not editorialize or infer intent beyond what is signed.
9. If the gloss is ambiguous or fragmentary, produce the most literal, minimal sentence possible rather than guessing extra meaning.

Return ONLY the final sentence, with no extra text, no markdown, no quotation marks around it."""


def build_user_prompt(gloss: List[str]) -> str:
    gloss_str = " ".join(gloss)
    return f'ASL gloss sequence: [{gloss_str}]\n\nConvert this into one fluent English sentence following the rules exactly.'


# ---------------------------------------------------------------------------
# 5. LOCAL LLM CALL (Ollama)
# ---------------------------------------------------------------------------

# Change this to whichever model you pulled with `ollama pull <name>`.
OLLAMA_MODEL = "phi3:mini"


def _call_ollama(gloss: List[str], model: str = OLLAMA_MODEL) -> Optional[str]:
    """
    Calls a local model through Ollama. Returns the raw text response, or
    None on any failure (Ollama not running, model not pulled, etc.) so the
    caller can fall back gracefully -- same contract as the Claude version.

    Requires: pip install ollama
    Requires: `ollama serve` running, and the model already pulled.
    """
    try:
        import ollama
    except ImportError:
        return None

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(gloss)},
            ],
            options={
                "temperature": 0,   # deterministic, this is formatting not creative writing
            },
        )
        text = response["message"]["content"].strip()
        # Small local models sometimes wrap output in quotes despite instructions -- strip them.
        text = text.strip('"').strip("'").strip()
        return text if text else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 6. OUTPUT SANITY CHECK (identical logic to the Claude version)
# ---------------------------------------------------------------------------

_FUNCTION_WORDS = {"to", "or", "for", "if", "in", "and", "but", "because", "with", "no", "not"}

def _stem(word: str) -> str:
    w = word.lower()
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def sanity_check(gloss: List[str], sentence: str) -> List[str]:
    warnings = []
    sentence_words = {_stem(w) for w in re.findall(r"[A-Za-z']+", sentence)}
    for g in gloss:
        gl = g.lower()
        if gl in _FUNCTION_WORDS:
            continue
        if _stem(gl) not in sentence_words and gl not in sentence.lower():
            warnings.append(f"Content word '{g}' from the gloss does not appear to be reflected in the output sentence.")
    if sentence.count(".") + sentence.count("?") + sentence.count("!") > 1:
        warnings.append("Output may contain more than one sentence.")
    return warnings


# ---------------------------------------------------------------------------
# 7. RULE-BASED FALLBACK (identical to the Claude version)
# ---------------------------------------------------------------------------

_PRONOUNS = {"i": "I"}
_QUESTION_WORDS = {"who", "where", "how", "if"}

def rule_based_fallback(gloss: List[str]) -> str:
    if not gloss:
        return ""

    words = [w.lower() for w in gloss]
    is_question = any(w in _QUESTION_WORDS for w in words)

    out = []
    for i, w in enumerate(words):
        token = _PRONOUNS.get(w, w)
        if w == "want" and i + 1 < len(words) and words[i + 1] in {"eat", "drink", "go", "play", "buy", "help", "talk", "meet"}:
            out.append(token)
            out.append("to")
            continue
        out.append(token)

    sentence = " ".join(out)
    sentence = sentence[0].upper() + sentence[1:] if sentence else sentence
    sentence = sentence.rstrip(".?!")
    sentence += "?" if is_question else "."
    return sentence


# ---------------------------------------------------------------------------
# 8. PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

def gloss_to_text(gloss: List[str], model: str = OLLAMA_MODEL) -> GlossToTextResult:
    """
    Main function to call from the FastAPI endpoint (Stage 3 of the
    architecture: "Text -> Sign Mapping" / NLP Mapping block).

    Example:
        >>> result = gloss_to_text(["I", "want", "eat", "apple"])
        >>> result.sentence
        'I want to eat an apple.'
    """
    warnings = validate_gloss(gloss)

    llm_output = _call_ollama(gloss, model=model)

    if llm_output:
        sanity_warnings = sanity_check(gloss, llm_output)
        if not sanity_warnings:
            return GlossToTextResult(
                gloss=gloss,
                sentence=llm_output,
                method="llm_local",
                confidence=0.9,
                warnings=warnings,
            )
        else:
            warnings = warnings + sanity_warnings + ["Local LLM output failed sanity check; used rule-based fallback instead."]

    fallback_sentence = rule_based_fallback(gloss)
    return GlossToTextResult(
        gloss=gloss,
        sentence=fallback_sentence,
        method="rule_based_fallback",
        confidence=0.6,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# 9. QUICK MANUAL TEST / DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        ["I", "want", "eat", "apple"],
        ["I", "happy", "meet", "friend"],
        ["where", "you", "go"],
        ["I", "sad", "sick", "headache"],
        ["thank", "you", "help", "me"],
    ]

    for gloss in test_cases:
        result = gloss_to_text(gloss)
        print(json.dumps(result.to_dict(), indent=2))
        print("-" * 60)
