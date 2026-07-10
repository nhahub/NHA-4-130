"""
Bridges the gloss buffer to a fluent sentence:

    gloss_buffer -> process_oov() -> gloss_to_text() -> sentence

Drop your real `oov_handler.py` and `gloss_to_text_ollama.py` next to
this file (backend/app/) to use your trained OOV logic / Ollama model.
If they aren't present, lightweight built-in fallbacks are used so the
app still works end-to-end out of the box.
"""
from dataclasses import dataclass, field
from typing import List

try:
    from .oov_handler import process_oov as _process_oov  # type: ignore
except Exception:
    _process_oov = None

try:
    from .gloss_to_text_ollama import (  # type: ignore
        gloss_to_text as _gloss_to_text,
        rule_based_fallback as _rule_based_fallback,
        GlossToTextResult as _GlossToTextResult,
    )
except Exception:
    _gloss_to_text = None
    _rule_based_fallback = None
    _GlossToTextResult = None


@dataclass
class OOVResult:
    mapped_words: List[str]
    method: str = "no_oov_handler"


@dataclass
class GlossToTextResult:
    gloss: List[str]
    sentence: str
    method: str
    confidence: float
    warnings: list = field(default_factory=list)


def process_oov(gloss_buffer: List[str]) -> OOVResult:
    if _process_oov is not None:
        result = _process_oov(gloss_buffer)
        return OOVResult(mapped_words=result.mapped_words, method=result.method)
    # fallback: passthrough, no OOV resolution
    return OOVResult(mapped_words=list(gloss_buffer), method="no_oov_handler")


def rule_based_fallback(words: List[str]) -> str:
    """Very small rule-based gloss -> sentence builder used when no LLM
    is available/enabled. Capitalizes first word, joins with spaces,
    appends a period."""
    if _rule_based_fallback is not None:
        return _rule_based_fallback(words)
    if not words:
        return ""
    sentence = " ".join(words)
    sentence = sentence[0].upper() + sentence[1:]
    if not sentence.endswith((".", "?", "!")):
        sentence += "."
    return sentence


def gloss_to_text(words: List[str], llm_enabled: bool) -> GlossToTextResult:
    """If llm_enabled and a real gloss_to_text_ollama.py is available,
    delegate to it. Otherwise use the rule-based fallback directly
    (mirrors the 'L' key toggle in the original script)."""
    if llm_enabled and _gloss_to_text is not None:
        result = _gloss_to_text(words)
        return GlossToTextResult(
            gloss=result.gloss,
            sentence=result.sentence,
            method=result.method,
            confidence=result.confidence,
            warnings=list(getattr(result, "warnings", [])),
        )

    sentence = rule_based_fallback(words)
    warnings = []
    if llm_enabled and _gloss_to_text is None:
        warnings.append("gloss_to_text_ollama.py not available; used rule-based fallback.")
    elif not llm_enabled:
        warnings.append("LLM disabled by user; used rule-based fallback.")

    return GlossToTextResult(
        gloss=words,
        sentence=sentence,
        method="rule_based_forced" if not llm_enabled else "rule_based_fallback",
        confidence=0.6,
        warnings=warnings,
    )


NLP_AVAILABLE = _gloss_to_text is not None
