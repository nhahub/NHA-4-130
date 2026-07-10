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

import concurrent.futures
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("gloss_to_text")

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
    "use", "where", "with", "yes", "you", "your",
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
    """Returns de-duplicated, order-preserving warnings about the gloss."""
    warnings: List[str] = []
    if not gloss:
        warnings.append("Empty gloss sequence.")
        return warnings

    unknown_seen = set()
    for w in gloss:
        wl = w.lower()
        if wl not in VOCAB_SET and wl not in unknown_seen:
            unknown_seen.add(wl)
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

# How long to wait for the local model before giving up and falling back to
# the rule-based sentence builder. Configurable via env var so it can be
# tuned per-machine without touching code. Small local models on CPU can
# occasionally hang or take much longer than expected; without a timeout,
# a single stuck call would block the whole real-time inference loop.
OLLAMA_TIMEOUT_SECONDS = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "8"))

# Small local instruction-following models frequently ignore "no markdown /
# no extra text" instructions. These patterns strip the most common leakage.
_CODE_FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$")
_PREAMBLE_RE = re.compile(
    r'^(?:sure[!,.]?|here(?:\'s| is)[^:]*:|sentence:|answer:|output:)\s*',
    re.IGNORECASE,
)


def _clean_llm_output(text: str) -> str:
    """
    Best-effort cleanup of common small-model leakage: markdown fences,
    "Sure! Here's the sentence:" preambles, wrapping quotes, and trailing
    extra sentences beyond the first. This runs BEFORE sanity_check, which
    still independently verifies content-word fidelity.
    """
    text = text.strip()
    text = _CODE_FENCE_RE.sub("", text).strip()
    text = _PREAMBLE_RE.sub("", text).strip()
    text = text.strip('"').strip("'").strip()

    # If the model ignored "exactly one sentence", keep only the first.
    match = re.search(r"[.!?]", text)
    if match:
        text = text[: match.end()]

    return text.strip()


def _call_ollama(gloss: List[str], model: str = OLLAMA_MODEL) -> Optional[str]:
    """
    Calls a local model through Ollama. Returns the raw text response, or
    None on any failure (Ollama not running, model not pulled, timeout,
    etc.) so the caller can fall back gracefully -- same contract as the
    Claude version.

    Requires: pip install ollama
    Requires: `ollama serve` running, and the model already pulled.
    """
    try:
        import ollama
    except ImportError:
        logger.debug("ollama package not installed; skipping local LLM call.")
        return None

    def _do_call() -> Optional[str]:
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
        return response["message"]["content"]

    try:
        # Ollama's Python client has no built-in per-call timeout, so this
        # is enforced with a worker thread. Note the underlying request
        # keeps running in the background if it times out (the thread isn't
        # killed) -- acceptable here since generation is stateless.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_do_call)
            raw_text = future.result(timeout=OLLAMA_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        logger.warning("Ollama call timed out after %.1fs; using fallback.", OLLAMA_TIMEOUT_SECONDS)
        return None
    except Exception:
        logger.warning("Ollama call failed; using fallback.", exc_info=True)
        return None

    if not raw_text:
        return None

    cleaned = _clean_llm_output(raw_text)
    return cleaned if cleaned else None


# ---------------------------------------------------------------------------
# 6. OUTPUT SANITY CHECK (identical logic to the Claude version)
# ---------------------------------------------------------------------------

# Function/grammar words the LLM is explicitly allowed to add per
# SYSTEM_PROMPT rule 2 (articles, common auxiliaries, prepositions,
# conjunctions). Used both to skip gloss words during the "is every gloss
# word present" check, and to tell allowed additions apart from possible
# hallucinated content words.
_ALLOWED_ADDITIONS = {
    "a", "an", "the",
    "is", "am", "are", "was", "were", "do", "does", "did", "will", "to",
    "to", "at", "in", "on", "of", "for",
    "and", "or", "but", "because", "if",
    "not", "no",
}

_FUNCTION_WORDS = {"to", "or", "for", "if", "in", "and", "but", "because", "with", "no", "not"}


def _stem(word: str) -> str:
    w = word.lower()
    # Longer/more specific suffixes must be checked before shorter ones,
    # otherwise e.g. "ies" would only ever match the trailing "es".
    for suf in ("ies", "ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def sanity_check(gloss: List[str], sentence: str) -> List[str]:
    """
    Verifies the LLM output against SYSTEM_PROMPT's two hardest
    constraints:
      - rule 4: every content word from the gloss must appear (by stem)
      - rule 3: no *new* content words may have been introduced

    The previous version of this function only checked the first
    direction, so an LLM output that dropped every gloss word in favor of
    invented content (e.g. gloss ['i','happy'] -> "The weather is nice
    today") would have passed silently.
    """
    warnings: List[str] = []
    sentence_words_raw = re.findall(r"[A-Za-z']+", sentence)
    sentence_stems = {_stem(w) for w in sentence_words_raw}

    gloss_stems = set()
    for g in gloss:
        gl = g.lower()
        gloss_stems.add(_stem(gl))
        if gl in _FUNCTION_WORDS:
            continue
        if _stem(gl) not in sentence_stems and gl not in sentence.lower():
            warnings.append(f"Content word '{g}' from the gloss does not appear to be reflected in the output sentence.")

    # Reverse check: flag output words that are neither in the gloss nor in
    # the explicitly-allowed function-word set. This is a heuristic (it
    # can't catch every synonym substitution), but it catches the clearest
    # hallucination cases, which the old version missed entirely.
    for w in sentence_words_raw:
        wl = w.lower()
        if wl in _ALLOWED_ADDITIONS or _stem(wl) in gloss_stems:
            continue
        warnings.append(f"Word '{w}' in the output does not appear in the gloss and is not an allowed function word (possible hallucination).")

    if sentence.count(".") + sentence.count("?") + sentence.count("!") > 1:
        warnings.append("Output may contain more than one sentence.")

    return warnings


# ---------------------------------------------------------------------------
# 7. RULE-BASED FALLBACK
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Vocabulary categorization (scanned from TARGET_FOLDERS)
# ---------------------------------------------------------------------------
# Every content word in TARGET_FOLDERS falls into one of these buckets.
# Function/closed-class words (after, again, also, always, and, because,
# before, but, can, do, finally, for, from, how, if, in, more, next, no,
# not, now, or, please, to, where, who, will, with, yes, bye, hello,
# "all day") need no special handling -- they pass through unchanged.

# "me" and "i" both surface in gloss as the first-person subject; ASL
# commonly signs the object-form pronoun even when functioning as subject.
_PRONOUNS = {"i": "I", "me": "I"}
_QUESTION_WORDS = {"who", "where", "how", "if"}

# Adjectives / state words: angry, blue, deaf, fun, happy, hungry, sad,
# sick, sorry, tired. These need a copula ("be") inserted before them
# whenever they directly follow a subject.
_STATE_WORDS = {"sad", "angry", "happy", "sick", "tired", "hungry", "deaf", "fun", "blue", "sorry"}

# Subject pronoun -> matching form of "be" ("i"/"me" -> "am", "you" ->
# "are"). Any other subject (a noun or proper noun) defaults to "is",
# handled by _needs_generic_copula below rather than listed here.
_COPULA_BY_SUBJECT = {"i": "am", "me": "am", "you": "are"}

# Verbs: born, buy, choose, drink, eat, enjoy, forget, go, have, help,
# know, like, listen, love, meet, need, play, say, talk, thank,
# understand, use, want.
#
# Verbs that take a "to + verb" complement when immediately followed by
# another action verb in the gloss (e.g. "want drink" -> "want to drink").
_TO_INFINITIVE_VERBS = {"want", "love", "like", "need"}
# Action verbs that can fill that complement slot.
_COMPLEMENT_VERBS = {
    "eat", "drink", "go", "play", "buy", "help", "talk", "meet",
    "understand", "enjoy", "listen", "choose", "use", "know", "say",
}

# Nouns: countable singular nouns get an indefinite article when they
# don't already follow a determiner. Mass/uncountable nouns (bread,
# candy, coffee, milk, people) and possession-implied nouns (name) are
# deliberately left out so no article gets forced onto them.
_ARTICLE_FOR_NOUN = {
    "apple": "an", "adult": "an", "alarm": "an",
    "banana": "a", "book": "a", "baby": "a", "boy": "a",
    "ball": "a", "beard": "a", "car": "a", "cat": "a",
    "camera": "a", "child": "a", "clock": "a", "friend": "a",
    "family": "a", "headache": "a", "phone": "a", "week": "a",
    "conversation": "a", "project": "a", "word": "a", "brother": "a",
    "mustache": "a", "day": "a",
}

# Proper nouns / acronyms: always capitalized regardless of sentence
# position (a plain "sentence[0].upper()" pass doesn't catch these).
_PROPER_NOUN_CAPITALIZATION = {
    "asl": "ASL",
    "egypt": "Egypt",
    "saturday": "Saturday", "sunday": "Sunday", "monday": "Monday",
    "tuesday": "Tuesday", "wednesday": "Wednesday", "thursday": "Thursday",
    "friday": "Friday",
}
_DAYS_OF_WEEK = {"saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday"}

# Adjectives that commonly take an infinitive complement ("happy to meet
# you", "sorry to hear"). Not every state word does this naturally (e.g.
# "tired to meet you" / "blue to meet you" don't read as English), so this
# is a deliberately narrower subset of _STATE_WORDS.
_STATE_WORDS_TAKING_INFINITIVE = {"happy", "sad", "sorry"}

_DETERMINERS = {"a", "an", "the", "my", "your", "next"}
_PREPOSITIONS_BEFORE_DAY = {"on", "before", "after", "in", "next"}

# All standalone nouns (countable + uncountable + proper), used to detect
# when two nouns appear back-to-back with no linking word ("apple banana"
# -> "an apple and a banana"). Uncountable nouns (milk, bread, coffee,
# candy, people, name) don't get articles but still need "and" joining.
_UNCOUNTABLE_NOUNS = {"milk", "bread", "coffee", "candy", "people", "name"}
_ALL_NOUNS = set(_ARTICLE_FOR_NOUN) | _UNCOUNTABLE_NOUNS | set(_PROPER_NOUN_CAPITALIZATION)

# Subject pronouns that take "do" (not "does") for negated verbs
# ("I not want" -> "I do not want"). Third-person/noun subjects would
# need "does" plus verb agreement, which this fallback doesn't attempt --
# see rule_based_fallback's docstring.
_AUX_DO_SUBJECTS = {"i", "me", "you"}
_NEGATABLE_VERBS = _TO_INFINITIVE_VERBS | _COMPLEMENT_VERBS


def _needs_generic_copula(word: str, next_word: Optional[str]) -> bool:
    """
    True when `word` is acting as a non-pronoun subject immediately
    followed by a state word, and so needs a default "is" copula
    ("asl fun" -> "asl is fun"). Pronoun subjects are handled separately
    via _COPULA_BY_SUBJECT since they need "am"/"are" instead of "is".
    """
    if next_word not in _STATE_WORDS:
        return False
    if word in _COPULA_BY_SUBJECT or word in _STATE_WORDS:
        return False
    if word in _FUNCTION_WORDS or word in _QUESTION_WORDS:
        return False
    if word in _TO_INFINITIVE_VERBS or word in _COMPLEMENT_VERBS:
        return False
    return True


def rule_based_fallback(gloss: List[str]) -> str:
    """
    Lightweight, deterministic gloss->sentence builder used when the local
    LLM is unavailable, disabled, or fails its sanity check. It only ever
    adds words from closed classes (copulas, "to", "do"/"does", articles,
    "and", "on"), never new content words, matching the same constraint
    the LLM prompt is held to.

    Known limitations (by design, to keep this predictable rather than
    guessy):
      - No reordering for Wh-questions ("where you go" stays "Where you
        go?" rather than "Where do you go?").
      - "do"/"does" negation support only covers pronoun subjects (i/me/
        you); a noun subject like "cat not want eat" won't get "does".
      - Dual noun/verb gloss words (e.g. "help") are always treated as
        verbs when preceded by want/need/love/like, so "need help" becomes
        "need to help" rather than "need help" (as an object). This is a
        genuine ambiguity in the gloss itself, not something fixable
        without more context.
    """
    if not gloss:
        return ""

    words = [w.lower() for w in gloss]
    is_question = any(w in _QUESTION_WORDS for w in words)

    out: List[str] = []
    prev_was_state_word = False
    prev_was_noun = False
    prev_was_action_verb = False

    for i, w in enumerate(words):
        next_word = words[i + 1] if i + 1 < len(words) else None
        next_next_word = words[i + 2] if i + 2 < len(words) else None

        # Consecutive state words get "and" between them instead of just
        # running together ("I am sad sick" -> "I am sad and sick").
        if w in _STATE_WORDS and prev_was_state_word:
            out.append("and")

        # Consecutive nouns get "and" ("want apple banana" -> "want an
        # apple and a banana"); consecutive action verbs likewise ("eat
        # drink" -> "eat and drink").
        if w in _ALL_NOUNS and prev_was_noun:
            out.append("and")
        if w in _COMPLEMENT_VERBS and prev_was_action_verb:
            out.append("and")

        # Preposition before a bare day-of-week noun ("meet friday" ->
        # "meet on Friday"), unless one is already implied by the
        # preceding word ("before friday", "next friday", "and", etc.).
        if w in _DAYS_OF_WEEK and out and out[-1].lower() not in _PREPOSITIONS_BEFORE_DAY and out[-1].lower() != "and":
            out.append("on")

        # Indefinite article insertion for known singular countable nouns,
        # as long as one isn't already present (e.g. via a possessive or
        # "next").
        if w in _ARTICLE_FOR_NOUN and (not out or out[-1].lower() not in _DETERMINERS):
            out.append(_ARTICLE_FOR_NOUN[w])

        # Proper nouns/acronyms are force-capitalized regardless of
        # position; everything else goes through the pronoun map (which
        # only rewrites "i"/"me" -> "I") unchanged.
        if w in _PROPER_NOUN_CAPITALIZATION:
            token = _PROPER_NOUN_CAPITALIZATION[w]
        else:
            token = _PRONOUNS.get(w, w)
        out.append(token)

        # Adjective + infinitive complement: "happy meet" -> "happy to
        # meet" (a different pattern than the want/love/like/need one
        # below -- here the trigger word is a state word, not a verb).
        if w in _STATE_WORDS_TAKING_INFINITIVE and next_word in _COMPLEMENT_VERBS:
            out.append("to")

        # "want/love/like/need" + action verb -> insert "to".
        if w in _TO_INFINITIVE_VERBS and next_word in _COMPLEMENT_VERBS:
            out.append("to")

        # Copula/auxiliary insertion for "<subject> [not] <word>" patterns:
        #   "I happy"      -> "I am happy"        (pronoun + state word)
        #   "asl fun"      -> "asl is fun"         (noun subject + state word)
        #   "I not happy"  -> "I am not happy"     (copula before "not")
        #   "I not want eat" -> "I do not want eat" (do-support before "not"
        #                                             when what follows is a
        #                                             verb, not a state word)
        if w in _COPULA_BY_SUBJECT:
            if next_word in _STATE_WORDS:
                out.append(_COPULA_BY_SUBJECT[w])
            elif next_word == "not":
                if next_next_word in _STATE_WORDS:
                    out.append(_COPULA_BY_SUBJECT[w])
                elif next_next_word in _NEGATABLE_VERBS and w in _AUX_DO_SUBJECTS:
                    out.append("do")
        elif _needs_generic_copula(w, next_word) or (next_word == "not" and _needs_generic_copula(w, next_next_word)):
            out.append("is")

        prev_was_state_word = w in _STATE_WORDS
        prev_was_noun = w in _ALL_NOUNS
        prev_was_action_verb = w in _COMPLEMENT_VERBS

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
    if not gloss:
        return GlossToTextResult(
            gloss=gloss,
            sentence="",
            method="empty_input",
            confidence=0.0,
            warnings=["Empty gloss sequence; nothing to convert."],
        )

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
            logger.info("Local LLM output failed sanity check: %s", sanity_warnings)
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
    logging.basicConfig(level=logging.INFO)

    test_cases = [
        ["I", "want", "eat", "apple"],
        ["I", "happy", "meet", "friend"],
        ["where", "you", "go"],
        ["I", "sad", "sick", "headache"],
        ["thank", "you", "help", "me"],
        ["I", "not", "happy"],
        ["I", "love", "drink"],
        [],
    ]

    for gloss in test_cases:
        result = gloss_to_text(gloss)
        print(json.dumps(result.to_dict(), indent=2))
        print("-" * 60)