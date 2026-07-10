"""
oov_handler.py

Out-Of-Vocabulary (OOV) handler for an American Sign Language (ASL)
translation system.

Where this fits in the pipeline
--------------------------------
Speech/Text -> OOV Handler -> Gloss Generator -> Sign Mapping

The sign-language recognition model was trained on ~120 vocabulary words.
This OOV handler detects words outside that known sign vocabulary and
rewrites them into semantically similar English words while preserving
meaning. The Gloss Generator later converts the rewritten sentence into
sign-supported gloss.

The handler currently uses four fallback layers in order:

    1. LLM (Ollama)        - best quality, needs a running local model
    2. Semantic similarity - no LLM needed, needs sentence-transformers
    3. Manual dictionary   - instant, zero dependencies, curated by hand
    4. Keep original word  - final safety net, guarantees no crash

Each layer only runs if the previous one failed or produced an invalid
result, so the module always returns *something* usable, and prefers the
highest-quality answer it can actually produce.

Install:
    pip install ollama                     # Layer 1 (LLM)
    ollama pull phi3:mini
    pip install sentence-transformers       # Layer 2 (semantic similarity, optional)

Run:
    python3 oov_handler.py
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

logger = logging.getLogger("oov_handler")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Local Ollama model used to resolve OOV words (Layer 1).
OLLAMA_MODEL: str = "phi3:mini"

#: Sampling temperature for the LLM call. Lower = more conservative,
#: more likely to stick to the vocabulary instead of getting creative.
OLLAMA_TEMPERATURE: float = 0.2

#: Maximum number of replacement words accepted for a single OOV word.
MAX_REPLACEMENT_WORDS: int = 4

#: sentence-transformers model used for Layer 2. Small and fast; downloads
#: once and is cached locally afterward.
SEMANTIC_MODEL_NAME: str = "all-MiniLM-L6-v2"

#: Minimum cosine similarity (0-1) required to accept a semantic match.
#: Short single-word embeddings are noisy, so this is set high on purpose
#: to reject weak/coincidental matches (e.g. "hospital" ~ "egypt" at 0.37
#: is NOT a real semantic relationship, just the least-bad option in a
#: small vocabulary). Raise this further if Layer 2 still proposes
#: unrelated words; lower it only if it's rejecting genuinely good matches.
SEMANTIC_SIMILARITY_THRESHOLD: float = 0.55

#: How many top semantic matches to consider returning. A second word is
#: only kept if it also clears SEMANTIC_SECOND_MATCH_MARGIN below.
SEMANTIC_TOP_K: int = 2

#: A second semantic match is only included if its score is within this
#: margin of the top match's score (prevents weak "runner-up" words like
#: "candy" tagging along with "eat" for "pizza").
SEMANTIC_SECOND_MATCH_MARGIN: float = 0.05


# --------------------------------------------------------------------------
# Vocabulary (same list the recognition model was trained on)
# --------------------------------------------------------------------------

TARGET_FOLDERS = sorted(list(set([
    "if", "conversation", "to", "sad", "say", "or", "for", "alarm", "after",
    "angry", "banana", "beard", "before", "blue", "book", "want", "buy",
    "cat", "camera", "sick", "headache", "go", "brother", "egypt", "apple",
    "candy", "child", "coffee", "forget", "how", "in", "know", "listen",
    "me", "milk", "please", "play", "saturday", "sunday", "monday",
    "tuesday", "wednesday", "thursday", "friday", "will", "meet", "always",
    "finally", "project", "help", "word", "no", "need", "more", "next",
    "week", "who", "I", "again", "and", "asl", "ball", "because", "boy",
    "but", "bye", "can", "car", "day", "deaf", "do", "drink", "eat",
    "enjoy", "family", "friend", "from", "fun", "happy", "have", "hello",
    "like", "love", "my", "name", "not", "now", "people", "phone", "sorry",
    "talk", "thank", "tired", "understand", "use", "where", "with", "yes",
    "your",
])))

VOCAB_SET = set(TARGET_FOLDERS)
_VOCAB_SET_LOWER = {w.lower() for w in TARGET_FOLDERS}


#: Layer 3 - manual synonym dictionary. Replacement words may be any
#: simple English words or short phrases. These entries are intended to
#: preserve meaning; the Gloss Generator will later convert the rewritten
#: sentence into glossary-supported sign gloss. The OOV handler does not
#: require these words to belong to the sign vocabulary.
MANUAL_SYNONYMS: Dict[str, List[str]] = {
    "hospital": ["sick"],
    "doctor": ["help", "sick"],
    "nurse": ["help", "sick"],
    "medicine": ["sick", "help"],
    "pizza": ["eat"],
    "sandwich": ["eat"],
    "burger": ["eat"],
    "restaurant": ["eat"],
    "juice": ["drink"],
    "soda": ["drink"],
    "water": ["drink"],
    "tea": ["coffee"],
    "laptop": ["phone"],
    "computer": ["phone"],
    "television": ["phone"],
    "tv": ["phone"],
    "mom": ["family"],
    "dad": ["family"],
    "mother": ["family"],
    "father": ["family"],
    "sister": ["brother", "family"],
    "parents": ["family"],
    "dog": ["cat"],
    "pet": ["cat"],
    "school": ["project"],
    "teacher": ["friend"],
    "class": ["project"],
    "movie": ["fun"],
    "game": ["play", "fun"],
    "sports": ["play"],
    "weekend": ["saturday", "sunday"],
    "tomorrow": ["next", "day"],
    "yesterday": ["before", "day"],
    "today": ["now", "day"],
}


# --------------------------------------------------------------------------
# Result object
# --------------------------------------------------------------------------

@dataclass
class OOVResult:
    """Holds the outcome of an OOV resolution pass over a word list.

    Attributes:
        original_words: The input word list, unchanged.
        mapped_words: The final word list after OOV replacement, ready to
            be handed to gloss_to_text.py.
        method: A summary of which layer(s) resolved the OOV words in this
            call: "no_oov" (nothing to resolve), "llm_local" (all resolved
            by the LLM), "semantic" (all resolved by embeddings),
            "dictionary" (all resolved by the manual dictionary), "mixed"
            (different words resolved by different layers), or "fallback"
            (nothing could be resolved; originals were kept).
        warnings: Any issues encountered along the way.
        confidence: Average per-word confidence across all OOV words,
            based on which layer resolved each one (see
            CONFIDENCE_BY_LAYER). 1.0 for a fully LLM-resolved mapping,
            lower as weaker layers are relied on, 0.0 on total failure.
        replacements: Maps each OOV input word to the list of replacement
            words it was replaced with (or to itself if unresolved).
        resolution_layers: Maps each OOV input word to the layer that
            resolved it: "llm", "semantic", "dictionary", or "unresolved".
    """
    original_words: List[str]
    mapped_words: List[str]
    method: str
    warnings: List[str] = field(default_factory=list)
    confidence: float = 1.0
    replacements: Dict[str, List[str]] = field(default_factory=dict)
    resolution_layers: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a plain-dict / JSON-serializable representation."""
        return {
            "original_words": self.original_words,
            "mapped_words": self.mapped_words,
            "method": self.method,
            "warnings": self.warnings,
            "confidence": self.confidence,
            "replacements": self.replacements,
            "resolution_layers": self.resolution_layers,
        }

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


#: Confidence assigned to a word depending on which layer resolved it.
CONFIDENCE_BY_LAYER: Dict[str, float] = {
    "llm": 1.0,
    "semantic": 0.75,
    "dictionary": 0.6,
    "unresolved": 0.3,
}


# --------------------------------------------------------------------------
# Step 1: find OOV words
# --------------------------------------------------------------------------

def find_oov_words(words: List[str]) -> List[str]:
    """Identify which words in the input are outside the known sign vocabulary.

    Args:
        words: The input word list (e.g. classifier or user-typed words).

    Returns:
        A list of the distinct OOV words, in first-seen order. Case is
        ignored for the membership check but the original casing of each
        word is preserved in the returned list.
    """
    seen: set = set()
    oov: List[str] = []
    for word in words:
        if word.lower() not in _VOCAB_SET_LOWER and word.lower() not in seen:
            oov.append(word)
            seen.add(word.lower())

    if oov:
        logger.info("Unknown words detected: %s", oov)
    else:
        logger.info("No OOV words detected; all input words are in vocabulary.")

    return oov


# --------------------------------------------------------------------------
# Layer 1: LLM (Ollama)
# --------------------------------------------------------------------------

def build_system_prompt() -> str:
    """Build the strict system prompt used for every LLM OOV resolution call.

    Includes a few-shot block to help the model replace the unknown word
    with simple English words or short phrases while preserving the
    original meaning.

    Returns:
        The fixed system prompt string.
    """
    return """You are an Out-Of-Vocabulary resolver for an American Sign Language translation system.

You will be given one unknown word and the full sentence context.

Instructions:
- Replace the unknown word with simple English words or a short phrase.
- Preserve the original meaning.
- Keep the sentence natural.
- Do not explain or apologize.
- Return only a JSON array of replacement words, nothing else.
- If no reasonable replacement exists, return an empty array: []
- Use as few words as possible (prefer 1, use 2-3 only if truly needed).

Examples of valid output style:
["sick"]
["eat","apple"]
["father"]
[]"""


def build_user_prompt(word: str, sentence: Optional[str] = None) -> str:
    """Build the dynamic user prompt for a single OOV word.

    Args:
        word: The unknown word to resolve.
        sentence: The full original sentence context, if available.

    Returns:
        A formatted prompt including the sentence and the target word.
    """
    sentence_block = f"Input sentence:\n{sentence}\n\n" if sentence else ""
    return f"{sentence_block}Unknown word:\n{word}\n\nReturn the JSON array now."


def _call_ollama(word: str, sentence: Optional[str] = None) -> Optional[str]:
    """Send a single OOV word to the local Ollama model and return its raw answer.

    Args:
        word: The unknown word to resolve.
        sentence: The full original sentence context, if available.

    Returns:
        The model's raw text response (stripped), or None if the package
        is missing, the connection fails, the model errors out, or any
        other unexpected exception occurs.
    """
    try:
        import ollama
    except ImportError:
        logger.error(
            "The 'ollama' package is not installed. Install it with: pip install ollama"
        )
        return None

    try:
        logger.info("LLM request: resolving OOV word '%s' with model '%s'.", word, OLLAMA_MODEL)
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": build_user_prompt(word, sentence)},
            ],
            options={"temperature": OLLAMA_TEMPERATURE},
        )
    except ConnectionError:
        logger.error("Connection refused: is the Ollama server running?")
        return None
    except Exception as exc:  # noqa: BLE001 - never crash the pipeline
        logger.error("Ollama call failed for word '%s': %s", word, exc)
        return None

    try:
        content = response["message"]["content"]
    except (KeyError, TypeError) as exc:
        logger.error("Unexpected Ollama response structure: %s", exc)
        return None

    logger.info("LLM response for '%s': %s", word, content)
    return content.strip() if content else None


def parse_response(raw_response: str) -> Optional[List[str]]:
    """Parse the model's raw text into a JSON list of strings.

    Handles minor formatting noise (code fences, stray whitespace) but
    does not attempt to repair fundamentally broken JSON.

    Args:
        raw_response: The raw text returned by the model.

    Returns:
        A list of strings on success, or None if parsing failed or the
        structure wasn't a JSON array.
    """
    if not raw_response:
        return None

    text = raw_response.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            logger.error("Parsing failed: no JSON array found in response.")
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.error("Parsing failed: could not decode extracted JSON array.")
            return None

    if not isinstance(parsed, list):
        logger.error("Parsing failed: response was not a JSON array.")
        return None

    if not all(isinstance(item, str) for item in parsed):
        logger.error("Parsing failed: response array contained non-string items.")
        return None

    return parsed


def validate_mapping(candidate_words: List[str]) -> Optional[List[str]]:
    """Validate a replacement list against basic format and size limits.

    Rejects the mapping if:
        - it is empty
        - it exceeds MAX_REPLACEMENT_WORDS
        - any element is not a non-empty string

    Used for LLM output and manual synonym entries to prevent malformed
    replacements from entering the pipeline.

    Args:
        candidate_words: The candidate list of replacement words.

    Returns:
        The validated, lowercase list of replacement words, or None if
        validation failed for any reason.
    """
    if not candidate_words:
        logger.warning("Validation failed: empty replacement list.")
        return None

    if len(candidate_words) > MAX_REPLACEMENT_WORDS:
        logger.warning(
            "Validation failed: replacement list too long (%d > %d).",
            len(candidate_words),
            MAX_REPLACEMENT_WORDS,
        )
        return None

    normalized = []
    for item in candidate_words:
        if not isinstance(item, str):
            logger.warning("Validation failed: replacement item is not a string: %r.", item)
            return None
        candidate = item.strip().lower()
        if not candidate:
            logger.warning("Validation failed: replacement list contains an empty string.")
            return None
        normalized.append(candidate)

    return normalized


def llm_fallback(word: str, sentence: Optional[str] = None) -> Optional[List[str]]:
    """Layer 1: attempt to resolve a single OOV word via the local LLM.

    Args:
        word: The unknown word to resolve.
        sentence: The full original sentence context, if available.

    Returns:
        A validated list of replacement words, or None if the LLM is
        unavailable, its response is malformed, or validation fails.
    """
    raw_response = _call_ollama(word, sentence)
    if raw_response is None:
        return None

    parsed = parse_response(raw_response)
    if parsed is None:
        return None

    return validate_mapping(parsed)


# --------------------------------------------------------------------------
# Layer 2: semantic similarity (embeddings, no LLM needed)
# --------------------------------------------------------------------------

_semantic_model = None
_semantic_model_load_attempted = False
_vocab_embeddings = None


def _get_semantic_model():
    """Lazily load and cache the sentence-transformers model.

    Returns:
        The loaded model instance, or None if the package isn't installed
        or the model failed to load for any reason (e.g. no internet on
        first run to download it).
    """
    global _semantic_model, _semantic_model_load_attempted

    if _semantic_model is not None:
        return _semantic_model
    if _semantic_model_load_attempted:
        # Already tried and failed this process; don't retry every call.
        return None

    _semantic_model_load_attempted = True

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning(
            "sentence-transformers not installed; semantic fallback layer disabled. "
            "Install with: pip install sentence-transformers"
        )
        return None

    try:
        _semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001 - never crash the pipeline
        logger.warning("Could not load semantic model '%s': %s", SEMANTIC_MODEL_NAME, exc)
        return None

    return _semantic_model


def _get_vocab_embeddings():
    """Lazily compute and cache embeddings for every known sign vocabulary word.

    Returns:
        A 2D array of embeddings (one row per TARGET_FOLDERS entry, same
        order), or None if the semantic model isn't available.
    """
    global _vocab_embeddings

    if _vocab_embeddings is not None:
        return _vocab_embeddings

    model = _get_semantic_model()
    if model is None:
        return None

    try:
        _vocab_embeddings = model.encode(TARGET_FOLDERS, normalize_embeddings=True)
    except Exception as exc:  # noqa: BLE001 - never crash the pipeline
        logger.warning("Failed to compute vocabulary embeddings: %s", exc)
        return None

    return _vocab_embeddings


def semantic_fallback(word: str) -> Optional[List[str]]:
    """Layer 2: find the closest semantically similar replacement word(s).

    Embeds the OOV word and compares it against known candidates,
    returning words with cosine similarity above
    SEMANTIC_SIMILARITY_THRESHOLD (up to SEMANTIC_TOP_K of them).

    Args:
        word: The unknown word to resolve.

    Returns:
        A list of one or more replacement words, or None if the semantic
        model is unavailable or no candidate is similar enough.
    """
    model = _get_semantic_model()
    vocab_embeddings = _get_vocab_embeddings()
    if model is None or vocab_embeddings is None:
        return None

    try:
        import numpy as np

        word_embedding = model.encode([word], normalize_embeddings=True)[0]
        similarities = vocab_embeddings @ word_embedding  # cosine sim (both normalized)

        ranked_indices = np.argsort(similarities)[::-1][:SEMANTIC_TOP_K]
        ranked: List[Tuple[str, float]] = [
            (TARGET_FOLDERS[i], float(similarities[i])) for i in ranked_indices
        ]
    except Exception as exc:  # noqa: BLE001 - never crash the pipeline
        logger.warning("Semantic fallback failed for '%s': %s", word, exc)
        return None

    logger.info("Semantic candidates for '%s': %s", word, ranked)

    if not ranked or ranked[0][1] < SEMANTIC_SIMILARITY_THRESHOLD:
        logger.info(
            "Semantic fallback: no match for '%s' above threshold %.2f.",
            word,
            SEMANTIC_SIMILARITY_THRESHOLD,
        )
        return None

    top_word, top_score = ranked[0]
    accepted = [top_word]

    # Only keep a second word if it's nearly as strong a match as the top
    # one - otherwise a weak runner-up (e.g. "candy" trailing "eat" for
    # "pizza") gets pulled in just because it scored second-highest.
    for w, score in ranked[1:]:
        if score >= SEMANTIC_SIMILARITY_THRESHOLD and (top_score - score) <= SEMANTIC_SECOND_MATCH_MARGIN:
            accepted.append(w)

    logger.info("Semantic fallback resolved '%s' -> %s", word, accepted)
    return accepted


# --------------------------------------------------------------------------
# Layer 3: manual synonym dictionary
# --------------------------------------------------------------------------

def dictionary_fallback(word: str) -> Optional[List[str]]:
    """Layer 3: look up a curated hand-written replacement.

    Args:
        word: The unknown word to resolve.

    Returns:
        A validated list of English replacement words, or None if there's
        no dictionary entry for this word (or the entry fails validation).
    """
    candidates = MANUAL_SYNONYMS.get(word.lower())
    if not candidates:
        return None

    validated = validate_mapping(candidates)
    if validated is None:
        logger.warning(
            "MANUAL_SYNONYMS entry for '%s' failed validation - check the dictionary for typos.",
            word,
        )
        return None

    logger.info("Dictionary fallback resolved '%s' -> %s", word, validated)
    return validated


# --------------------------------------------------------------------------
# Layered resolution for a single word
# --------------------------------------------------------------------------

def resolve_single_word(word: str, sentence: Optional[str] = None) -> Tuple[List[str], str]:
    """Resolve one OOV word by trying each layer in order.

    Order: LLM -> manual dictionary -> semantic similarity -> keep original.

    The curated dictionary runs before semantic similarity on purpose:
    a hand-checked entry is more trustworthy than a same-model embedding
    match on a single out-of-context word, which can produce coincidental
    matches (e.g. "hospital" ~ "egypt"). Semantic similarity is reserved
    for words nobody has curated a dictionary entry for yet.

    Args:
        word: The unknown word to resolve.

    Returns:
        A tuple of (replacement_words, layer_used), where layer_used is
        one of "llm", "dictionary", "semantic", or "unresolved".
    """
    llm_result = llm_fallback(word, sentence)
    if llm_result is not None:
        return llm_result, "llm"

    dictionary_result = dictionary_fallback(word)
    if dictionary_result is not None:
        return dictionary_result, "dictionary"

    semantic_result = semantic_fallback(word)
    if semantic_result is not None:
        return semantic_result, "semantic"

    logger.warning("All layers failed for '%s'; keeping original word.", word)
    return [word], "unresolved"


# --------------------------------------------------------------------------
# Replace OOV tokens in the original word list
# --------------------------------------------------------------------------

def replace_words(words: List[str], replacements: Dict[str, List[str]]) -> List[str]:
    """Rebuild the word list, substituting each OOV word with its mapping.

    In-vocabulary words are kept exactly as they appeared. OOV words with
    a resolved replacement are expanded in place, preserving position.

    Args:
        words: The original input word list.
        replacements: Maps each OOV word (original casing) to its list of
            replacement words.

    Returns:
        The final, position-preserving word list.
    """
    result: List[str] = []
    for word in words:
        if word.lower() in _VOCAB_SET_LOWER:
            result.append(word)
        elif word in replacements:
            result.extend(replacements[word])
        else:
            result.append(word)
    return result


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def _summarize_method(layers_used: List[str]) -> str:
    """Summarize the overall method from the set of layers actually used.

    Args:
        layers_used: The layer that resolved each OOV word.

    Returns:
        "llm_local", "semantic", "dictionary", "fallback" (all unresolved),
        or "mixed" (different words resolved by different layers).
    """
    unique_layers = set(layers_used)

    if unique_layers == {"llm"}:
        return "llm_local"
    if unique_layers == {"semantic"}:
        return "semantic"
    if unique_layers == {"dictionary"}:
        return "dictionary"
    if unique_layers == {"unresolved"}:
        return "fallback"
    return "mixed"


def process_oov(words: List[str]) -> OOVResult:
    """Resolve all out-of-vocabulary words in a gloss/word list.

    This is the only function the rest of the pipeline needs to call.

    Pipeline:
        1. Find which words are OOV (find_oov_words).
        2. For each distinct OOV word, try each layer in order until one
           succeeds (resolve_single_word): LLM -> semantic similarity ->
           manual dictionary -> keep the original word.
        3. Substitute resolved replacements back into the word list
           (replace_words), preserving position. Unresolved words are
           kept unchanged so the pipeline never crashes or silently
           drops signed content.

    Args:
        words: The input word list, e.g. ["I", "want", "sandwich"].

    Returns:
        An OOVResult containing the original words, the final mapped
        words, the overall method, confidence, warnings, a per-word
        replacement map, and a per-word resolution-layer map. The
        rewritten words are passed on to the Gloss Generator for later
        vocabulary-supported gloss conversion.
    """
    if not words:
        return OOVResult(
            original_words=words,
            mapped_words=[],
            method="no_oov",
            warnings=["Empty word list provided."],
            confidence=0.0,
            replacements={},
            resolution_layers={},
        )

    oov_words = find_oov_words(words)

    if not oov_words:
        return OOVResult(
            original_words=words,
            mapped_words=list(words),
            method="no_oov",
            warnings=[],
            confidence=1.0,
            replacements={},
            resolution_layers={},
        )

    warnings: List[str] = []
    replacements: Dict[str, List[str]] = {}
    resolution_layers: Dict[str, str] = {}

    sentence_context = " ".join(words)
    for word in oov_words:
        resolved_words, layer = resolve_single_word(word, sentence_context)
        replacements[word] = resolved_words
        resolution_layers[word] = layer

        if layer == "unresolved":
            warnings.append(f"Could not resolve '{word}' with any layer; keeping original word.")

    mapped_words = replace_words(words, replacements)

    method = _summarize_method(list(resolution_layers.values()))
    confidence = sum(CONFIDENCE_BY_LAYER[layer] for layer in resolution_layers.values()) / len(
        resolution_layers
    )

    return OOVResult(
        original_words=words,
        mapped_words=mapped_words,
        method=method,
        warnings=warnings,
        confidence=round(confidence, 3),
        replacements=replacements,
        resolution_layers=resolution_layers,
    )


# --------------------------------------------------------------------------
# Demonstration
# --------------------------------------------------------------------------

if __name__ == "__main__":
    examples = [
        ["I", "want", "sandwich"],
        ["hospital", "go"],
        ["pizza", "eat"],
        ["I", "want", "apple"],  # no OOV words at all
        ["mom", "and", "dad"],   # dictionary-layer example
    ]

    for word_list in examples:
        result = process_oov(word_list)
        print(result)
        print("-" * 60)
