"""Suggest fixes for words the OCR probably misread (PRD P9). No UI dependencies.

A word is checked only when its letters are not in the ingredient vocabulary
(src/data/ingredient_vocabulary.txt, built by scripts/build_vocabulary.py from
the thesis train and validation labels plus common ingredient terms). The
closest vocabulary word by an OCR-aware edit distance is suggested when it is
close enough: within one or two edits for a low-confidence word, but only a
look-alike letter (I/l, O/0, rn/m...) for a word the model was sure about.
Nothing here changes the OCR text; the caller decides what to apply.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

VOCABULARY_PATH = Path(__file__).with_name("data") / "ingredient_vocabulary.txt"
LOW_CONFIDENCE = 0.80  # same threshold as the highlighted words

# Letters OCR confuses (lower case). A swap inside a group costs 0.3 instead of 1.
_LOOK_ALIKE_GROUPS = ("il1|!", "o0", "s5", "z2", "b8", "g9", "ec", "uv", "nh")
_LOOK_ALIKE = {(a, b) for group in _LOOK_ALIKE_GROUPS for a in group for b in group if a != b}
# Two letters read for one ("rn" for "m"), or one for two. Cost 0.4.
_SPLIT = {("rn", "m"), ("vv", "w"), ("vi", "w"), ("vl", "w"), ("cl", "d"), ("ii", "u"), ("ri", "n"), ("li", "h")}
_MERGE = {(single, pair) for pair, single in _SPLIT}
LOOK_ALIKE_COST = 0.3
SPLIT_COST = 0.4

_WORD = re.compile(r"^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$")  # punctuation, letters, punctuation


class WordLike(Protocol):
    text: str
    confidence: float


@dataclass(frozen=True)
class Suggestion:
    index: int  # position in the sequence of words that was checked
    original: str  # the word as read, for example "Vilater,"
    replacement: str  # the suggested word, for example "Water,"


@dataclass(frozen=True)
class _Vocabulary:
    spelling: dict[str, str]  # lower-case key -> spelling to show (NaCl keeps its capitals)
    count: dict[str, int]
    by_length: dict[int, tuple[str, ...]]


@lru_cache(maxsize=1)
def _vocabulary(path: Path = VOCABULARY_PATH) -> _Vocabulary:
    spelling, count, by_length = {}, {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        word, _, number = line.partition("\t")
        if not word:
            continue
        key = word.lower()
        spelling[key] = word
        count[key] = int(number or 1)
        by_length.setdefault(len(key), []).append(key)
    return _Vocabulary(spelling, count, {length: tuple(words) for length, words in by_length.items()})


def ocr_distance(read: str, truth: str, limit: float = float("inf")) -> float:
    """Edit distance from an OCR reading to a word, cheaper for typical OCR slips.

    Both strings are lower case. Returns ``inf`` as soon as ``limit`` is exceeded.
    """
    n, m = len(read), len(truth)
    if abs(n - m) * SPLIT_COST > limit:
        return float("inf")
    before = None
    previous = [float(j) for j in range(m + 1)]
    for i in range(1, n + 1):
        current = [float(i)] + [0.0] * m
        a = read[i - 1]
        for j in range(1, m + 1):
            b = truth[j - 1]
            if a == b:
                substitution = 0.0
            elif (a, b) in _LOOK_ALIKE:
                substitution = LOOK_ALIKE_COST
            else:
                substitution = 1.0
            best = min(previous[j] + 1.0, current[j - 1] + 1.0, previous[j - 1] + substitution)
            if i >= 2 and (read[i - 2 : i], b) in _SPLIT:
                best = min(best, before[j - 1] + SPLIT_COST)
            if j >= 2 and (a, truth[j - 2 : j]) in _MERGE:
                best = min(best, previous[j - 2] + SPLIT_COST)
            current[j] = best
        if min(current) > limit:
            return float("inf")
        before, previous = previous, current
    return previous[m]


def _limit(length: int, confidence: float) -> float:
    if confidence >= LOW_CONFIDENCE:
        return 0.6 if length >= 3 else 0.0  # look-alike letters only
    if length <= 3:
        return 0.6
    if length <= 5:
        return 1.0
    if length <= 8:
        return 1.5
    return 2.0


def _match_case(word: str, template: str, spelling: str) -> str:
    if any(char.isupper() for char in spelling[1:]):
        return spelling  # a fixed spelling such as NaCl
    if template.isupper() and len(template) > 1:
        return word.upper()
    if template[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def _best_match(core: str, confidence: float, vocabulary: _Vocabulary) -> str | None:
    key = core.lower()
    limit = _limit(len(key), confidence)
    if limit <= 0:
        return None
    best: tuple[float, int, str] | None = None
    for length in range(len(key) - 2, len(key) + 3):
        for candidate in vocabulary.by_length.get(length, ()):
            distance = ocr_distance(key, candidate, limit)
            if distance <= limit:
                rank = (distance, -vocabulary.count[candidate], candidate)
                if best is None or rank < best:
                    best = rank
    return best[2] if best else None


def suggest(text: str, confidence: float = 0.0) -> str | None:
    """The corrected form of one OCR word, or None when it looks right.

    Surrounding punctuation is kept ("Vilater," -> "Water,"). Words with digits
    or inner hyphens, and words already in the vocabulary, are left alone.
    """
    match = _WORD.match(text)
    if not match:
        return None
    prefix, core, suffix = match.groups()
    vocabulary = _vocabulary()
    if core.lower() in vocabulary.count:
        return None
    word = _best_match(core, confidence, vocabulary)
    if word is None:
        return None
    fixed = _match_case(word, core, vocabulary.spelling[word])
    return None if fixed == core else f"{prefix}{fixed}{suffix}"


def suggest_for_words(words: Sequence[WordLike]) -> list[Suggestion]:
    """Suggestions for every word that probably was misread, in reading order."""
    suggestions = []
    for index, word in enumerate(words):
        replacement = suggest(word.text, word.confidence)
        if replacement is not None:
            suggestions.append(Suggestion(index, word.text, replacement))
    return suggestions
