"""Unit tests for src/correction.py (PRD P9).

Run with plain Python:
    python tests/test_correction.py
or with pytest:
    pytest tests/test_correction.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.correction import apply_suggestions, ocr_distance, suggest, suggest_for_words  # noqa: E402


def test_prd_examples():
    assert suggest("Vilater,", 0.40) == "Water,"
    assert suggest("NaCI,", 0.95) == "NaCl,"  # confident, but a look-alike letter
    assert suggest("tocopheroi).", 0.60) == "tocopherol)."


def test_known_words_and_non_words_are_left_alone():
    for word in ("Water,", "SUGAR", "milk", "(20%),", "E471", "L-carnitine", "and"):
        assert suggest(word, 0.30) is None, word


def test_confident_words_only_get_look_alike_fixes():
    assert suggest("Tomatu", 0.99) is None  # one real edit away, but the model was sure
    assert suggest("Tomatu", 0.50) == "Tomato"


def test_case_follows_the_word_as_read():
    assert suggest("Vilater", 0.30) == "Water"
    assert suggest("vilater", 0.30) == "water"
    assert suggest("VILATER", 0.30) == "WATER"
    assert suggest("SALf", 0.30) == "Salt"  # mixed case: capitalised like the first letter


def test_distance_prefers_ocr_slips():
    assert ocr_distance("naci", "nacl") == 0.3  # I and l look alike
    assert ocr_distance("corn", "com") == 0.4  # "rn" read as "m"
    assert ocr_distance("water", "water") == 0.0
    assert ocr_distance("abcdef", "uvwxyz", limit=1.0) == float("inf")


@dataclass
class FakeWord:
    text: str
    confidence: float


def test_suggestions_keep_the_word_positions():
    words = [FakeWord("INGREDIENTS:", 0.99), FakeWord("Vilater,", 0.40), FakeWord("Sugar,", 0.99), FakeWord("NaCI", 0.97)]
    suggestions = suggest_for_words(words)
    assert [(s.index, s.original, s.replacement) for s in suggestions] == [(1, "Vilater,", "Water,"), (3, "NaCI", "NaCl")]


def test_apply_suggestions_skips_the_undone_ones():
    words = [FakeWord("INGREDIENTS:", 0.99), FakeWord("Vilater,", 0.40), FakeWord("Sugar,", 0.99), FakeWord("NaCI", 0.97)]
    suggestions = suggest_for_words(words)
    assert [word.text for word in apply_suggestions(words, suggestions)] == ["INGREDIENTS:", "Water,", "Sugar,", "NaCl"]
    corrected = apply_suggestions(words, suggestions, skip={3})
    assert [word.text for word in corrected] == ["INGREDIENTS:", "Water,", "Sugar,", "NaCI"]
    assert corrected[1].confidence == 0.40  # only the text changes
    assert words[1].text == "Vilater,"  # the words as read are left alone


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} passed")
