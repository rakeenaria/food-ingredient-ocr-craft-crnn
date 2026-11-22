"""Unit tests for src/allergens.py (PRD P8).

Run with plain Python:
    python tests/test_allergens.py
or with pytest:
    pytest tests/test_allergens.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.allergens import find_allergens, find_in_list, groups_in, is_precautionary, summarize  # noqa: E402


def test_each_group_is_found():
    text = (
        "Skim milk, whole eggs, roasted peanuts, cashew nuts, soya lecithin, wheat flour, anchovy extract, "
        "dried shrimp, sesame seeds"
    )
    assert groups_in(text) == [
        "Milk", "Egg", "Peanut", "Tree nuts", "Soy", "Wheat / gluten", "Fish", "Shellfish", "Sesame",
    ]


def test_derivatives_and_regional_names():
    assert groups_in("whey powder, sodium caseinate, ghee") == ["Milk"]
    assert groups_in("Maida, atta, suji") == ["Wheat / gluten"]
    assert groups_in("Gingelly oil") == ["Sesame"]
    assert groups_in("Kesar Badam") == ["Tree nuts"]
    assert groups_in("barley malt extract, rolled oats") == ["Wheat / gluten"]


def test_whole_words_only():
    assert groups_in("buckwheat, nutmeg, coconut, doughnut, butternut, eggplant, maltodextrin") == []
    assert groups_in("shellfish") == ["Shellfish"]  # not also Fish
    assert groups_in("Whole-wheat flour") == ["Wheat / gluten"]


def test_other_foods_named_like_dairy():
    assert groups_in("coconut milk, cocoa butter, cream of tartar") == []
    assert groups_in("peanut butter") == ["Peanut"]
    assert groups_in("ice cream, milk chocolate") == ["Milk"]


def test_negations_are_ignored():
    assert groups_in("gluten-free oats") == ["Wheat / gluten"]  # oats remain, the gluten claim is not a hit
    assert [hit.term for hit in find_allergens("gluten-free oats")] == ["oats"]
    assert groups_in("Free from milk, egg and nuts.") == []
    assert groups_in("Dairy free. Non-dairy creamer. No added milk.") == []
    assert groups_in("Contains no artificial colours, milk powder") == ["Milk"]
    assert groups_in("Made without artificial flavours, contains milk") == ["Milk"]


def test_precautionary_statements():
    assert is_precautionary("May contain traces of peanuts")
    assert is_precautionary("Manufactured on Shared Equipment that Processes Peanuts, Tree Nuts, Milk")
    assert is_precautionary("Made in a facility that also processes soy")
    assert not is_precautionary("Contains: Wheat, Milk")


def test_summary_splits_contains_and_may_contain():
    summary = summarize(
        ["Water", "milk (fresh milk (20%), milk powder)", "sugar", "Emulsifier (Soya Lecithin)"],
        ["Contains: Milk, Soy", "May contain traces of peanuts and milk"],
    )
    assert list(summary.contains) == ["Milk", "Soy"]
    assert list(summary.may_contain) == ["Peanut"]  # milk is already contained
    assert [term.lower() for term in summary.contains["Milk"]] == ["milk"]  # spellings are merged


def test_thesis_ground_truth_label():
    # data/valid/gt10.txt, bahan_1 (macaroni and cheese)
    items = [
        "ENRICHED MACARONI PRODUCT (WHEAT FLOUR, NIACIN, FERROUS SULFATE [IRON], FOLIC ACID)",
        "CHEESE SAUCE (MILK, WHEY, WATER, CANOLA OIL, SALT, ENZYMES)",
    ]
    summary = summarize(items, ["CONTAINS: WHEAT, MILK"])
    assert list(summary.contains) == ["Milk", "Wheat / gluten"]
    assert summary.may_contain == {}


def test_find_in_list_places_words_and_marks_may_contain():
    text = "Ingredients: sugar, whey powder. Contains: soy. May contain traces of nuts."
    piece = lambda words: (text.index(words), text.index(words) + len(words))  # noqa: E731
    hits = find_in_list(
        text,
        [piece("sugar"), piece("whey powder")],
        [piece("Contains: soy"), piece("May contain traces of nuts")],
    )
    assert [(text[hit.start : hit.end], hit.group, hit.precaution) for hit in hits] == [
        ("whey", "Milk", False),
        ("soy", "Soy", False),
        ("nuts", "Tree nuts", True),
    ]


def test_group_names_are_found_whole():
    # The text box colors the whole name, as on the label: "Tree Nuts", not just "Nuts".
    assert [hit.term for hit in find_allergens("Processes Peanuts, Tree Nuts, Milk")] == ["Peanuts", "Tree Nuts", "Milk"]


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} passed")
