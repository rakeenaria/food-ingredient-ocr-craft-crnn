"""Unit tests for src/postprocess.py (no model weights needed).

Run with plain Python:
    python tests/test_postprocess.py
or with pytest:
    pytest tests/test_postprocess.py
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocess import (  # noqa: E402
    compose_text,
    join_wrapped_lines,
    parse_ingredients,
    to_lines_text,
    to_paragraph,
)

FIXTURE = ROOT / "tests" / "fixtures" / "baseline_gambar_simulasi.txt"


def items(text: str) -> list[str]:
    return [item.text for item in parse_ingredients(text).items]


def notes(text: str) -> list[str]:
    return [note.text for note in parse_ingredients(text).notes]


# ------------------------------------------------------------------ PRD P5
def test_prd_acceptance_nested_parentheses():
    assert items("milk (fresh milk (20%), milk powder), sugar") == ["milk (fresh milk (20%), milk powder)", "sugar"]


def test_heading_is_removed():
    for text in (
        "Ingredients: Water, Sugar.",
        "INGREDIENTS: Water, Sugar",
        "INGREDIENTS Water, Sugar",
        "ingredients:Water, Sugar",
        "INCREDIENTS: Water, Sugar",  # frequent OCR confusion
        "lngredients: Water, Sugar",
        "LOW FAT YOGHURT INGREDIENTS: Water, Sugar",
    ):
        result = parse_ingredients(text)
        assert [item.text for item in result.items] == ["Water", "Sugar"], text
        assert result.heading is not None, text


def test_word_ingredients_inside_a_note_is_not_a_heading():
    result = parse_ingredients("Water, salt. Allergy advice: see ingredients in bold.")
    assert result.heading is None
    assert [item.text for item in result.items] == ["Water", "salt"]
    assert [note.text for note in result.notes] == ["Allergy advice: see ingredients in bold"]


def test_heading_read_out_of_order_keeps_the_ingredients_before_it():
    assert items("Cranberries, Sugar, Sunflower Oil Ingredients:") == ["Cranberries", "Sugar", "Sunflower Oil"]
    assert items("Cheese, Milk Solids, Ingredients: Salt, Vinegar") == ["Cheese", "Milk Solids", "Salt", "Vinegar"]
    # ...but a nutrition panel or a product description before the heading is not.
    assert items("Nutrition Facts Energy, kcal 557, Fat, g 33 INGREDIENTS: Water, Sugar") == ["Water", "Sugar"]
    assert items("VANILLA ICE CREAM. TIMELESS VANILLA ICE CREAM INGREDIENTS: WATER, SUGAR") == ["WATER", "SUGAR"]


def test_single_ingredient_product():
    assert items("Ingredients: Durum Wheat Semolina") == ["Durum Wheat Semolina"]


def test_ingredient_names_that_look_like_statements():
    assert items("Ingredients: Cashews, Nutritional Yeast, Salt") == ["Cashews", "Nutritional Yeast", "Salt"]
    assert items("Ingredients: Pasta made with durum wheat, Salt") == ["Pasta made with durum wheat", "Salt"]


def test_statements_inside_a_piece_end_the_list():
    text = "Salt, Antioxidant (INS 320) CONTAINS ADDED FLAVOURS (NATURE IDENTICAL) CONTAINS GLUTEN, ALMONDS. MAY CONTAIN SOY."
    assert items(text) == ["Salt", "Antioxidant (INS 320)"]
    assert notes(text) == ["CONTAINS ADDED FLAVOURS (NATURE IDENTICAL) CONTAINS GLUTEN, ALMONDS", "MAY CONTAIN SOY"]
    text = "INGREDIENTS Gingelly Oil NUTRITION FACTS (Approximate) Per 100 Grams"
    assert items(text) == ["Gingelly Oil"]
    assert notes(text) == ["NUTRITION FACTS (Approximate) Per 100 Grams"]


def test_square_and_curly_brackets_nest():
    text = "Rolled Oats (25.1%), Candied Fruit (24%) {Cranberry (14%), Almonds [10%]}, Sugar"
    assert items(text) == ["Rolled Oats (25.1%)", "Candied Fruit (24%) {Cranberry (14%), Almonds [10%]}", "Sugar"]


def test_unbalanced_brackets_do_not_swallow_the_list():
    assert items("Rolled Oats 14%), Almond (5%), Sugar") == ["Rolled Oats 14%)", "Almond (5%)", "Sugar"]
    assert items("Chocolate (sugar, cocoa butter, Salt") == ["Chocolate (sugar", "cocoa butter", "Salt"]
    assert items("Raising Agents [500(ii), 503(i)), 450(i)], Salt") == ["Raising Agents [500(ii), 503(i)), 450(i)]", "Salt"]


def test_semicolons_split_and_contains_statement_becomes_a_note():
    text = "INGREDIENTS: PASTA (WHEAT FLOUR, NIACIN); CHEESE SAUCE (MILK, SALT). CONTAINS: WHEAT, MILK."
    result = parse_ingredients(text)
    assert [item.text for item in result.items] == ["PASTA (WHEAT FLOUR, NIACIN)", "CHEESE SAUCE (MILK, SALT)"]
    assert [note.text for note in result.notes] == ["CONTAINS: WHEAT, MILK"]


def test_contains_less_than_stays_in_the_list():
    text = "Carrots, Peas, Contains 2% or less of: Celery, Salt. Contains: Milk."
    assert items(text) == ["Carrots", "Peas", "Contains 2% or less of: Celery", "Salt"]
    assert notes(text) == ["Contains: Milk"]


def test_several_notes_are_split_into_sentences():
    text = "Ingredients: Sugar, Flour. Contains: Wheat. May contain traces of peanuts and other nuts."
    assert items(text) == ["Sugar", "Flour"]
    assert notes(text) == ["Contains: Wheat", "May contain traces of peanuts and other nuts"]


def test_note_without_full_stop_is_still_found():
    assert items("Sugar, Salt Contains: Wheat") == ["Sugar", "Salt"]
    assert notes("Sugar, Salt Contains: Wheat") == ["Contains: Wheat"]


def test_full_stop_read_instead_of_comma_still_separates():
    assert items("Water. Sugar, Salt") == ["Water", "Sugar", "Salt"]


def test_numbers_are_not_split():
    assert items("Honey 0. 2%, Vitamin A 1,000 IU, Salt") == ["Honey 0. 2%", "Vitamin A 1,000 IU", "Salt"]
    assert items("Green Tea 3.95%, Vit. C, Salt") == ["Green Tea 3.95%", "Vit. C", "Salt"]


def test_empty_and_heading_only_text():
    assert items("") == []
    assert items("Ingredients:") == []
    assert items(" , ; . ") == []


def test_spans_point_into_the_source_text():
    text = "INGREDIENTS: Water, milk (fresh milk (20%), milk powder) 30%, sugar. Contains: milk."
    result = parse_ingredients(text)
    for span in [*result.items, *result.notes]:
        assert text[span.start : span.end] == span.text


# --------------------------------------------------------- samples (OCR text)
def read_baseline() -> dict[str, str]:
    content = FIXTURE.read_text(encoding="utf-8").strip("\n")
    blocks = re.split(r"\n\n(?=[^\n\t]+\t)", content)
    return dict(block.split("\t", 1) for block in blocks)


def test_sample_labels_split_as_expected():
    baseline = read_baseline()
    paragraphs = {name: to_paragraph(text.splitlines()) for name, text in baseline.items()}
    assert [len(items(paragraphs[name])) for name in ("01", "02", "03", "04")] == [12, 8, 13, 6]
    assert items(paragraphs["02"]) == [
        "Water",
        "milk (fresh milk (20%), full cream milk powder, milk solid, skimmed milk powder) 30%",
        "sugar",
        "cocoa powder (1 %)",
        "stabilizer (vegetable base, trisodium citrate)",
        "vitamin premix",
        "almond powder (0.01%)",
        "synthetic flavor",
    ]
    assert items(paragraphs["05"]) == ["Cocoa Powder", "Synthetic Flavour"]
    assert notes(paragraphs["05"]) == [
        "Manufactured on Shared Equipment that Processes Peanuts, Tree Nuts, Milk, Soy and Wheat"
    ]


# ----------------------------------------------------------- line handling
def test_join_wrapped_lines_keeps_hyphenated_words_together():
    assert join_wrapped_lines("Expeller-\npressed oil,\nsalt") == "Expeller-pressed oil, salt"
    assert join_wrapped_lines("Mono-\n& diglycerides") == "Mono- & diglycerides"
    assert join_wrapped_lines("  Water,\n\n  sugar  ") == "Water, sugar"


@dataclass
class FakeWord:
    text: str
    line: int
    position: int


def test_compose_text_matches_line_and_paragraph_text():
    words = [
        FakeWord("sugar,", 2, 1),
        FakeWord("Expeller-", 1, 2),
        FakeWord("Ingredients:", 1, 1),
        FakeWord("", 2, 2),  # empty recognition
        FakeWord("pressed", 2, 3),
        FakeWord("oil.", 3, 1),
    ]
    lines = ["Ingredients: Expeller-", "sugar,  pressed", "oil."]
    as_lines = compose_text(words, as_lines=True)
    paragraph = compose_text(words)
    assert as_lines.text == to_lines_text(lines)
    assert paragraph.text == to_paragraph(lines)
    for composed in (as_lines, paragraph):
        for word, span in zip(words, composed.spans):
            if word.text:
                assert composed.text[span[0] : span[1]] == word.text
            else:
                assert span is None


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} passed")
