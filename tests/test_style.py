"""Checks for the colors in ui/style.py: allergen groups and word marks (PRD D8, D10).

Run with plain Python:
    python tests/test_style.py
or with pytest:
    pytest tests/test_style.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.allergens import GROUPS  # noqa: E402
from ui import imaging  # noqa: E402
from ui.style import ALLERGEN_COLORS, CHECK_COLOR, FIXED_COLOR, allergen_slug  # noqa: E402

ASSETS = ROOT / "ui" / "assets"


def _luminance(color: str) -> float:
    channels = [int(color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_every_group_has_a_color():
    assert set(ALLERGEN_COLORS) == set(GROUPS)


def test_tag_text_is_readable():
    for group, (fill, ink, line) in ALLERGEN_COLORS.items():
        assert _contrast(fill, ink) >= 4.5, group  # WCAG AA for small text
        assert _contrast(line, "#FFFFFF") >= 3.0, group  # "may contain" outline on white


def test_slugs_are_css_safe_and_unique():
    slugs = [allergen_slug(group) for group in GROUPS]
    assert slugs[GROUPS.index("Wheat / gluten")] == "wheat-gluten"
    assert all(slug.replace("-", "").isalpha() for slug in slugs)
    assert len(set(slugs)) == len(slugs)


def test_mark_colors_are_readable_and_not_allergen_colors():
    allergen = {color.upper() for colors in ALLERGEN_COLORS.values() for color in colors}
    for color in (CHECK_COLOR, FIXED_COLOR):
        assert _contrast(color, "#FFFFFF") >= 4.5, color  # a 2px dotted line, and legend text
        assert color.upper() not in allergen, color
    assert CHECK_COLOR.upper() != FIXED_COLOR.upper()


def test_mark_colors_match_the_components_and_the_photo():
    for name in ("label_text.css", "ingredient_list.css"):
        css = (ASSETS / name).read_text(encoding="utf-8").lower()
        assert CHECK_COLOR.lower() in css and FIXED_COLOR.lower() in css, name
    to_bgr = lambda color: tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (4, 2, 0))  # noqa: E731
    assert imaging.CHECK_BGR == to_bgr(CHECK_COLOR)
    assert imaging.FIXED_BGR == to_bgr(FIXED_COLOR)


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} passed")
