"""Checks for the allergen colors in ui/style.py (PRD D8).

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
from ui.style import ALLERGEN_COLORS, allergen_slug  # noqa: E402


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


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} passed")
