"""Text helpers on top of OCR output. No UI dependencies."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

# A line ending in "letter-" that continues with a letter was wrapped inside a
# hyphenated word ("Expeller-" / "pressed"): keep the hyphen, drop the break.
_HYPHEN_WRAP = re.compile(r"(?<=[^\W\d_]-)[ \t]*\r?\n\s*(?=[^\W\d_])")


def clean_line(line: str) -> str:
    """Collapse runs of whitespace (empty recognitions leave double spaces)."""
    return " ".join(line.split())


def to_lines_text(lines: list[str]) -> str:
    """Text with the label's original line breaks."""
    return "\n".join(cleaned for cleaned in (clean_line(line) for line in lines) if cleaned)


def join_wrapped_lines(text: str) -> str:
    """One paragraph: line breaks become spaces, except inside hyphenated words."""
    return " ".join(_HYPHEN_WRAP.sub("", text).split())


def to_paragraph(lines: list[str]) -> str:
    """The label's lines as one paragraph (see ``join_wrapped_lines``)."""
    return join_wrapped_lines(to_lines_text(lines))


# ------------------------------------------------------------ composed text
class WordLike(Protocol):
    text: str
    line: int
    position: int


@dataclass(frozen=True)
class ComposedText:
    """Reading-order text plus the place each input word landed in it."""

    text: str
    spans: tuple[tuple[int, int] | None, ...]  # per input word; None for empty words


def _is_hyphen_wrap(previous: str, following: str) -> bool:
    return len(previous) >= 2 and previous[-1] == "-" and previous[-2].isalpha() and following[:1].isalpha()


def compose_text(words: Sequence[WordLike], *, as_lines: bool = False) -> ComposedText:
    """Build the text from words in reading order, remembering each word's span.

    The result equals ``to_lines_text`` (``as_lines=True``) or ``to_paragraph``
    of the engine's lines, so highlights can be placed on exactly the text
    that is shown, copied and parsed.
    """
    order = sorted(range(len(words)), key=lambda index: (words[index].line, words[index].position))
    spans: list[tuple[int, int] | None] = [None] * len(words)
    parts: list[str] = []
    length = 0
    previous_line: int | None = None
    previous_token = ""
    for index in order:
        token = clean_line(words[index].text)
        if not token:
            continue
        if parts:
            if words[index].line == previous_line:
                separator = " "
            elif as_lines:
                separator = "\n"
            elif _is_hyphen_wrap(previous_token, token):
                separator = ""
            else:
                separator = " "
            parts.append(separator)
            length += len(separator)
        spans[index] = (length, length + len(token))
        parts.append(token)
        length += len(token)
        previous_line = words[index].line
        previous_token = token
    return ComposedText("".join(parts), tuple(spans))


def remap_spans(spans: Sequence[tuple[int, int]], source: str, target: str) -> list[tuple[int, int] | None]:
    """Move character ranges of ``source`` onto ``target``, a copy that differs only in whitespace.

    The paragraph that is parsed and the text that is shown can differ in line
    breaks and spaces (the label's lines, or the user's own edit). Ranges are
    matched through their first and last non-space characters. A range that
    starts or ends on whitespace gives None, and so does every range when the
    two texts differ in more than whitespace.
    """
    source_chars = [index for index, char in enumerate(source) if not char.isspace()]
    target_chars = [index for index, char in enumerate(target) if not char.isspace()]
    if len(source_chars) != len(target_chars) or any(
        source[i] != target[j] for i, j in zip(source_chars, target_chars)
    ):
        return [None] * len(spans)
    rank = [-1] * len(source)
    for number, index in enumerate(source_chars):
        rank[index] = number
    moved: list[tuple[int, int] | None] = []
    for start, end in spans:
        if not 0 <= start < end <= len(source) or rank[start] < 0 or rank[end - 1] < 0:
            moved.append(None)
            continue
        moved.append((target_chars[rank[start]], target_chars[rank[end - 1]] + 1))
    return moved


# ---------------------------------------------------------- ingredient list
@dataclass(frozen=True)
class Span:
    """A piece of the parsed text; ``text[start:end]`` for single-spaced input."""

    start: int
    end: int
    text: str


@dataclass(frozen=True)
class IngredientList:
    items: list[Span]  # the ingredients, in label order
    notes: list[Span]  # statements after the list: "Contains: milk", "May contain nuts"
    heading: Span | None  # the "Ingredients:" heading, if one was found


_OPENERS = "([{"
_CLOSER_TO_OPENER = {")": "(", "]": "[", "}": "{"}

# "Ingredients" plus frequent OCR confusions (lNGREDIENTS, INCREDIENTS, Ingredlents).
_HEADING = re.compile(
    r"(?<![^\W\d_])(?P<word>[il1|]?n[gc]r[eé]d[il1|]ents?)(?![^\W\d_])(?P<colon>\s*[:;\-–—])?",
    re.IGNORECASE,
)

# Statements that end the ingredient list when a piece starts with them.
_NOTE_START = re.compile(
    r"""(?:
        contains?\b(?!\s*(?:less|not|up\s+to|one|max|min|\d|two))
      | may\s+(?:also\s+)?contain
      | allerg
      | (?:manufactured|produced|processed|packed|packaged|made|prepared)\s+(?:on|in|at|by|under)\b
      | (?:distributed|imported|marketed)\s+by
      | this\s+product
      | (?:store|keep)\s
      | best\s+before | use\s+by | expiry | net\s+(?:wt|weight|content)
      | nutrition(?:al)?\s+(?:facts|information|value|panel) | directions | instructions | warning | caution
      | no\s+(?:artificial|added|preservative)
      | suitable\s+for
      | for\s+allergens
      | free\s+from
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Strong statements that end the list even in the middle of a piece
# (OCR often loses the full stop before them).
_NOTE_INLINE = re.compile(
    r"""(?<=\s)(?:
        may\s+(?:also\s+)?contain
      | contains?\s*:
      | contains?\s+(?:added|permitted)\b
      | allergen | allergy\s+advice
      | (?:manufactured|produced|processed|packed)\s+(?:on|in|at|by)\b
      | nutrition(?:al)?\s+(?:facts|information|value)
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_ABBREVIATIONS = {"vit", "approx", "incl", "no", "nos", "sp", "spp", "var", "wt", "conc", "min", "max", "st"}
_TRIM = " \t\r\n,;:.!?-–—•·"


def _find_heading(text: str) -> re.Match[str] | None:
    """The first plausible "Ingredients" heading.

    A heading needs a colon, or must be upper case or open the text and sit
    near the top. This keeps "see ingredients in bold" from counting.
    """
    limit = max(80, int(len(text) * 0.3))
    for match in _HEADING.finditer(text):
        word = match.group("word")
        letters = [char for char in word if char.isalpha()]
        mostly_upper = sum(char.isupper() for char in letters) >= 0.8 * len(letters)
        at_start = not text[: match.start()].strip()
        if match.group("colon") or (match.start() <= limit and (mostly_upper or at_start)):
            return match
    return None


def _bracket_depth(text: str, start: int, end: int) -> list[int]:
    """Nesting depth of each character in ``text[start:end]``.

    Unmatched brackets (common in OCR output) are treated as plain characters,
    so one missing ")" does not swallow the rest of the list.
    """
    stack: list[tuple[str, int]] = []
    pairs: list[tuple[int, int]] = []
    for index in range(start, end):
        char = text[index]
        if char in _OPENERS:
            stack.append((char, index))
        elif char in _CLOSER_TO_OPENER:
            wanted = _CLOSER_TO_OPENER[char]
            for level in range(len(stack) - 1, -1, -1):
                if stack[level][0] == wanted:
                    pairs.append((stack[level][1], index))
                    del stack[level:]  # openers above it never closed
                    break
    change = [0] * (end - start + 1)
    for open_index, close_index in pairs:
        change[open_index - start + 1] += 1
        change[close_index - start] -= 1
    depth, level = [], 0
    for offset in range(end - start):
        level += change[offset]
        depth.append(level)
    return depth


def _is_sentence_end(text: str, index: int) -> bool:
    following = text[index + 1 : index + 2]
    if following and not following.isspace():
        return False  # "2.5", "e.g", "Wheat.Contains"
    rest = text[index + 1 :].lstrip()
    if rest and not (rest[0].isalpha() or rest[0] in "([{\"'"):
        return False  # "0. 2%": OCR put a space inside a number
    if text[index] == ".":
        word = re.search(r"([^\W\d_]+)$", text[:index])
        if word and word.group(1).lower() in _ABBREVIATIONS:
            return False
    return True


def _is_digit_group(text: str, index: int) -> bool:
    return 0 < index < len(text) - 1 and text[index - 1].isdigit() and text[index + 1].isdigit()


def _trimmed(text: str, start: int, end: int) -> Span | None:
    while start < end and text[start] in _TRIM:
        start += 1
    while end > start and text[end - 1] in _TRIM:
        end -= 1
    piece = " ".join(text[start:end].split())
    if not any(char.isalnum() for char in piece):
        return None
    return Span(start, end, piece)


def _split(text: str, start: int, end: int) -> tuple[list[Span], list[int]]:
    """Top-level pieces of ``text[start:end]`` (trimmed, non-empty) and its depth map."""
    depth = _bracket_depth(text, start, end)
    pieces: list[Span] = []
    piece_start = start
    for index in range(start, end + 1):
        if index < end:
            if depth[index - start]:
                continue
            char = text[index]
            breaks = (char in ",;" and not _is_digit_group(text, index)) or (
                char in ".!?" and _is_sentence_end(text, index)
            )
            if not breaks:
                continue
        piece = _trimmed(text, piece_start, index)
        if piece:
            pieces.append(piece)
        piece_start = index + 1
    return pieces, depth


def _note_start(text: str, piece: Span, depth: list[int], offset: int) -> int | None:
    """Where a statement such as "Contains: milk" begins inside ``piece``, if anywhere."""
    if _NOTE_START.match(piece.text):
        return piece.start
    for match in _NOTE_INLINE.finditer(text, piece.start, piece.end):
        if depth[match.start() - offset] == 0:
            return match.start()
    return None


def parse_ingredients(text: str) -> IngredientList:
    """Split an ingredient statement into items and trailing notes.

    Items are separated by commas and semicolons outside brackets, so
    ``milk (fresh milk (20%), milk powder), sugar`` gives two items. A full
    stop also separates items (OCR often reads "," as "."); the list ends at
    the first statement such as "Contains: ..." or "May contain ...".
    Pass one paragraph (see ``join_wrapped_lines``).
    """
    end = len(text)
    items: list[Span] = []
    heading = None
    body_start = 0
    heading_match = _find_heading(text)
    if heading_match:
        heading = Span(heading_match.start(), heading_match.end(), text[heading_match.start() : heading_match.end()].strip())
        body_start = heading_match.end()
        # Text before the heading is usually a product name ("LOW FAT YOGHURT").
        # A comma-separated list there means the heading was read out of order
        # (two-column labels, or a heading printed below the list): keep it.
        preamble, preamble_depth = _split(text, 0, heading_match.start())
        is_list = any(text[index] in ",;" and not preamble_depth[index] for index in range(heading_match.start()))
        if is_list and all(_note_start(text, piece, preamble_depth, 0) is None for piece in preamble):
            items.extend(preamble)

    pieces, depth = _split(text, body_start, end)
    notes_start: int | None = None
    for piece in pieces:
        note_start = _note_start(text, piece, depth, body_start)
        if note_start is None:
            items.append(piece)
            continue
        head = _trimmed(text, piece.start, note_start)
        if head:
            items.append(head)
        notes_start = note_start
        break

    notes: list[Span] = []
    if notes_start is not None:
        sentence_start = notes_start
        for index in range(notes_start, end):
            if depth[index - body_start] == 0 and text[index] in ".!?" and _is_sentence_end(text, index):
                note = _trimmed(text, sentence_start, index)
                if note:
                    notes.append(note)
                sentence_start = index + 1
        note = _trimmed(text, sentence_start, end)
        if note:
            notes.append(note)
    return IngredientList(items=items, notes=notes, heading=heading)
