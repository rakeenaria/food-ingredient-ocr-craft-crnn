"""Text helpers on top of OCR output. No UI dependencies."""

from __future__ import annotations


def clean_line(line: str) -> str:
    """Collapse runs of whitespace (empty recognitions leave double spaces)."""
    return " ".join(line.split())


def to_lines_text(lines: list[str]) -> str:
    """Text with the label's original line breaks."""
    return "\n".join(cleaned for cleaned in (clean_line(line) for line in lines) if cleaned)


def to_paragraph(lines: list[str]) -> str:
    """One paragraph: lines joined with single spaces."""
    return " ".join(cleaned for cleaned in (clean_line(line) for line in lines) if cleaned)
