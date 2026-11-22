"""Find common allergens in ingredient text (PRD P8). No UI dependencies.

The word lists cover the nine groups in the PRD. Matching is by whole words,
so "buckwheat" is not wheat and "nutmeg" is not a nut. Phrases that name a
different food ("coconut milk", "cocoa butter", "cream of tartar") and
statements such as "gluten-free" or "free from milk" are ignored.

The gluten group follows the Codex list of cereals containing gluten (wheat,
rye, barley, oats, spelt and their malts and flours). Shellfish covers both
crustaceans and molluscs.

This is a reading aid, not a safety check: it misses names the lists do not
know and words the OCR misread.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

# Display order of the groups.
GROUPS = ("Milk", "Egg", "Peanut", "Tree nuts", "Soy", "Wheat / gluten", "Fish", "Shellfish", "Sesame")

_TERMS: dict[str, tuple[str, ...]] = {
    "Milk": (
        r"milks?", r"milk\s*solids?", r"creams?", r"butter", r"buttermilk", r"butterfat", r"milkfat", r"cheeses?",
        r"whey", r"caseins?", r"caseinates?", r"lactose", r"lactalbumin", r"lactoglobulin", r"yogh?urts?",
        r"curds?", r"ghee", r"dairy", r"paneer", r"khoa", r"khoya",
    ),
    "Egg": (
        r"eggs?", r"yolks?", r"albumen", r"albumin", r"ovalbumin", r"ovomucoid", r"lysozyme", r"mayonnaise",
        r"meringues?",
    ),
    "Peanut": (r"peanuts?", r"groundnuts?", r"arachis", r"monkey\s+nuts?"),
    "Tree nuts": (
        r"almonds?", r"badam", r"cashews?", r"hazelnuts?", r"filberts?", r"walnuts?", r"pecans?", r"pistachios?",
        r"macadamias?", r"brazil\s+nuts?", r"pine\s+nuts?", r"chestnuts?", r"praline", r"marzipan", r"nuts?",
    ),
    "Soy": (r"soy", r"soya", r"soja", r"soybeans?", r"soyabeans?", r"tofu", r"edamame", r"miso", r"tempeh", r"shoyu", r"tamari"),
    "Wheat / gluten": (
        r"wheat", r"gluten", r"semolina", r"suji", r"sooji", r"durum", r"spelt", r"kamut", r"farro", r"bulgur",
        r"couscous", r"seitan", r"maida", r"atta", r"rye", r"barley", r"malt", r"malted", r"oats?", r"oatmeal",
        r"triticale",
    ),
    "Fish": (
        r"fish", r"fishes", r"anchov(?:y|ies)", r"tuna", r"salmon", r"cod", r"sardines?", r"mackerel", r"haddock",
        r"pollock", r"tilapia", r"trout", r"herrings?", r"surimi", r"bonito",
    ),
    "Shellfish": (
        r"shellfish", r"crustaceans?", r"shrimps?", r"prawns?", r"crabs?", r"lobsters?", r"crayfish", r"krill",
        r"langoustines?", r"mussels?", r"oysters?", r"clams?", r"scallops?", r"squids?", r"octopus", r"cuttlefish",
        r"molluscs?", r"mollusks?",
    ),
    "Sesame": (r"sesame", r"tahini", r"tahina", r"gingelly", r"benne"),
}

# Words right before a milk term that make it a different food.
_NOT_MILK_BEFORE = {"coconut", "almond", "soy", "soya", "oat", "rice", "cocoa", "cacao", "shea", "peanut", "nut",
                    "cashew", "hazelnut", "apple", "mango"}
_NOT_MILK_AFTER = re.compile(r"\s+(?:of\s+tartar|thistle)\b", re.IGNORECASE)
# "Free from milk, egg and nuts" / "does not contain ..." negate the rest of the clause.
_NEGATED_CLAUSE = re.compile(r"\b(?:free\s+(?:from|of)|does\s+not\s+contain|do\s+not\s+contain|without)\b", re.IGNORECASE)
_CONTAINS = re.compile(r"\bcontains?\b", re.IGNORECASE)
_NO_BEFORE = re.compile(r"\b(?:no|non)[\s-]+(?:added\s+)?$", re.IGNORECASE)  # "no milk", "non-dairy"
_FREE_AFTER = re.compile(r"^[\s-]*free\b", re.IGNORECASE)  # "milk-free", "gluten free"
_PRECAUTION = re.compile(
    r"may\s+(?:also\s+)?contain|traces?\s+of|shared\s+(?:equipment|facilit)|same\s+(?:line|facility|factory)"
    r"|(?:facility|factory|equipment|line)\s+(?:that\s+)?(?:also\s+)?(?:process|handle|use)",
    re.IGNORECASE,
)

_PATTERNS = [
    (group, re.compile(rf"(?<!\w)(?:{'|'.join(terms)})(?!\w)", re.IGNORECASE)) for group, terms in _TERMS.items()
]


@dataclass(frozen=True)
class AllergenHit:
    group: str  # one of GROUPS
    start: int
    end: int
    term: str  # the words as written in the text


def _previous_word(text: str, start: int) -> str:
    match = re.search(r"([A-Za-z]+)[\s-]*$", text[:start])
    return match.group(1).lower() if match else ""


def _negated(text: str, start: int, end: int) -> bool:
    if _NO_BEFORE.search(text[max(0, start - 20) : start]) or _FREE_AFTER.match(text[end:]):
        return True
    clause = text[max(text.rfind(mark, 0, start) for mark in ".;!?") + 1 : start]
    negations = list(_NEGATED_CLAUSE.finditer(clause))
    # "Made without colours, contains milk": a later "contains" ends the negation.
    return bool(negations) and not _CONTAINS.search(clause, negations[-1].end())


def _ignored(group: str, text: str, start: int, end: int) -> bool:
    if _negated(text, start, end):
        return True
    if group == "Milk":
        if _previous_word(text, start) in _NOT_MILK_BEFORE or _NOT_MILK_AFTER.match(text[end:]):
            return True
    return False


def find_allergens(text: str) -> list[AllergenHit]:
    """Every allergen word in ``text``, in order, without overlaps."""
    hits: list[AllergenHit] = []
    for group, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            if not _ignored(group, text, match.start(), match.end()):
                hits.append(AllergenHit(group, match.start(), match.end(), match.group(0)))
    hits.sort(key=lambda hit: (hit.start, -(hit.end - hit.start)))
    kept: list[AllergenHit] = []
    for hit in hits:
        if not kept or hit.start >= kept[-1].end:
            kept.append(hit)
    return kept


def groups_in(text: str) -> list[str]:
    """The allergen groups named in ``text``, in display order."""
    found = {hit.group for hit in find_allergens(text)}
    return [group for group in GROUPS if group in found]


def is_precautionary(statement: str) -> bool:
    """True for "may contain" style statements (traces, shared equipment)."""
    return bool(_PRECAUTION.search(statement))


@dataclass
class AllergenSummary:
    contains: dict[str, list[str]] = field(default_factory=dict)  # group -> distinct terms as written
    may_contain: dict[str, list[str]] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.contains and not self.may_contain


def summarize(items: Sequence[str], notes: Sequence[str] = ()) -> AllergenSummary:
    """Allergens named in the ingredients and statements, split into contains / may contain.

    A group that is both contained and "may contain" is reported as contained.
    """
    contains: dict[str, list[str]] = {}
    may_contain: dict[str, list[str]] = {}
    sources = [(text, False) for text in items] + [(text, is_precautionary(text)) for text in notes]
    for text, precautionary in sources:
        target = may_contain if precautionary else contains
        for hit in find_allergens(text):
            terms = target.setdefault(hit.group, [])
            if hit.term.lower() not in (term.lower() for term in terms):
                terms.append(hit.term)
    ordered = lambda found: {group: found[group] for group in GROUPS if group in found}
    return AllergenSummary(
        contains=ordered(contains),
        may_contain=ordered({group: terms for group, terms in may_contain.items() if group not in contains}),
    )
