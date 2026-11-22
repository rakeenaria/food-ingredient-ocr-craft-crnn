"""Build src/data/ingredient_vocabulary.txt for the OCR correction suggestions (PRD P9).

Sources:
- the word labels of the thesis train and validation splits (LMDB under data/);
- src/data/ingredient_terms.txt, common ingredient words written by hand.

The test split is left out on purpose, so the suggestions can be evaluated on
it honestly. Each output line is "<word>\t<count>": lower case, or the mixed-case
spelling when one exists (NaCl). Run from the repository root:

    python scripts/build_vocabulary.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import lmdb

ROOT = Path(__file__).resolve().parents[1]
SPLITS = [ROOT / "data" / "train_lmdb_801010_gt2", ROOT / "data" / "valid_lmdb_801010_gt2"]
TERMS = ROOT / "src" / "data" / "ingredient_terms.txt"
OUTPUT = ROOT / "src" / "data" / "ingredient_vocabulary.txt"
CORE = re.compile(r"^[^A-Za-z]*([A-Za-z](?:[A-Za-z']*[A-Za-z])?)[^A-Za-z]*$")


def label_words(path: Path) -> list[str]:
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with env.begin(write=False) as txn:
            count = int(txn.get(b"num-samples"))
            labels = [txn.get(b"label-%09d" % index).decode("utf-8") for index in range(1, count + 1)]
    finally:
        env.close()
    words = []
    for label in labels:
        match = CORE.match(label)
        if match:
            words.append(match.group(1))
    return words


def main() -> int:
    missing = [path for path in SPLITS if not path.exists()]
    if missing:
        print(f"Missing datasets: {', '.join(str(path) for path in missing)}", file=sys.stderr)
        return 1

    counts: Counter[str] = Counter()
    spelling: dict[str, str] = {}
    for path in SPLITS:
        for word in label_words(path):
            counts[word.lower()] += 1
    for line in TERMS.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        for word in line.split():
            key = word.lower()
            counts[key] += 1
            if word != key:
                spelling[key] = word  # keep NaCl as NaCl

    rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    OUTPUT.write_text("".join(f"{spelling.get(word, word)}\t{count}\n" for word, count in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} words to {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
