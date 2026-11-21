"""Regression test: OCREngine must reproduce the thesis pipeline output exactly.

The baseline (tests/fixtures/baseline_gambar_simulasi.txt) is the merged output
the original pipeline produced for images/gambar_simulasi with the final model.
Requires the model weights in saved_models/ (not stored in git).

Run with plain Python:
    python tests/test_engine_regression.py            # auto device
    python tests/test_engine_regression.py --device cpu
or with pytest:
    pytest tests/
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2 as cv  # noqa: E402

from src.ocr_engine import DEFAULT_DETECTOR_PATH, DEFAULT_RECOGNIZER_PATH, OCREngine  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "baseline_gambar_simulasi.txt"
IMAGES_DIR = ROOT / "images" / "gambar_simulasi"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_baseline(path: Path) -> dict[str, str]:
    """Parse the CLI merged format: '<image stem>\\t<text with newlines>' blocks split by blank lines."""
    content = path.read_text(encoding="utf-8").strip("\n")
    baseline = {}
    for block in re.split(r"\n\n(?=[^\n\t]+\t)", content):
        name, text = block.split("\t", 1)
        baseline[name] = text.rstrip("\n")
    return baseline


def run_regression(device: str = "auto"):
    """Return (rows, engine) where rows = [(stem, passed, expected, actual, seconds)]."""
    baseline = read_baseline(FIXTURE)
    engine = OCREngine(device=device)
    rows = []
    for image_path in sorted(IMAGES_DIR.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        result = engine.process(cv.imread(str(image_path)))
        expected = baseline.get(image_path.stem)
        rows.append((image_path.stem, result.text == expected, expected, result.text, result.timings["total"]))
    missing = set(baseline) - {row[0] for row in rows}
    if missing:
        raise AssertionError(f"Baseline images not found in {IMAGES_DIR}: {sorted(missing)}")
    return rows, engine


def test_engine_matches_thesis_pipeline():
    if not (DEFAULT_RECOGNIZER_PATH.exists() and DEFAULT_DETECTOR_PATH.exists()):
        import pytest

        pytest.skip("Model weights are not available in saved_models/.")
    rows, _ = run_regression()
    failures = [row[0] for row in rows if not row[1]]
    assert not failures, f"OCR text differs from the baseline for: {failures}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    args = parser.parse_args()

    rows, engine = run_regression(args.device)
    print(f"device: {engine.device} | model: {engine.recognizer_path.relative_to(ROOT)}")
    for stem, passed, expected, actual, seconds in rows:
        print(f"  {'PASS' if passed else 'FAIL'}  {stem}  ({seconds:.2f} s)")
        if not passed:
            print(f"    expected: {expected!r}\n    actual:   {actual!r}")
    failed = sum(1 for row in rows if not row[1])
    print(f"{len(rows) - failed}/{len(rows)} images identical to the baseline")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
