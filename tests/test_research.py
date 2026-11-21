"""Unit tests for the researcher app modules (no model weights needed).

Run with plain Python:
    python tests/test_research.py
or with pytest:
    pytest tests/test_research.py
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.catalog import ALPHANUMERIC, _pretrained_config  # noqa: E402
from research.editing import apply_geometry, is_untouched  # noqa: E402
from research.evaluation import (  # noqa: E402
    WordSample,
    image_metrics,
    levenshtein,
    normalize_key,
    prepare_word_samples,
    read_image_ground_truth,
    word_metrics,
)
from src.ocr_engine import RecognizerConfig  # noqa: E402
from src.ocr_pipeline import SENSITIVE_CHARACTER_SET  # noqa: E402


def close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return abs(a - b) <= tolerance


# --------------------------------------------------------------- metrics
def test_levenshtein_on_strings_and_word_lists():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein(["a", "b", "c"], ["a", "c"]) == 1


def test_word_metrics_match_hand_computed_values():
    report = word_metrics(["abc", "abd", ""], ["abc", "abc", "x"], [0.9, 0.5, 0.0], sensitive=True)
    assert report.samples == 3
    assert close(report.accuracy, 100 / 3)
    assert close(report.norm_ed, (1 + (1 - 1 / 3) + 0) / 3)
    assert close(report.cer, 2 / 7)
    assert close(report.wer, 2 / 3)
    assert close(report.f1, 0.4)
    assert close(report.coverage, 1 / 3) and close(report.accuracy_at_threshold, 1.0)


def test_ignore_case_protocol_compares_letters_and_digits_only():
    report = word_metrics(["Water"], ["water,"], [1.0], sensitive=True, filtering_off=True)
    assert report.accuracy == 100.0


def _png(width: int = 40, height: int = 16) -> bytes:
    return cv.imencode(".png", np.full((height, width, 3), 255, np.uint8))[1].tobytes()


def test_sample_filtering_follows_the_thesis_data_loader():
    samples = [WordSample("Water,", _png()), WordSample("Sugar", _png()), WordSample("x" * 30, _png())]
    lower = RecognizerConfig(character=ALPHANUMERIC, batch_max_length=25)
    indices, labels, images = prepare_word_samples(samples, lower)
    assert indices == [1] and labels == ["sugar"] and images[0].mode == "L"
    indices, labels, _ = prepare_word_samples(samples, lower, filtering_off=True)
    assert labels == ["water", "sugar", "x" * 30]
    sensitive = RecognizerConfig(character=SENSITIVE_CHARACTER_SET, batch_max_length=25)
    indices, labels, _ = prepare_word_samples(samples, sensitive)
    assert labels == ["Water,", "Sugar"]


def test_ground_truth_keys_match_test2():
    assert normalize_key("bahan_01") == normalize_key("bahan_1") == "bahan_1"
    assert normalize_key("01") == "01"
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "gt.txt"
        path.write_text("﻿bahan\\bahan_01.png\tWater,  Sugar\nimages/x/02.jpeg\tSalt\n", encoding="utf-8")
        truths = read_image_ground_truth(path)
    assert [(truth.key, truth.text) for truth in truths] == [("bahan_1", "Water, Sugar"), ("02", "Salt")]


def test_image_metrics_equal_test2_output():
    pairs = [
        ("01", "Ingredients: Water, Sugar, Salt, Citric Acid.", "Ingredients: Vilater, Sugar, Salt Citric Acid."),
        ("02", "Milk powder (20%), cocoa powder, sugar", "Milk powder (20%), cocoa, powder sugar extra"),
        ("03", "Wheat flour, palm oil", ""),
    ]
    ours = image_metrics(pairs)
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        (folder / "gt.txt").write_text("".join(f"{key}.jpg\t{gt}\n" for key, gt, _ in pairs), encoding="utf-8")
        for key, _, prediction in pairs:
            (folder / f"res_{key}_merged.txt").write_text(prediction + "\n", encoding="utf-8")
        output = subprocess.run(
            [sys.executable, str(ROOT / "test2.py"), "--gt", str(folder / "gt.txt"), "--merged_dir", str(folder)],
            capture_output=True, text=True, check=True,
        ).stdout
    expected = {
        "norm_ED": ours.norm_ed, "F1 (word-level)": ours.f1, "BLEU-1": ours.bleu1, "ROUGE-L": ours.rouge_l,
        "Word Accuracy": ours.word_accuracy, "CER": ours.cer, "WER": ours.wer,
    }
    for name, value in expected.items():
        printed = float(re.search(rf"^{re.escape(name)}: ([0-9.]+)$", output, re.MULTILINE).group(1))
        assert f"{value:.3f}" == f"{printed:.3f}", (name, value, printed)


# ------------------------------------------------------------- checkpoints
def test_pretrained_settings_come_from_the_file_name():
    config = _pretrained_config(Path("TPS-ResNet-BiLSTM-Attn-case-sensitive.pth"))
    assert (config.Transformation, config.FeatureExtraction, config.SequenceModeling, config.Prediction) == (
        "TPS", "ResNet", "BiLSTM", "Attn"
    )
    assert config.character == SENSITIVE_CHARACTER_SET and config.batch_max_length == 25
    crnn = _pretrained_config(Path("CRNN-PyTorchCTC.pth"))
    assert (crnn.Transformation, crnn.FeatureExtraction, crnn.SequenceModeling, crnn.Prediction) == (
        "None", "VGG", "BiLSTM", "CTC"
    )
    assert crnn.character == ALPHANUMERIC
    assert _pretrained_config(Path("craft_mlt_25k.pth")) is None


# ---------------------------------------------------------------- editing
def _test_image() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(30, 50, 3), dtype=np.uint8)


def test_quarter_turns_are_exact_and_clockwise():
    image = _test_image()
    full = {"x": 0, "y": 0, "w": 30, "h": 50}
    assert np.array_equal(apply_geometry(image, quarter=1, fine=0, crop=full), np.rot90(image, k=-1))
    assert np.array_equal(apply_geometry(image, quarter=3, fine=0, crop=full), np.rot90(image, k=1))
    half = apply_geometry(image, quarter=2, fine=0, crop={"x": 0, "y": 0, "w": 25, "h": 30})
    assert np.array_equal(half, np.rot90(image, k=2)[:, :25])


def test_general_rotation_matches_quarter_turns_and_crop_size():
    image = _test_image()
    exact = apply_geometry(image, quarter=1, fine=0, crop={"x": 0, "y": 0, "w": 30, "h": 50})
    general = apply_geometry(image, quarter=0, fine=90.0, crop={"x": 0, "y": 0, "w": 30, "h": 50})
    assert general.shape == exact.shape
    assert np.abs(general.astype(int) - exact.astype(int)).max() <= 2
    tilted = apply_geometry(image, quarter=0, fine=5.5, crop={"x": 3.2, "y": 4.7, "w": 20.4, "h": 10.6})
    assert tilted.shape[:2] == (11, 20)


def test_untouched_geometry():
    assert is_untouched({"quarter": 0, "fine": 0, "width": 50, "height": 30, "crop": {"x": 0, "y": 0, "w": 50, "h": 30}})
    assert not is_untouched({"quarter": 1, "fine": 0, "width": 50, "height": 30, "crop": {"x": 0, "y": 0, "w": 30, "h": 50}})


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} passed")
