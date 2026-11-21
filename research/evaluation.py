"""Evaluation that mirrors the thesis scripts, so the app reproduces their numbers.

- Word images (recognizer only): ``test.py`` ``validation()``, with the label
  filtering and cleaning of ``dataset.LmdbDataset``.
- Photos (end-to-end): ``test2.py``, which compares the reconstructed text of
  each image with an image-level ground-truth file.

No Streamlit dependency.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

import lmdb
from PIL import Image

from research.catalog import list_images
from research.editing import read_image
from src.ocr_engine import Detection, RecognizerConfig, TextDetector, WordRecognizer, build_result

CONFIDENCE_THRESHOLD = 0.8  # test.py's conf_thr


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance between two strings or two word lists (test2.py)."""
    previous_row = list(range(len(b) + 1))
    for i, item_a in enumerate(a, 1):
        diagonal, previous_row[0] = previous_row[0], i
        for j, item_b in enumerate(b, 1):
            above = previous_row[j]
            previous_row[j] = diagonal if item_a == item_b else 1 + min(diagonal, previous_row[j], previous_row[j - 1])
            diagonal = above
    return previous_row[-1]


# ---------------------------------------------------------------- word level
@dataclass(frozen=True)
class WordSample:
    label: str  # as stored in the dataset
    image: bytes  # encoded image


def read_lmdb(path: Path) -> list[WordSample]:
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with env.begin(write=False) as txn:
            count = int(txn.get(b"num-samples"))
            return [
                WordSample(txn.get(b"label-%09d" % index).decode("utf-8"), bytes(txn.get(b"image-%09d" % index)))
                for index in range(1, count + 1)
            ]
    finally:
        env.close()


def is_case_sensitive(config: RecognizerConfig) -> bool:
    """test.py's --sensitive flag: the character set has upper-case letters."""
    return any(char.isupper() for char in config.character)


def prepare_word_samples(
    samples: Sequence[WordSample], config: RecognizerConfig, *, filtering_off: bool = False
) -> tuple[list[int], list[str], list[Image.Image]]:
    """Filter, decode and clean the samples for one checkpoint, as LmdbDataset does.

    Returns the indices of the samples that are kept, their cleaned labels and
    their images in the checkpoint's color mode.
    """
    out_of_char = f"[^{config.character}]"  # built exactly like dataset.py
    mode = "RGB" if config.rgb else "L"
    sensitive = is_case_sensitive(config)
    indices, labels, images = [], [], []
    for index, sample in enumerate(samples):
        label = sample.label
        if not filtering_off:
            if len(label) > config.batch_max_length:
                continue
            if re.search(out_of_char, label.lower()):
                continue
        try:
            image = Image.open(io.BytesIO(sample.image)).convert(mode)
        except OSError:
            image = Image.new(mode, (config.imgW, config.imgH))
            label = "[dummy_label]"
        if not sensitive:
            label = label.lower()
        indices.append(index)
        labels.append(re.sub(out_of_char, "", label))
        images.append(image)
    return indices, labels, images


@dataclass
class WordReport:
    samples: int
    accuracy: float  # percent
    norm_ed: float  # ICDAR2019 normalized edit distance (higher is better)
    cer: float
    wer: float
    f1: float
    avg_confidence: float
    coverage: float  # share of words with confidence >= threshold
    accuracy_at_threshold: float  # accuracy among those words
    rows: list[tuple[int, str, str, float, bool]] = field(default_factory=list)  # index, label, prediction, confidence, correct


def word_metrics(
    predictions: Sequence[str],
    labels: Sequence[str],
    confidences: Sequence[float],
    *,
    sensitive: bool,
    filtering_off: bool = False,
    indices: Sequence[int] | None = None,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> WordReport:
    """test.py validation() metrics for already decoded predictions."""
    indices = list(indices) if indices is not None else list(range(len(labels)))
    n_correct = 0
    norm_ed = 0.0
    char_edits = char_total = 0
    word_edits = word_total = 0
    true_positive_words = predicted_words = gt_words_total = 0
    confidence_sum = 0.0
    selected = selected_correct = 0
    rows = []
    for index, pred, gt, confidence in zip(indices, predictions, labels, confidences):
        if sensitive and filtering_off:
            pred = re.sub("[^0-9a-z]", "", pred.lower())
            gt = re.sub("[^0-9a-z]", "", gt.lower())
        correct = pred == gt
        n_correct += correct

        if len(gt) == 0 or len(pred) == 0:
            norm_ed += 0
        elif len(gt) > len(pred):
            norm_ed += 1 - levenshtein(pred, gt) / len(gt)
        else:
            norm_ed += 1 - levenshtein(pred, gt) / len(pred)

        if len(gt) > 0:
            char_edits += levenshtein(pred, gt)
            char_total += len(gt)

        gt_words, pred_words = gt.split(), pred.split()
        if gt_words:
            word_edits += levenshtein(pred_words, gt_words)
            word_total += len(gt_words)
        true_positive_words += sum((Counter(gt_words) & Counter(pred_words)).values())
        predicted_words += len(pred_words)
        gt_words_total += len(gt_words)

        confidence_sum += float(confidence)
        if float(confidence) >= threshold:
            selected += 1
            selected_correct += correct
        rows.append((index, gt, pred, float(confidence), correct))

    count = len(rows)
    precision = true_positive_words / predicted_words if predicted_words else 0.0
    recall = true_positive_words / gt_words_total if gt_words_total else 0.0
    return WordReport(
        samples=count,
        accuracy=n_correct / count * 100 if count else 0.0,
        norm_ed=norm_ed / count if count else 0.0,
        cer=char_edits / char_total if char_total else 0.0,
        wer=word_edits / word_total if word_total else 0.0,
        f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        avg_confidence=confidence_sum / count if count else 0.0,
        coverage=selected / count if count else 0.0,
        accuracy_at_threshold=selected_correct / selected if selected else 0.0,
        rows=rows,
    )


def evaluate_words(
    recognizer: WordRecognizer, samples: Sequence[WordSample], *, filtering_off: bool = False
) -> WordReport:
    """Run one checkpoint over a word dataset and score it like test.py."""
    indices, labels, images = prepare_word_samples(samples, recognizer.config, filtering_off=filtering_off)
    predictions, confidences = recognizer.recognize_images(images)
    return word_metrics(
        predictions,
        labels,
        confidences,
        sensitive=is_case_sensitive(recognizer.config),
        filtering_off=filtering_off,
        indices=indices,
    )


# --------------------------------------------------------------- image level
def normalize_key(stem: str) -> str:
    """test2.py: bahan_01 and bahan_1 map to the same key."""
    match = re.fullmatch(r"(.*_)(\d+)", stem)
    return f"{match.group(1)}{int(match.group(2))}" if match else stem


def normalize_text(text: str) -> str:
    """test2.py: newlines, tabs and repeated spaces become one space."""
    return " ".join(text.split())


@dataclass(frozen=True)
class ImageTruth:
    name: str  # as written in the gt file
    key: str  # normalize_key of the file stem
    text: str  # normalized ground-truth text


def read_image_ground_truth(path: Path) -> list[ImageTruth]:
    """test2.py read_gt(): one entry per key; a later line wins."""
    truths: dict[str, ImageTruth] = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            key = normalize_key(PureWindowsPath(parts[0]).stem)
            truths[key] = ImageTruth(parts[0], key, normalize_text(parts[1]))
    return list(truths.values())


def match_images(truths: Sequence[ImageTruth], folder: Path) -> tuple[list[tuple[ImageTruth, Path]], list[ImageTruth]]:
    """Pair ground-truth entries with images in ``folder`` by normalized file stem."""
    index = {normalize_key(path.stem): path for path in list_images(folder)}
    matched = [(truth, index[truth.key]) for truth in truths if truth.key in index]
    missing = [truth for truth in truths if truth.key not in index]
    return matched, missing


def guess_image_folder(truths: Sequence[ImageTruth], folders: Sequence[Path]) -> Path | None:
    """The folder that matches the most ground-truth entries."""
    best, best_count = None, 0
    for folder in folders:
        count = len(match_images(truths, folder)[0])
        if count > best_count:
            best, best_count = folder, count
    return best


def _f1(g: str, p: str) -> float:
    g_words, p_words = g.split(), p.split()
    if not g_words and not p_words:
        return 1.0
    if not g_words or not p_words:
        return 0.0
    tp = sum((Counter(g_words) & Counter(p_words)).values())
    precision, recall = tp / len(p_words), tp / len(g_words)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _bleu1(g: str, p: str) -> float:
    g_words, p_words = g.split(), p.split()
    if not p_words:
        return 0.0
    overlap = sum((Counter(g_words) & Counter(p_words)).values())
    p1 = overlap / len(p_words)
    brevity = 1.0 if len(p_words) > len(g_words) else pow(2.718281828, 1 - len(g_words) / len(p_words))
    return brevity * p1


def _rouge_l(g: str, p: str) -> float:
    g_words, p_words = g.split(), p.split()
    if not g_words and not p_words:
        return 1.0
    if not g_words or not p_words:
        return 0.0
    m, n = len(g_words), len(p_words)
    table = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            table[i + 1][j + 1] = table[i][j] + 1 if g_words[i] == p_words[j] else max(table[i][j + 1], table[i + 1][j])
    lcs = table[m][n]
    precision, recall = lcs / n, lcs / m
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


@dataclass
class ImageScore:
    key: str
    gt_words: int
    pred_words: int
    norm_ed: float
    f1: float
    bleu1: float
    rouge_l: float
    word_accuracy: float  # percent
    cer: float
    wer: float
    ground_truth: str
    prediction: str


@dataclass
class ImageReport:
    samples: int
    gt_words: int
    pred_words: int
    norm_ed: float
    f1: float
    bleu1: float
    rouge_l: float
    word_accuracy: float  # percent, 1 - corpus WER
    cer: float
    wer: float
    images: list[ImageScore] = field(default_factory=list)


def image_metrics(pairs: Sequence[tuple[str, str, str]]) -> ImageReport:
    """test2.py metrics for (key, ground truth, prediction) triples, in key order."""
    scores = []
    char_edits = char_total = word_edits = word_total = 0
    for key, ground_truth, prediction in sorted(pairs):
        g, p = normalize_text(ground_truth), normalize_text(prediction)
        gt_words, pred_words = g.split(), p.split()
        sample_char_edits = levenshtein(p, g)
        sample_word_edits = levenshtein(pred_words, gt_words)
        sample_wer = sample_word_edits / len(gt_words) if gt_words else 0.0
        scores.append(
            ImageScore(
                key=key,
                gt_words=len(gt_words),
                pred_words=len(pred_words),
                norm_ed=1 - levenshtein(p, g) / max(len(g), len(p)) if g and p else 0,
                f1=_f1(g, p),
                bleu1=_bleu1(g, p),
                rouge_l=_rouge_l(g, p),
                word_accuracy=max(0.0, 1.0 - sample_wer) * 100.0,
                cer=sample_char_edits / len(g) if g else 0.0,
                wer=sample_wer,
                ground_truth=g,
                prediction=p,
            )
        )
        char_edits += sample_char_edits
        char_total += len(g)
        word_edits += sample_word_edits
        word_total += len(gt_words)

    n = len(scores)
    wer = word_edits / word_total if word_total else 0.0
    return ImageReport(
        samples=n,
        gt_words=word_total,
        pred_words=sum(score.pred_words for score in scores),
        norm_ed=sum(score.norm_ed for score in scores) / n if n else 0.0,
        f1=sum(score.f1 for score in scores) / n if n else 0.0,
        bleu1=sum(score.bleu1 for score in scores) / n if n else 0.0,
        rouge_l=sum(score.rouge_l for score in scores) / n if n else 0.0,
        word_accuracy=max(0.0, 1.0 - wer) * 100.0,
        cer=char_edits / char_total if char_total else 0.0,
        wer=wer,
        images=scores,
    )


def detect_images(
    detector: TextDetector,
    matched: Sequence[tuple[ImageTruth, Path]],
    on_progress: Callable[[int, int], None] | None = None,
) -> list[tuple[ImageTruth, Detection]]:
    """Detect words once per image; every checkpoint then reads the same crops."""
    detections = []
    for number, (truth, path) in enumerate(matched, start=1):
        image = read_image(path)
        if image is not None:
            detections.append((truth, detector.detect(image)))
        if on_progress:
            on_progress(number, len(matched))
    return detections


def evaluate_photos(recognizer: WordRecognizer, detections: Sequence[tuple[ImageTruth, Detection]]) -> ImageReport:
    """One checkpoint end to end: read the detected words and score like test2.py."""
    pairs = []
    for truth, detection in detections:
        texts, confidences = recognizer.recognize(detection.crops)
        result = build_result(detection, texts, confidences)
        pairs.append((truth.key, truth.text, result.text))
    return image_metrics(pairs)
