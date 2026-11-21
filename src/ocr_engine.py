"""In-memory OCR engine: CRAFT text detection + CRNN (TPS-ResNet-BiLSTM-Attn) recognition.

The engine loads both models once and processes one image at a time without
touching the file system, so web apps (and a future API) can share it. It has no
Streamlit dependency. The CLI in ``src/ocr_pipeline.py`` uses the same building
blocks (detection, line grouping, word cropping), so both produce the same text.

``OCREngine`` combines a ``TextDetector`` (CRAFT) and a ``WordRecognizer`` (one
CRNN checkpoint). The researcher app uses the two parts directly, for example to
detect once and compare several checkpoints on the same words.

Example:
    engine = OCREngine()                      # final thesis model, auto device
    result = engine.process(cv2.imread("label.jpg"))
    print(result.text)
    for word in result.low_confidence_words(0.8):
        print(word.text, word.confidence)
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import SimpleNamespace

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from CRAFT.craft import CRAFT
from dataset import AlignCollate
from model import Model
from src.ocr_pipeline import (
    SENSITIVE_CHARACTER_SET,
    crop_word,
    detect_text_regions,
    group_boxes_by_lines,
    strip_module_prefix,
)
from utils import AttnLabelConverter, CTCLabelConverter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECOGNIZER_PATH = PROJECT_ROOT / "saved_models" / "FT_801010_off" / "best_accuracy.pth"
DEFAULT_DETECTOR_PATH = PROJECT_ROOT / "saved_models" / "craft_mlt_25k.pth"

ATTN_EOS_INDEX = 1  # AttnLabelConverter tokens: 0 = [GO], 1 = [s]


@dataclass
class RecognizerConfig:
    """CRNN settings. Defaults match the final thesis model (FT_801010_off)."""

    Transformation: str = "TPS"
    FeatureExtraction: str = "ResNet"
    SequenceModeling: str = "BiLSTM"
    Prediction: str = "Attn"
    num_fiducial: int = 20
    imgH: int = 32
    imgW: int = 100
    input_channel: int = 1
    output_channel: int = 512
    hidden_size: int = 256
    batch_max_length: int = 60
    character: str = SENSITIVE_CHARACTER_SET
    rgb: bool = False
    PAD: bool = False

    @classmethod
    def from_opt_file(cls, opt_path: str | Path) -> "RecognizerConfig":
        """Read the ``opt.txt`` that train.py writes next to each checkpoint."""
        values = {}
        for line in Path(opt_path).read_text(encoding="utf-8").splitlines():
            if ": " in line:
                key, value = line.split(": ", 1)
                values[key.strip()] = value

        defaults = cls()
        kwargs = {}
        for config_field in fields(cls):
            if config_field.name not in values:
                continue
            raw = values[config_field.name]
            default = getattr(defaults, config_field.name)
            if isinstance(default, bool):
                kwargs[config_field.name] = raw.strip().lower() in ("true", "1", "yes")
            elif isinstance(default, int):
                kwargs[config_field.name] = int(raw)
            elif config_field.name == "character":
                kwargs[config_field.name] = raw  # keep exact characters
            else:
                kwargs[config_field.name] = raw.strip()

        config = cls(**kwargs)
        if values.get("sensitive", "").strip().lower() == "true":
            config.character = SENSITIVE_CHARACTER_SET
        if config.rgb:
            config.input_channel = 3
        return config

    @classmethod
    def for_checkpoint(cls, checkpoint_path: str | Path) -> "RecognizerConfig":
        """Use the checkpoint's opt.txt when present, otherwise the defaults."""
        opt_path = Path(checkpoint_path).with_name("opt.txt")
        return cls.from_opt_file(opt_path) if opt_path.exists() else cls()


@dataclass
class DetectorConfig:
    """CRAFT post-processing settings used in the thesis."""

    text_threshold: float = 0.8
    low_text: float = 0.4
    link_threshold: float = 0.25
    canvas_size: int = 1280
    mag_ratio: float = 1.5
    poly: bool = False


@dataclass(frozen=True, eq=False)  # holds numpy arrays; compare by identity
class Word:
    text: str
    confidence: float  # product of per-character max probabilities, 0..1
    box: np.ndarray  # (4, 2) float32 quad in original image coordinates
    line: int  # 1-based line number in reading order
    position: int  # 1-based position within its detected line
    crop: np.ndarray  # BGR word image that was fed to the recognizer


@dataclass(eq=False)
class Detection:
    """The words CRAFT found in one image, cut out in reading order (no text yet)."""

    placements: list[tuple[int, int, np.ndarray]]  # (line, position, quad) for every crop
    crops: list[np.ndarray]  # BGR word images, same order as placements
    image_size: tuple[int, int]  # (width, height)
    score_text: np.ndarray  # CRAFT region score map
    score_link: np.ndarray  # affinity score map (refined when the refiner is on)
    polygons: list[np.ndarray] | None = None  # refiner polygons for drawing; None without refiner
    timings: dict[str, float] = field(default_factory=dict)  # seconds: detect, crop


@dataclass(eq=False)
class OCRResult:
    words: list[Word]
    lines: list[str]  # reconstructed lines in reading order
    image_size: tuple[int, int]  # (width, height)
    timings: dict[str, float] = field(default_factory=dict)  # seconds
    score_text: np.ndarray | None = None  # CRAFT region score map (optional)
    score_link: np.ndarray | None = None  # CRAFT affinity score map (optional)
    polygons: list[np.ndarray] | None = None  # refiner polygons (only with the refiner)

    @property
    def text(self) -> str:
        """Text with original line breaks (same format as the CLI's merged output)."""
        return "\n".join(self.lines)

    def low_confidence_words(self, threshold: float) -> list[Word]:
        return [word for word in self.words if word.confidence < threshold]


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, str) and device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available on this machine.")
    return resolved


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Expected a non-empty image array.")
    if image.ndim == 2:
        return cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv.cvtColor(image, cv.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _load_state(path: Path, device: torch.device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return strip_module_prefix(state)


def _require(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} weights not found: {path}")
    return path


class TextDetector:
    """CRAFT word detection plus reading-order cropping. Thread-safe.

    With ``refiner_path`` the CRAFT link refiner runs too, exactly as in the CLI:
    its polygons and refined link map are kept for inspection, while the words
    are still cut from the plain CRAFT boxes, so the text does not change.
    """

    def __init__(
        self,
        detector_path: str | Path = DEFAULT_DETECTOR_PATH,
        *,
        config: DetectorConfig | None = None,
        device: str | torch.device = "auto",
        refiner_path: str | Path | None = None,
    ):
        self.detector_path = _require(Path(detector_path), "CRAFT detector")
        self.refiner_path = _require(Path(refiner_path), "CRAFT refiner") if refiner_path else None
        self.device = resolve_device(device)
        self.config = config or DetectorConfig()

        self.model = CRAFT()
        self.model.load_state_dict(_load_state(self.detector_path, self.device))
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = False
        self.model = self.model.to(self.device).eval()

        self.refiner = None
        if self.refiner_path is not None:
            from refinenet import RefineNet

            refiner = RefineNet()
            refiner.load_state_dict(_load_state(self.refiner_path, self.device))
            self.refiner = refiner.to(self.device).eval()

        # detect_text_regions() expects CLI-style args.
        self._args = SimpleNamespace(**vars(self.config), cuda=self.device.type == "cuda")
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray) -> Detection:
        """Find and cut out every word of an upright BGR image."""
        image_bgr = _to_bgr(image)
        height, width = image_bgr.shape[:2]
        with self._lock, torch.no_grad():
            start = time.perf_counter()
            # As in the thesis pipeline, CRAFT receives the BGR array; kept for result parity.
            if self.refiner is None:
                boxes, _, score_text, score_link = detect_text_regions(self.model, image_bgr, self._args)
                polygons = None
            else:
                refined_args = SimpleNamespace(**{**vars(self._args), "poly": True})
                boxes_refined, polygons_refined, score_text, score_link = detect_text_regions(
                    self.model, image_bgr, refined_args, refine_net=self.refiner
                )
                boxes, _, _, _ = detect_text_regions(self.model, image_bgr, self._args)
                if boxes is None or len(boxes) == 0:
                    boxes = boxes_refined
                usable = polygons_refined is not None and len(polygons_refined) == len(boxes_refined)
                polygons = [np.asarray(polygon, dtype=np.float32) for polygon in (polygons_refined if usable else boxes)]
            detected = time.perf_counter()

            placements, crops = [], []
            for line_number, line_indices in enumerate(group_boxes_by_lines(boxes), start=1):
                for position, box_index in enumerate(line_indices, start=1):
                    quad = np.asarray(boxes[box_index], dtype=np.float32)
                    crop = crop_word(image_bgr, quad)
                    if crop is None:
                        continue
                    placements.append((line_number, position, quad))
                    crops.append(crop)
            cropped = time.perf_counter()

        return Detection(
            placements=placements,
            crops=crops,
            image_size=(width, height),
            score_text=score_text,
            score_link=score_link,
            polygons=polygons,
            timings={"detect": detected - start, "crop": cropped - detected},
        )


class WordRecognizer:
    """A CRNN checkpoint that reads word images. Thread-safe."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_RECOGNIZER_PATH,
        *,
        config: RecognizerConfig | None = None,
        device: str | torch.device = "auto",
        batch_size: int = 32,
    ):
        self.checkpoint_path = _require(Path(checkpoint_path), "Recognizer")
        self.device = resolve_device(device)
        self.config = config or RecognizerConfig.for_checkpoint(self.checkpoint_path)
        self.batch_size = batch_size

        if "CTC" in self.config.Prediction:
            self.converter = CTCLabelConverter(self.config.character)
        else:
            self.converter = AttnLabelConverter(self.config.character)
        model_opt = SimpleNamespace(**vars(self.config), num_class=len(self.converter.character))
        model = Model(model_opt)
        model.load_state_dict(_load_state(self.checkpoint_path, self.device))
        self.model = model.to(self.device).eval()
        self._align = AlignCollate(imgH=self.config.imgH, imgW=self.config.imgW, keep_ratio_with_pad=self.config.PAD)
        self._lock = threading.Lock()

    def recognize(self, crops_bgr: list[np.ndarray]) -> tuple[list[str], list[float]]:
        """Read BGR word crops; returns texts and confidences."""
        mode = "RGB" if self.config.rgb else "L"
        # Same pixels the CLI's RawDataset reads back from its PNG crops.
        images = [Image.fromarray(cv.cvtColor(crop, cv.COLOR_BGR2RGB)).convert(mode) for crop in crops_bgr]
        return self.recognize_images(images)

    def recognize_images(self, images: list[Image.Image]) -> tuple[list[str], list[float]]:
        """Read PIL word images that are already in the model's color mode (L or RGB)."""
        config = self.config
        texts: list[str] = []
        confidences: list[float] = []
        with self._lock, torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                image_tensors, _ = self._align([(image, index) for index, image in enumerate(batch)])
                image_tensors = image_tensors.to(self.device)
                count = image_tensors.size(0)
                text_for_pred = torch.LongTensor(count, config.batch_max_length + 1).fill_(0).to(self.device)

                if "CTC" in config.Prediction:
                    predictions = self.model(image_tensors, text_for_pred)
                    sizes = torch.IntTensor([predictions.size(1)] * count)
                    _, indices = predictions.max(2)
                    batch_texts = self.converter.decode(indices, sizes)
                    max_probs, _ = F.softmax(predictions, dim=2).max(dim=2)
                    batch_confidences = [float(row.cumprod(dim=0)[-1]) for row in max_probs]
                else:
                    predictions = self.model(image_tensors, text_for_pred, is_train=False)
                    _, indices = predictions.max(2)
                    lengths = torch.IntTensor([config.batch_max_length] * count)
                    batch_texts = [text.split("[s]")[0] for text in self.converter.decode(indices, lengths)]
                    max_probs, _ = F.softmax(predictions, dim=2).max(dim=2)
                    batch_confidences = []
                    for index_row, prob_row in zip(indices, max_probs):
                        eos = (index_row == ATTN_EOS_INDEX).nonzero()
                        end = int(eos[0]) if len(eos) else index_row.numel()
                        batch_confidences.append(float(prob_row[:end].prod()) if end > 0 else 0.0)

                texts.extend(batch_texts)
                confidences.extend(batch_confidences)
        return texts, confidences


def build_result(
    detection: Detection,
    texts: list[str],
    confidences: list[float],
    *,
    recognize_seconds: float = 0.0,
    keep_score_maps: bool = False,
) -> OCRResult:
    """Combine a detection with the recognizer output into words and lines."""
    words = [
        Word(text=text, confidence=confidence, box=quad, line=line, position=position, crop=crop)
        for (line, position, quad), crop, text, confidence in zip(
            detection.placements, detection.crops, texts, confidences
        )
    ]
    lines = []
    for line_number in sorted({word.line for word in words}):
        line_words = sorted((word for word in words if word.line == line_number), key=lambda w: w.position)
        lines.append(" ".join(word.text for word in line_words))

    timings = {**detection.timings, "recognize": recognize_seconds}
    timings["total"] = timings.get("detect", 0.0) + timings.get("crop", 0.0) + recognize_seconds
    return OCRResult(
        words=words,
        lines=lines,
        image_size=detection.image_size,
        timings=timings,
        score_text=detection.score_text if keep_score_maps else None,
        score_link=detection.score_link if keep_score_maps else None,
        polygons=detection.polygons,
    )


class OCREngine:
    """Loads CRAFT and the CRNN recognizer once; ``process`` runs one image in memory.

    Thread-safe: concurrent calls are serialized, which also avoids CPU
    oversubscription on small hosting machines.
    """

    def __init__(
        self,
        recognizer_path: str | Path = DEFAULT_RECOGNIZER_PATH,
        detector_path: str | Path = DEFAULT_DETECTOR_PATH,
        *,
        recognizer_config: RecognizerConfig | None = None,
        detector_config: DetectorConfig | None = None,
        device: str | torch.device = "auto",
        batch_size: int = 32,
        refiner_path: str | Path | None = None,
    ):
        _require(Path(recognizer_path), "Recognizer")
        _require(Path(detector_path), "CRAFT detector")
        self.device = resolve_device(device)
        self.word_recognizer = WordRecognizer(
            recognizer_path, config=recognizer_config, device=self.device, batch_size=batch_size
        )
        self.text_detector = TextDetector(
            detector_path, config=detector_config, device=self.device, refiner_path=refiner_path
        )
        self.recognizer_path = self.word_recognizer.checkpoint_path
        self.detector_path = self.text_detector.detector_path
        self.recognizer_config = self.word_recognizer.config
        self.detector_config = self.text_detector.config
        self.batch_size = batch_size
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        """True while another call to ``process`` holds the engine."""
        return self._lock.locked()

    def process(
        self,
        image: np.ndarray,
        *,
        keep_score_maps: bool = False,
        on_progress: Callable[[str, int], None] | None = None,
    ) -> OCRResult:
        """Detect, crop and recognize all words in an upright BGR image.

        ``on_progress(stage, count)`` is called with ("detect", 0) and then
        ("recognize", number_of_words) so a UI can report progress.
        """
        image_bgr = _to_bgr(image)
        notify = on_progress or (lambda stage, count: None)
        with self._lock:
            notify("detect", 0)
            detection = self.text_detector.detect(image_bgr)
            notify("recognize", len(detection.crops))
            start = time.perf_counter()
            texts, confidences = self.word_recognizer.recognize(detection.crops)
            recognize_seconds = time.perf_counter() - start
        return build_result(
            detection,
            texts,
            confidences,
            recognize_seconds=recognize_seconds,
            keep_score_maps=keep_score_maps,
        )


def draw_word_boxes(
    image_bgr: np.ndarray,
    words: list[Word],
    *,
    color: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
    low_confidence_threshold: float | None = None,
    low_confidence_color: tuple[int, int, int] = (0, 200, 255),
    outline_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Return a copy of the image with word boxes drawn (BGR colors).

    ``outline_color`` draws a wider halo under each box so boxes stay visible
    on both light and dark packaging.
    """
    canvas = _to_bgr(image_bgr).copy()
    for word in words:
        is_low = low_confidence_threshold is not None and word.confidence < low_confidence_threshold
        points = np.round(word.box).astype(np.int32).reshape(-1, 1, 2)
        if outline_color is not None:
            cv.polylines(canvas, [points], True, outline_color, thickness + 2, cv.LINE_AA)
        cv.polylines(canvas, [points], True, low_confidence_color if is_low else color, thickness, cv.LINE_AA)
    return canvas
