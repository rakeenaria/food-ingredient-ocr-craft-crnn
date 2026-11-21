"""What the researcher app can work with: checkpoints, image folders and ground truth.

Everything is discovered from the repository, so a new checkpoint folder or
dataset shows up without code changes. No Streamlit dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import lmdb

from src.ocr_engine import DEFAULT_RECOGNIZER_PATH, PROJECT_ROOT, RecognizerConfig
from src.ocr_pipeline import SENSITIVE_CHARACTER_SET

MODELS_DIR = PROJECT_ROOT / "saved_models"
DETECTOR_PATH = MODELS_DIR / "craft_mlt_25k.pth"
REFINER_PATH = MODELS_DIR / "craft_refiner_CTW1500.pth"
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ALPHANUMERIC = "0123456789abcdefghijklmnopqrstuvwxyz"

# Stage names exactly as model.py expects them.
_STAGE_NAMES = {
    "none": "None", "tps": "TPS", "vgg": "VGG", "rcnn": "RCNN", "resnet": "ResNet",
    "bilstm": "BiLSTM", "ctc": "CTC", "attn": "Attn",
}
_ARCHITECTURE = re.compile(r"(None|TPS)-(VGG|RCNN|ResNet)-(None|BiLSTM)-(CTC|Attn)", re.IGNORECASE)
# The benchmark's CRNN release uses its own file name.
_ALIASES = {"crnn-pytorchctc": ("None", "VGG", "BiLSTM", "CTC")}


def relative(path: Path) -> str:
    """A path relative to the repository (forward slashes) when possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


@dataclass(frozen=True, eq=False)
class Checkpoint:
    path: Path
    name: str  # "FT_801010_off / best_accuracy"
    group: str  # "Fine-tuned" or "Pretrained"
    config: RecognizerConfig
    settings_from: str  # "opt.txt" or "file name"

    @property
    def key(self) -> str:
        """Stable id for widgets and caches."""
        return relative(self.path)

    @property
    def is_final(self) -> bool:
        return self.path.resolve() == DEFAULT_RECOGNIZER_PATH.resolve()

    @property
    def label(self) -> str:
        return f"{self.name} (final thesis model)" if self.is_final else self.name

    @property
    def architecture(self) -> str:
        c = self.config
        return f"{c.Transformation}-{c.FeatureExtraction}-{c.SequenceModeling}-{c.Prediction}"

    @property
    def case_sensitive(self) -> bool:
        return any(char.isupper() for char in self.config.character)

    @property
    def characters(self) -> str:
        count = len(self.config.character)
        return f"{count}, case-sensitive" if self.case_sensitive else f"{count}, lower case and digits"


def _pretrained_config(path: Path) -> RecognizerConfig | None:
    """Settings of an original benchmark release, read from its file name."""
    stem = path.stem.lower()
    stages = _ALIASES.get(stem)
    if stages is None:
        match = _ARCHITECTURE.search(path.stem)
        if match is None:
            return None
        stages = match.groups()
    transformation, feature, sequence, prediction = (_STAGE_NAMES[stage.lower()] for stage in stages)
    return RecognizerConfig(
        Transformation=transformation,
        FeatureExtraction=feature,
        SequenceModeling=sequence,
        Prediction=prediction,
        batch_max_length=25,  # benchmark default
        character=SENSITIVE_CHARACTER_SET if "case-sensitive" in stem else ALPHANUMERIC,
        PAD=False,
        rgb=False,
    )


def discover_checkpoints(models_dir: Path = MODELS_DIR) -> list[Checkpoint]:
    """Fine-tuned folders (with opt.txt) and the pretrained benchmark files.

    The final thesis model comes first, then fine-tuned runs, then pretrained ones.
    """
    found: list[Checkpoint] = []
    if not models_dir.is_dir():
        return found
    for folder in sorted(path for path in models_dir.iterdir() if path.is_dir()):
        opt_path = folder / "opt.txt"
        weights = sorted(folder.glob("*.pth"), key=lambda path: (path.stem != "best_accuracy", path.stem))
        if not opt_path.exists() or not weights:
            continue
        config = RecognizerConfig.from_opt_file(opt_path)
        for weight in weights:
            found.append(Checkpoint(weight, f"{folder.name} / {weight.stem}", "Fine-tuned", config, "opt.txt"))
    for weight in sorted(models_dir.glob("*.pth")):
        if weight.name.lower().startswith("craft"):
            continue
        config = _pretrained_config(weight)
        if config is not None:
            found.append(Checkpoint(weight, weight.stem, "Pretrained", config, "file name"))
    found.sort(key=lambda checkpoint: (not checkpoint.is_final, checkpoint.group != "Fine-tuned", checkpoint.name.lower()))
    return found


# ------------------------------------------------------------------ images
def list_images(folder: Path) -> list[Path]:
    """Images directly inside ``folder``, sorted by name."""
    if not folder.is_dir():
        return []
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )


def image_folders(root: Path = PROJECT_ROOT) -> list[tuple[Path, int]]:
    """Known image folders with their image counts (empty ones are left out)."""
    candidates = sorted(path for path in (root / "images").glob("*") if path.is_dir())
    candidates += [root / "dataset_ingredients", root / "data" / "sample_inputs"]
    folders = []
    for folder in candidates:
        count = len(list_images(folder))
        if count:
            folders.append((folder, count))
    return folders


# ------------------------------------------------------------ ground truth
@dataclass(frozen=True)
class GroundTruthFile:
    path: Path
    entries: int


def image_ground_truth_files(data_dir: Path = DATA_DIR) -> list[GroundTruthFile]:
    """``gt*.txt`` files whose entries are whole images (several words per entry)."""
    files = []
    for path in sorted(data_dir.rglob("gt*.txt")):
        try:
            lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        except (OSError, UnicodeDecodeError):
            continue
        texts = [parts[1] for parts in (line.strip().split(maxsplit=1) for line in lines) if len(parts) == 2]
        if texts and sum(len(text.split()) for text in texts) / len(texts) >= 3:
            files.append(GroundTruthFile(path, len(texts)))
    return files


@dataclass(frozen=True)
class WordDataset:
    path: Path
    samples: int


def word_datasets(data_dir: Path = DATA_DIR) -> list[WordDataset]:
    """LMDB word-image datasets (folders with data.mdb), test splits first."""
    datasets = []
    for marker in sorted(data_dir.rglob("data.mdb")):
        folder = marker.parent
        try:
            env = lmdb.open(str(folder), readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin(write=False) as txn:
                count = int(txn.get(b"num-samples"))
            env.close()
        except Exception:  # noqa: BLE001 - an unreadable folder is simply not offered
            continue
        datasets.append(WordDataset(folder, count))
    datasets.sort(key=lambda dataset: ("test" not in dataset.path.name, "valid" not in dataset.path.name, dataset.path.name))
    return datasets
