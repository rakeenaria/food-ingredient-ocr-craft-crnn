---
license: apache-2.0
language:
  - en
library_name: pytorch
pipeline_tag: image-to-text
tags:
  - ocr
  - scene-text-recognition
  - craft
  - crnn
  - food-labels
---

# Ingredient Facts: OCR weights for food ingredient labels

These are the weights behind [Ingredient Facts](https://ingredient-facts.streamlit.app), which reads the ingredient list on a food label from a photo. They come from the undergraduate thesis *Pengembangan Sistem OCR Berbasis CRAFT dan CRNN untuk Pengenalan Teks Daftar Bahan pada Label Makanan* by Rakeen Aria Alireza. The code is at [github.com/rakeenaria/food-ingredient-ocr-craft-crnn](https://github.com/rakeenaria/food-ingredient-ocr-craft-crnn).

## Files

| File | What it is | SHA-256 |
|---|---|---|
| `best_accuracy.pth` | Word recognizer: CRNN (TPS-ResNet-BiLSTM-Attn) fine-tuned on food-label word images (run `FT_801010_off`) | `405389e61a024a1d807b6438d3add77e1a50735da42cbbedafc72a34200710a6` |
| `opt.txt` | The recognizer's training settings: case-sensitive characters, grayscale 32 × 100 input, no PAD, maximum length 60 | `63540fcee6c9a2e01bcb88276f3aaf7ff5bee48b5dacffe7cc7402eba3d7c330` |
| `craft_mlt_25k.pth` | Text detector: CRAFT, the original pretrained weights from Clova AI, unchanged | `4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17` |

## Results

These are the thesis results.

- **Test split, word images:** 94.693% word accuracy, CER 0.015, WER 0.053. The split holds 10% of 6,081 curated word images (80:10:10).
- **The same split before fine-tuning:** 90.381%, using the pretrained `TPS-ResNet-BiLSTM-Attn-case-sensitive`.
- **Five real product photos, end to end (detection and recognition):** 92.248% word accuracy.

## Usage

Clone the GitHub repository and run this from its root:

```python
import cv2
from huggingface_hub import hf_hub_download

from src.ocr_engine import OCREngine

repo = "rakeenarya/ingredient-reader"
recognizer = hf_hub_download(repo, "best_accuracy.pth")
hf_hub_download(repo, "opt.txt")  # settings, read next to the checkpoint
detector = hf_hub_download(repo, "craft_mlt_25k.pth")

engine = OCREngine(recognizer, detector)
result = engine.process(cv2.imread("label.jpg"))
print(result.text)
```

## Training data

The recognizer was fine-tuned on word crops from the Kaggle dataset *Ingredients Image from Food Label*. The crops were curated into 6,081 word images.

## Limitations

- Printed English ingredient lists only. Handwriting and other languages are not supported.
- Small, blurry, glossy or curved text lowers accuracy.
- OCR can be wrong. Always check the physical label, especially if you have an allergy.

## Credits and licenses

- **Recognizer:** the architecture, the training code and the pretrained starting point come from [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) by Clova AI Research, NAVER (Apache 2.0).
- **Detector:** [CRAFT](https://github.com/clovaai/CRAFT-pytorch) by Clova AI Research, NAVER (MIT). The weights are unchanged.
