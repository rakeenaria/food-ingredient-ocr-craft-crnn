# Food Ingredient OCR (CRAFT + CRNN TRBA)

## Overview
This project builds an end-to-end OCR pipeline to read food ingredient lists from packaging images.

Why this matters:
- Ingredient text is often small, dense, and hard to parse manually.
- Better OCR helps users quickly identify risky ingredients for allergy awareness.
- The same pipeline can be extended to halal/non-halal ingredient screening.

## Real-World Impact
- **Ingredient visibility**: converts printed labels into searchable digital text.
- **Allergy awareness**: helps flag potential allergens from recognized ingredients.
- **Halal extension path**: OCR output can be connected to rule-based or ML halal screening.

## Demo
### Quick Demo Flow
| Stage | Asset |
|---|---|
| Input image | `<img width="438" height="327" alt="bahan_4" src="https://github.com/user-attachments/assets/8682ea93-c2e2-4bce-a021-4b023f7637bc" />|
| CRAFT detected regions | ![bahan_4_overlay](https://github.com/user-attachments/assets/f5c6aa70-b1c4-46b5-9707-8b34d6b25095)|
| Final OCR reconstruction | BEST INGREDIENTS: Organic Whole Grain Wheat Flour (organic graham flour) Organic Wheat Flour, Organic Cane Sugar, Organic Expeller-pressed Sunflower Oil, Organic Honey, Organic Molasses, Leavening (baking soda, ammonium bicarbonate and cream of tartar), Organic Vanilla Flavor, Organic Brown Sugar Flavor, Sea Salt, Organic Rosemary Extract (to protect flavor). CONTAINS WHEAT INGREDIENTS. Made on shared equipment that also processes milk and soy.|

## Pipeline
`Image -> CRAFT -> Word Cropping -> CRNN (TRBA) -> Text Reconstruction`

1. Detect text regions with CRAFT.
2. Crop line regions, then crop word regions in reading order.
3. Recognize each word with TRBA (TPS-ResNet-BiLSTM-Attn).
4. Reconstruct final multi-line ingredient text.

## How It Works (Detailed)
### Why CRAFT for detection
CRAFT predicts character regions and character affinity maps, which is effective for compact, irregular label text and varying spacing.

### Why CRNN TRBA for recognition
TRBA combines:
- **TPS** for geometric normalization,
- **ResNet** for visual feature extraction,
- **BiLSTM** for sequence context,
- **Attention decoder** for character sequence generation.

### How fine-tuning improves performance
Fine-tuning on food-label domain data adapts the recognizer to ingredient vocabulary, punctuation patterns, and capitalization style common in packaging.

### Challenges in food label OCR
- Tiny fonts and dense layouts.
- Blur, glare, and low-contrast print.
- Mixed punctuation/capitalization and long ingredient sequences.

## Model Architecture
### CRAFT (Detection)
- Input image is resized and normalized.
- Network outputs text score map + link score map.
- Post-processing groups characters into word/line boxes.

### CRNN TRBA (Recognition)
- Per-word crops are normalized to fixed size.
- TRBA predicts character sequence per crop.
- Predictions are merged back by line and word order.

## Results
### Internal Evaluation (Validation)
| Model | Word Accuracy | CER | WER | Avg. Confidence |
|---|---:|---:|---:|---:|
| TRBA baseline | 64.764% | 0.155 | 0.352 | 0.70 |
| Fine-tuned | 92.848% | 0.019 | 0.072 | 0.95 |

### External Evaluation (10 Real-World Images)
| Model | NED | CER | WER | F1-score |
|---|---:|---:|---:|---:|
| TRBA baseline | 0.924 | 0.082 | 0.221 | 0.793 |
| Fine-tuned | 0.970 | 0.028 | 0.046 | 0.953 |

### Before vs After OCR (Sample Format)
```text
Input (ground truth): INGREDIENTS: CHICKEN BROTH: WATER CHICKEN FLAVOR (CHICKEN BROTH RENDERED CHICKEN FAT, SALT, FLAVOR, DISODIUM INOSINATE & DISODIUM GUANYLATE, POLYSORBATE 60, XANTHAN GUM), WHEAT FLOUR ENRICHED WITH (NIACIN, IRON, THIAMINE MONONITRATE, RIBOFLAVIN, FOLIC ACID), COOKED WHITE MEAT CHICKEN, INTERESTERIFIED SOYBEAN OIL, HYDROGENATED SOYBEAN OIL, CARROTS, PEAS, CONTAINS 2% OR LESS OF EACH OF: CELERY, MODIFIED CORN STARCH, ONIONS, HEAVY WHIPPING CREAM, SALT, RENDERED CHICKEN FAT, SOYBEAN OIL, WHEY POWDER, NONFAT DRY MILK, SUGAR, ISOLATED SOY PROTEIN, CARRAGEENAN, METHYLCELLULOSE, AUTOLYZED YEAST EXTRACT, FLAVORS, MONO- & DIGLYCERIDES, BLACK PEPPER, TURMERIC, CARAMEL COLOR, SOY LECITHIN. CONTAINS: MILK, SOY, WHEAT.
Baseline OCR        : INGREDIENTS: CHICKEN BROTH: WATER CHICKEN FLAVOR (CHICKEN BROTH RENDERED CHICKEN FAT, SALT," FLAVOR, DISODIUM INOSINATE but DISODIUM GUANYLATE, POLYSORBATE 6 XANTHAN GUM), WHEAT FLOUR ENRICHED WITHY (NIACIN, IRON, THIAMINE MONONITRATE, RIBOFLAVIN, FOLICH ACID)." COOKED WHITE MEATE CHICKEN, INTERESTERIFIED SOYBEAN OIL, HYDROGENATED SOYBEAN I CARROTS, PEAS, CONTAINS 2% ORRISTION LESSIN I EACHIN Of: CELERY, MODIFIED CORNIES STARCH, ONIONS, HEAVY WHIPPING CREAM, SALT, RENDERED CHICKEN FAT, SOYBEAN OIL, WHEY POWDER NONFAT DRYWE MILK, SUGAR, ISOLATED SOYER PROTEIN, CARRAGEENAN, METHYLCELLULOSE, AUTOLYZED YEAST EXTRACT, FLAVORS, MONO-T & DIGLYCERIDES, BLACK PEPPER, TURMERIC, CARAMEL COLOR, SOY-IE LECITHIN.] CONTAINS: MILK, SOY, WHEAT.
Fine-tuned OCR      : INGREDIENTS: CHICKEN BROTH: WATER CHICKEN FLAVOR (CHICKEN BROTH RENDERED CHICKEN FAT, SALT, FLAVOR, DISODIUM INOSINATE & DISODIUM GUANYLATE, POLYSORBATE 60, XANTHAN GUM), WHEAT FLOUR ENRICHED WITHL (NIACIN, IRON, THIAMINE MONONITRATE, RIBOFLAVIN, FOLIC ACID), COOKED WHITE MEAT CHICKEN, INTERESTERIFIED SOYBEAN OIL, HYDROGENATED SOYBEAN OIL, CARROTS, PEAS, CONTAINS 2% OR LESS OF EACH OF: CELERY, MODIFIED CORN STARCH, ONIONS, HEAVY WHIPPING CREAM, SALT, RENDERED CHICKEN FAT, SOYBEAN OIL, WHEY POWDER, NONFAT DRY MILK, SUGAR, ISOLATED SOY PROTEIN, CARRAGEENAN, METHYLCELLULOSE, AUTOLYZED YEAST EXTRACT, FLAVORS, MONO- & DIGLYCERIDES, BLACK PEPPER, TURMERIC, CARAMEL COLOR, SOY LECITHIN. CONTAINS: MILK, SOY, WHEAT.
```

### Sample Output Format
`craft_results/recognized.txt`
```text
craft_crops/bahan_1_line1_word1.png\tINGREDIENTS
craft_crops/bahan_1_line1_word2.png\tSUGAR
```

`craft_results/merged.txt`
```text
sample_image\tINGREDIENTS SUGAR PALM OIL COCOA POWDER SALT
```

## Tech Stack
- Python
- PyTorch
- OpenCV
- NumPy
- Pandas

## How to Run
1. Install dependencies:
```bash
pip install -r requirements-craft-crnn.txt
```

2. Download required weights (not stored in repo):
- CRAFT: https://github.com/clovaai/CRAFT-pytorch
- TRBA baseline: https://github.com/zihaomu/deep-text-recognition-benchmark
- Fine-tuned checkpoint: see project links in `models/README.md`

3. Run inference (legacy-compatible command):
```bash
python demo.py \
  --input_folder ./bahan \
  --trained_model saved_models/craft_mlt_25k.pth \
  --saved_model fine_tuned_model/best_accuracy.pth \
  --Transformation TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM \
  --Prediction Attn \
  --batch_max_length 60 \
  --sensitive \
  --PAD
```

4. Optional canonical script entrypoint:
```bash
python scripts/run_demo.py --saved_model fine_tuned_model/best_accuracy.pth
```

5. Outputs are written to:
- `outputs/crops/`
- `outputs/craft/`

## Project Structure
```text
.
|- src/
|  |- ocr_pipeline.py                # canonical OCR pipeline implementation
|- scripts/
|  |- run_demo.py                    # canonical runnable entrypoint
|- demo.py                           # compatibility wrapper (kept for existing usage)
|- legacy_demo_variant.py            # renamed legacy runner (old demo3.py)
|- legacy_demo_simple.py             # renamed legacy runner (old demo4.py)
|- demo3.py                          # compatibility wrapper -> legacy_demo_variant.py
|- demo4.py                          # compatibility wrapper -> legacy_demo_simple.py
|- data/
|  |- sample_inputs/                 # canonical sample-input location (documented)
|  |- valid/gt.txt                   # lightweight annotation sample
|- models/
|  |- README.md                      # weight provenance and download guidance
|- notebooks/
|  |- demo.ipynb                     # exploratory notebook
|- outputs/
|  |- crops/                         # cropped regions from detection
|  |- craft/                         # OCR text + heatmaps + merged output
|  |- examples/                      # recruiter-facing demo placeholders
|- THIRD_PARTY_NOTICES.md
|- LICENSE
```

Compatibility note:
- Existing `python demo.py ...` usage is preserved.
- Legacy output paths can still be used by passing `--crops_folder` and `--results_folder`.

## Author / Contribution
### Author
Rakeen Aria Alireza

### Key Contributions
- Designed and implemented the CRAFT + CRNN OCR pipeline for food ingredient labels.
- Performed preprocessing and labeling workflow for 6000+ word-image samples.
- Fine-tuned CRNN TRBA model from a pretrained case-sensitive checkpoint.
- Evaluated model quality using Word Accuracy, CER, WER, NED, and F1-score.

## License and Third-Party
- Project license: `LICENSE`
- Third-party attributions: `THIRD_PARTY_NOTICES.md`
