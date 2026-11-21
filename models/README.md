# Models Folder

This folder stores model metadata and download guidance only.
Large checkpoints are intentionally not committed.

## Required Weights
- `saved_models/craft_mlt_25k.pth`
- `saved_models/craft_refiner_CTW1500.pth` (optional)
- `saved_models/FT_801010_off/best_accuracy.pth` (final thesis model, trained without PAD)

## Sources
- CRAFT: https://github.com/clovaai/CRAFT-pytorch
- TRBA baseline: https://github.com/zihaomu/deep-text-recognition-benchmark
- Project fine-tuned weights: https://drive.google.com/drive/folders/1MRPI5mAcX4nht2R4AO8VovIEsw2q6rGN?usp=drive_link
  - Use `best_accuracy.pth` ([direct link](https://drive.google.com/file/d/1wyBZXzxK36qW_Y4ge2wH5EMlpPTIANb1/view?usp=drive_link)). It is byte-identical (SHA-256) to `saved_models/FT_801010_off/best_accuracy.pth`.
