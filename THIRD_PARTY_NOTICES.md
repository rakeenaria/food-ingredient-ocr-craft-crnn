# Third-party notices

This project builds on the following open-source work. Their licenses apply to the parts listed.

## deep-text-recognition-benchmark

- Source: <https://github.com/clovaai/deep-text-recognition-benchmark>
- Copyright (c) 2019-present NAVER Corp.
- License: Apache License 2.0
- Used for:
  - the recognizer architecture (`model.py`, `modules/`);
  - data loading and label conversion (`dataset.py`, `utils.py`);
  - the training and evaluation scripts;
  - the pretrained `TPS-ResNet-BiLSTM-Attn-case-sensitive.pth` checkpoint, which the thesis model was fine-tuned from.

## CRAFT: Character-Region Awareness For Text detection

- Source: <https://github.com/clovaai/CRAFT-pytorch>
- Copyright (c) 2019-present NAVER Corp.
- License: MIT License
- Used for:
  - the text detector (`CRAFT/`, `basenet/`, `refinenet.py`, `imgproc.py`);
  - the pretrained `craft_mlt_25k.pth` and `craft_refiner_CTW1500.pth` weights.

## Dataset

The recognizer was fine-tuned on word crops from the Kaggle dataset *Ingredients Image from Food Label*. Its terms are on the dataset page.
