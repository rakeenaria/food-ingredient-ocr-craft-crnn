# Third-Party Notices

This project builds on open-source work from the following repositories.

## 1) CRAFT Text Detector
- Project: CRAFT-pytorch
- URL: https://github.com/clovaai/CRAFT-pytorch
- License: MIT License
- Usage in this repository: text detection component and related utilities.

## 2) Deep Text Recognition Benchmark (CRNN/TRBA)
- Project: deep-text-recognition-benchmark (fork/source used in this work)
- URL: https://github.com/zihaomu/deep-text-recognition-benchmark
- Original upstream reference: https://github.com/clovaai/deep-text-recognition-benchmark
- License: Apache License 2.0
- Usage in this repository: text recognition model architecture, training/inference utilities.

## 3) Model Weights Provenance
- CRAFT pretrained weights are obtained from CRAFT-pytorch resources.
- CRNN/TRBA pretrained weights are obtained from deep-text-recognition-benchmark resources.
- The fine-tuned model released with this repository is derived from TRBA case-sensitive pretrained weights.

## Notes
- Please follow each upstream project's license and attribution requirements when reusing code or weights.
- If any notice is inaccurate, open an issue so it can be corrected.
