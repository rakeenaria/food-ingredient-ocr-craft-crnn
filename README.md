# OCR Daftar Bahan Makanan (CRAFT + CRNN)

Repository ini berisi implementasi sistem OCR untuk membaca teks daftar bahan pada label makanan kemasan.

Fokus repository ini adalah **menjalankan inference dengan `demo.py`** menggunakan kombinasi:
- CRAFT untuk deteksi teks.
- CRNN (TRBA: TPS-ResNet-BiLSTM-Attn) untuk pengenalan teks.

## Ringkasan Metode (Sesuai Paper)

Pipeline sistem di `demo.py`:
1. Input citra label makanan.
2. Deteksi teks menggunakan CRAFT (region map + affinity map).
3. Ekstraksi bertahap line-level lalu word-level untuk menjaga urutan baca.
4. Pengenalan kata dengan model CRNN TRBA.
5. Rekonstruksi hasil menjadi teks daftar bahan utuh.

## Ringkasan Dataset dan Fine-Tuning (Sesuai Paper)

- Sumber domain data: dataset label makanan berbahasa Inggris dari Kaggle.
- Data awal: 182 citra label.
- Word-image hasil ekstraksi anotasi: 6798.
- Setelah kurasi kualitas gambar/label: 6081.
- Split data: 80:20.
- Data latih: 4554.
- Data validasi: 1524.
- Fine-tuning dilakukan pada model TRBA pra-latih (case-sensitive), dengan target adaptasi domain teks ingredients.

## Ringkasan Hasil Evaluasi (Sesuai Paper)

### Evaluasi Internal (validasi)

| Model | Word Accuracy | CER | WER | Avg. Confidence |
|---|---:|---:|---:|---:|
| TRBA awal | 64.764% | 0.155 | 0.352 | 0.70 |
| Fine-tuned | 92.848% | 0.019 | 0.072 | 0.95 |

### Evaluasi Eksternal (10 citra dunia nyata)

| Model | NED | CER | WER | F1-score |
|---|---:|---:|---:|---:|
| TRBA awal | 0.924 | 0.082 | 0.221 | 0.793 |
| Fine-tuned | 0.970 | 0.028 | 0.046 | 0.953 |

Catatan tambahan dari paper:
- Total kata evaluasi eksternal: 613 kata.
- Total kesalahan turun dari 125 menjadi 24 (penurunan 80.8%).

## Prasyarat

- Python 3.9+ (disarankan)
- Dependensi:

```bash
pip install -r requirements-craft-crnn.txt
```

## Sumber Model dan Link Download

Karena file `.pth` besar, model tidak disimpan langsung di GitHub repository.

- Sumber CRAFT (`craft_mlt_25k.pth`, `craft_refiner_CTW1500.pth`):
  - https://github.com/clovaai/CRAFT-pytorch?tab=readme-ov-file#training
- Sumber CRNN/TRBA baseline (termasuk pretrained TRBA case-sensitive):
  - https://github.com/zihaomu/deep-text-recognition-benchmark?tab=readme-ov-file
- Link model fine-tuned proyek ini (`fine_tuned_model/best_accuracy.pth`):
  - https://drive.google.com/drive/u/0/folders/1MRPI5mAcX4nht2R4AO8VovIEsw2q6rGN

## Struktur Folder Minimal

```text
.
|- demo.py
|- bahan/                         # folder input gambar
|- saved_models/
|  |- craft_mlt_25k.pth
|  |- craft_refiner_CTW1500.pth   # opsional
|- fine_tuned_model/
|  |- best_accuracy.pth
```

## Menjalankan Inference (`demo.py`)

Perintah utama (konfigurasi TRBA case-sensitive sesuai hasil fine-tuning):

```bash
python demo.py --input_folder ./bahan --trained_model saved_models/craft_mlt_25k.pth --saved_model fine_tuned_model/best_accuracy.pth --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --batch_max_length 60 --sensitive --PAD
```

Jika ingin mengaktifkan CRAFT refiner:

```bash
python demo.py --input_folder ./bahan --trained_model saved_models/craft_mlt_25k.pth --refine --refiner_model saved_models/craft_refiner_CTW1500.pth --saved_model fine_tuned_model/best_accuracy.pth --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --batch_max_length 60 --sensitive --PAD
```

## Output Inference

Hasil akan tersimpan di:
- `craft_crops/`: crop line dan word.
- `craft_results/recognized.txt`: prediksi per crop.
- `craft_results/merged.txt`: hasil gabungan per gambar.
- `craft_results/res_<nama_gambar>_merged.txt`: hasil gabungan per file.
- `craft_results/*_textmap.png` dan `craft_results/*_linkmap.png`: heatmap CRAFT.

## Argumen Utama `demo.py`

- `--input_folder`: folder gambar input (recursive)
- `--trained_model`: path model CRAFT
- `--saved_model`: path model recognition
- `--refine`: aktifkan CRAFT refiner
- `--crops_folder`: folder output crop (default `craft_crops`)
- `--results_folder`: folder output hasil (default `craft_results`)
- `--cuda true/false`: pakai GPU atau CPU

## Catatan

- README ini difokuskan pada inference (`demo.py`), bukan langkah training ulang.
- Konfigurasi command mengacu ke `fine_tuned_model/opt.txt` pada repository ini.

## Reproducibility

Environment used for experiments in this project:
- Python: 3.x (recommended 3.9+)
- Framework: PyTorch
- OCR pipeline: CRAFT + TRBA (TPS-ResNet-BiLSTM-Attn)
- Inference entry point: `demo.py`

For exact model-specific settings, see `fine_tuned_model/opt.txt`.

## Known Limitations

- Teks yang sangat blur, resolusi sangat rendah, atau glare/reflection kuat masih dapat menurunkan akurasi.
- Layout yang sangat padat dan overlap antar kata bisa memengaruhi proses line/word grouping.
- Model saat ini ditujukan untuk domain label bahan makanan berbahasa Inggris; performa pada domain lain bisa menurun.

## Citation

Jika repository ini membantu riset Anda, silakan sitasi paper berikut (update metadata saat paper sudah final):

```bibtex
@article{alireza2026craftcrnn,
  title   = {Pengembangan Sistem OCR untuk Pengenalan Teks Daftar Bahan pada Label Makanan Berbasis CRAFT dan CRNN},
  author  = {Alireza, Rakeen Aria and Kurniawardhani, Arrie},
  journal = {JPIT},
  year    = {2026},
  note    = {Manuscript under review}
}
```
