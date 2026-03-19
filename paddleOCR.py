from paddleocr import PaddleOCR
from pathlib import Path
import json

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True)

# Run OCR inference on a sample image 
result = ocr.predict(
    input="bahan/")

# Visualize the results and save the JSON results
for res in result:
    res.print()
    res.save_to_img("paddle_results")
    res.save_to_json("paddle_results")

# Proses semua file JSON di folder paddle_results
paddle_results_dir = Path("paddle_results")
json_files = sorted(paddle_results_dir.glob("*_res.json"))

print(f"\n{'='*80}")
print(f"HASIL OCR - MERGE TEXT DARI SETIAP GAMBAR")
print(f"{'='*80}\n")

for json_file in json_files:
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ambil rec_texts dari JSON
        rec_texts = data.get('rec_texts', [])
        rec_scores = data.get('rec_scores', [])
        image_path = data.get('input_path', 'Unknown')
        
        # Merge semua teks dengan spasi
        merged_text = ' '.join(rec_texts)
        
        # Tampilkan di terminal
        print(f"File: {json_file.name}")
        print(f"Image: {image_path}")
        print(f"Jumlah baris terdeteksi: {len(rec_texts)}")
        print(f"\nMerged Text:")
        print(f"{merged_text}")
        print(f"\nDetail per baris:")
        for idx, (text, score) in enumerate(zip(rec_texts, rec_scores), 1):
            print(f"  Baris {idx}: {text} (score: {score:.4f})")
        print(f"\n{'-'*80}\n")
        
        # Simpan hasil merged ke file
        with open("paddle_results/merged_results.txt", "a", encoding="utf-8") as f:
            f.write(f"File: {json_file.name}\n")
            f.write(f"Image: {image_path}\n")
            f.write(f"Merged Text: {merged_text}\n\n")
    
    except Exception as e:
        print(f"Error processing {json_file}: {e}\n")

print(f"{'='*80}")
print(f"Hasil merge telah disimpan ke: paddle_results/merged_results.txt")
print(f"{'='*80}")