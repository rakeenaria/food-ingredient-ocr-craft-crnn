# test_paddleocr.py - prediksi atau evaluasi dengan PaddleOCR (det+rec default)
import argparse
import time
from pathlib import Path
from paddleocr import PaddleOCR


def levenshtein(a, b):
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def read_gt(gt_path):
    gt = {}
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                pth, label = parts
                gt[Path(pth).stem] = label
    return gt


def collect_texts(ocr, img_path, conf_thr):
    res = ocr.ocr(str(img_path))
    texts, confs = [], []

    # handle OCRResult objects (paddleocr >=2.8 pipelines)
    if isinstance(res, list):
        for line in res:
            if isinstance(line, list):
                for det in line:
                    if hasattr(det, "text") and hasattr(det, "score"):  # OCRResult
                        text = det.text
                        score_f = float(det.score) if det.score is not None else 0.0
                        if score_f >= conf_thr:
                            texts.append(text)
                            confs.append(score_f)
                    elif len(det) >= 2 and isinstance(det[1], (list, tuple)):
                        text, score = det[1]
                        try:
                            score_f = float(score)
                        except Exception:
                            score_f = 0.0
                        if score_f >= conf_thr:
                            texts.append(text)
                            confs.append(score_f)

    if not texts:
        try:
            rec_only = ocr.text_recognizer(str(img_path))
            if isinstance(rec_only, list):
                for item in rec_only:
                    if hasattr(item, "text") and hasattr(item, "score"):
                        text = item.text
                        score_f = float(item.score) if item.score is not None else 0.0
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        text, score_f = item[0], float(item[1])
                    else:
                        continue
                    if score_f >= conf_thr:
                        texts.append(str(text))
                        confs.append(score_f)
        except Exception:
            pass

    merged = " ".join(texts)
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return merged, avg_conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder gambar (recursive)")
    ap.add_argument("--gt", help="opsional: gt.txt berisi 'path label'. Jika diisi, akan dihitung metrik.")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--conf_thr", type=float, default=0.8)
    ap.add_argument("--out", default="paddle_results/recognized.txt")
    args = ap.parse_args()

    ##ocr = PaddleOCR(lang=args.lang, use_textline_orientation=True)
    
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False)

    # Run OCR inference on a sample image 
    result = ocr.predict(
        input="bahan_inggris/")

    # Visualize the results and save the JSON results
    for res in result:
        res.print()
        res.save_to_img("output")
        res.save_to_json("output")

    imgs = [p for p in Path(args.images).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]
    imgs.sort()
    if not imgs:
        print("No images.")
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Prediksi saja
    if not args.gt:
        total_time = 0.0
        with open(args.out, "w", encoding="utf-8") as f:
            for img_path in imgs:
                t0 = time.time()
                pred, avg_conf = collect_texts(ocr, img_path, args.conf_thr)
                total_time += time.time() - t0
                f.write(f"{img_path}\t{pred}\t{avg_conf:.3f}\n")
                print(f"{img_path}: {pred} (conf {avg_conf:.3f})")
        print(f"Avg infer: {total_time / len(imgs) * 1000:.3f} ms/img")
        return

    # Evaluasi dengan GT
    gt = read_gt(args.gt)
    pairs = [(p, gt[p.stem]) for p in imgs if p.stem in gt]
    if not pairs:
        print("Tidak ada gambar yang cocok dengan GT (cek penamaan file).")
        return

    n_correct = 0
    norm_ED = 0.0
    cer_sum = 0
    cer_den = 0
    wer_sum = 0
    wer_den = 0
    conf_sum = 0
    conf_count = 0
    cov_sel = 0
    cov_correct = 0
    total_time = 0.0

    for img_path, gt_txt in pairs:
        t0 = time.time()
        pred, avg_conf = collect_texts(ocr, img_path, args.conf_thr)
        total_time += time.time() - t0

        conf_sum += avg_conf
        conf_count += 1
        if avg_conf >= args.conf_thr:
            cov_sel += 1
            if pred == gt_txt:
                cov_correct += 1
        if pred == gt_txt:
            n_correct += 1

        if len(gt_txt) == 0 or len(pred) == 0:
            norm_ED += 0
        elif len(gt_txt) > len(pred):
            norm_ED += 1 - levenshtein(pred, gt_txt) / len(gt_txt)
        else:
            norm_ED += 1 - levenshtein(pred, gt_txt) / len(pred)

        cer_sum += levenshtein(pred, gt_txt)
        cer_den += len(gt_txt)

        gt_words = gt_txt.split()
        pred_words = pred.split()
        wer_sum += levenshtein(pred_words, gt_words)
        wer_den += len(gt_words) if gt_words else 0

    length_of_data = len(pairs)
    accuracy = n_correct / length_of_data * 100
    norm_ED /= length_of_data
    cer = cer_sum / cer_den if cer_den > 0 else 0.0
    wer = wer_sum / wer_den if wer_den > 0 else 0.0
    avg_conf_all = conf_sum / conf_count if conf_count > 0 else 0.0
    coverage = cov_sel / length_of_data
    acc_at_thr = cov_correct / cov_sel if cov_sel > 0 else 0.0
    avg_ms = total_time / length_of_data * 1000

    print(f"Samples: {length_of_data}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"norm_ED: {norm_ED:.3f}")
    print(f"CER: {cer:.3f}")
    print(f"WER: {wer:.3f}")
    print(f"AvgConf: {avg_conf_all:.3f}")
    print(f"Coverage@{args.conf_thr}: {coverage:.3f}")
    print(f"Acc@{args.conf_thr}: {acc_at_thr:.3f}")
    print(f"Avg infer: {avg_ms:.3f} ms/img")


if __name__ == "__main__":
    main()
