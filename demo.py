import argparse
import os
import shutil
from pathlib import Path
import time
import numpy as np
import cv2 as cv
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import defaultdict

# CRAFT detection deps
from CRAFT.craft import CRAFT
import CRAFT.craft_utils as craft_utils
import imgproc

# Recognition deps from this repo
from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model


def order_points(pts): # Menata 4 titik jadi urutan konsisten/rapi (TL, TR, BR, BL).
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def quad_crop(image, box): # Memotong tulisan yg miring.
    pts = np.array(box, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 4:
        return None
    rect = order_points(pts[:4])
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))
    if maxW < 2 or maxH < 2:
        return None
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv.getPerspectiveTransform(rect, dst)
    return cv.warpPerspective(image, M, (maxW, maxH), flags=cv.INTER_CUBIC)

# menghitung seberapa mirip 2 kotak berdasarkan tumpang tindih axis-aligned mereka.
def iou_rect(a, b): 
    """
    IoU berbasis bounding box axis-aligned dari dua quadrilateral.
    a, b: ndarray/list shape (4,2)
    """
    a = np.array(a)
    b = np.array(b)
    ax0, ay0 = a[:,0].min(), a[:,1].min()
    ax1, ay1 = a[:,0].max(), a[:,1].max()
    bx0, by0 = b[:,0].min(), b[:,1].min()
    bx1, by1 = b[:,0].max(), b[:,1].max()
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

#mengurutkan kotak dalam urutan baca
def sort_boxes_reading_order(boxes, tol_factor=0.8, min_tol=10.0): 
    """
    Sort boxes top-to-bottom then left-to-right within each line.
    Menggunakan baseline (rata-rata dua titik y terbawah) seperti di pipeline CRAFT+CRNN asli.
    """
    if len(boxes) == 0:
        return []
    stats = []  # (idx, min_x, min_y, max_y, height, baseline_y)
    for i, box in enumerate(boxes):
        b = np.array(box)
        ys = b[:, 1]
        xs = b[:, 0]
        min_y, max_y = float(ys.min()), float(ys.max())
        min_x = float(xs.min())
        height = max_y - min_y
        ys_sorted = np.sort(ys)
        baseline_y = float(np.mean(ys_sorted[-2:])) if ys_sorted.shape[0] >= 2 else float(max_y)
        stats.append((i, min_x, min_y, max_y, height, baseline_y))

    median_h = float(np.median([s[4] for s in stats])) if stats else 0.0
    # Toleransi baris mengikuti implementasi asli: max(10px, 0.5 * median_h)
    line_tol = max(min_tol, 0.5 * median_h) if median_h > 0 else min_tol

    # sort by baseline_y to create lines
    stats.sort(key=lambda s: s[5])
    lines = []
    current = []
    current_top = None
    for s in stats:
        if current and current_top is not None and abs(s[5] - current_top) > line_tol:
            lines.append(current)
            current = [s]
            current_top = s[5]
        else:
            current.append(s)
            current_top = s[5] if current_top is None else (current_top * (len(current) - 1) + s[5]) / len(current)
    if current:
        lines.append(current)

    order = []
    for line in lines:
        line.sort(key=lambda s: s[1])  # left-to-right by min_x
        order.extend([s[0] for s in line])
    return order


def cluster_lines_with_text(items, tol_factor=0.8, min_tol=10.0): # Mengelompokkan (idx, teks, box) menjadi baris-baris berdasarkan posisi y, lalu mengurutkan kiri→kanan dalam baris.
    """
    items: list of tuples (idx, text, box)
    box: 4x2 array-like
    Returns list of lines, each line is list of (idx, text, box) in reading order.
    Menggunakan baseline (rata-rata dua titik y terbawah) dan toleransi max(10px, 0.5*median_h).
    """
    if not items:
        return []
    enriched = []
    for idx, txt, box in items:
        b = np.array(box)
        ys = b[:, 1]; xs = b[:, 0]
        min_y, max_y = float(ys.min()), float(ys.max())
        min_x = float(xs.min())
        height = max_y - min_y
        ys_sorted = np.sort(ys)
        baseline_y = float(np.mean(ys_sorted[-2:])) if ys_sorted.shape[0] >= 2 else float(max_y)
        enriched.append((idx, txt, box, min_x, baseline_y, height))

    median_h = float(np.median([e[5] for e in enriched])) if enriched else 0.0
    tol = max(min_tol, 0.5 * median_h) if median_h > 0 else min_tol

    enriched.sort(key=lambda e: e[4])  # by baseline_y
    lines = []
    current = []
    current_y = None
    for e in enriched:
        if current and current_y is not None and abs(e[4] - current_y) > tol:
            lines.append(current)
            current = [e]
            current_y = e[4]
        else:
            current.append(e)
            current_y = e[4] if current_y is None else (current_y * (len(current) - 1) + e[4]) / len(current)
    if current:
        lines.append(current)

    # sort each line left->right by min_x
    sorted_lines = []
    for line in lines:
        line.sort(key=lambda e: e[3])
        sorted_lines.append(line)
    return sorted_lines


def copy_state_dict(state_dict): #menghapus 'module.' dari kunci state_dict jika ada
    # strip 'module.' if present
    if len(state_dict) > 0 and list(state_dict.keys())[0].startswith('module.'):
        new = {}
        for k, v in state_dict.items():
            new[k[len('module.'):]] = v
        return new
    return state_dict


def load_craft(trained_model_path: str, use_cuda: bool): #Memuat model CRAFT dari path yang diberikan
    net = CRAFT()
    if use_cuda:
        net.load_state_dict(copy_state_dict(torch.load(trained_model_path)))
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    else:
        net.load_state_dict(copy_state_dict(torch.load(trained_model_path, map_location='cpu')))
    net.eval()
    return net


def craft_detect(net, image_bgr, args, refine_net=None): # Mendeteksi teks dalam gambar menggunakan model CRAFT
    # resize keeping aspect ratio as in official CRAFT
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(image_bgr, args.canvas_size, interpolation=cv.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # HWC -> CHW
    x = Variable(x.unsqueeze(0))
    if args.cuda:
        x = x.cuda()
    with torch.no_grad():
        y, feature = net(x)
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()
    # refine link if enabled
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()
    # post process
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, args.text_threshold, args.link_threshold, args.low_text, args.poly)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    try:
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    except Exception:
        # fallback: if polys has inconsistent shapes, reuse boxes
        polys = boxes
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]
    return boxes, polys, score_text, score_link


def build_recognizer(opt): #Menyiapkan converter label, membangun model recognizer (CRNN/TPS-ResNet-BiLSTM-Attn), memuat checkpoint, dan mengembalikannya dalam mode eval.
    # Build label converter
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Recognizer params', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    # robust load for state dicts with/without 'module.'
    print(f'Loading recognizer weights from {opt.saved_model}')
    state = torch.load(opt.saved_model, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    try:
        model.load_state_dict(state)
    except Exception:
        from collections import OrderedDict
        stripped = OrderedDict((k[7:], v) if k.startswith('module.') else (k, v) for k, v in state.items())
        model.load_state_dict(stripped, strict=False)
    model = torch.nn.DataParallel(model).to(device)
    model.eval()
    return model, converter, device


def recognize_folder(model, converter, device, crops_dir: Path, opt): #Memuat semua crop dari folder, menjalankan inferensi batch, dan mengembalikan daftar (path, prediksi teks). 
    align_collate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    dataset = RawDataset(root=str(crops_dir), opt=opt)
    if len(dataset) == 0:
        return []
    loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,
                                         num_workers=int(opt.workers), collate_fn=align_collate, pin_memory=True)
    results = []  # (path, text)
    with torch.no_grad():
        for image_tensors, image_path_list in loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, preds_size)
            else:
                preds = model(image, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)
                # strip after [s]
                preds_str = [s.split('[s]')[0] for s in preds_str]
            for pth, txt in zip(image_path_list, preds_str):
                results.append((pth, txt))
    return results


def main():
    ap = argparse.ArgumentParser()
    # Detection args
    ap.add_argument('--input_folder', default='./bahan', help='folder berisi gambar input (recursive)')
    ap.add_argument('--trained_model', default='saved_models/craft_mlt_25k.pth', help='CRAFT weight path')
    ap.add_argument('--text_threshold', type=float, default=0.7)
    ap.add_argument('--low_text', type=float, default=0.4)
    ap.add_argument('--link_threshold', type=float, default=0.4)
    ap.add_argument('--canvas_size', type=int, default=1280)
    ap.add_argument('--mag_ratio', type=float, default=1.5)
    ap.add_argument('--poly', action='store_true')
    ap.add_argument('--refine', action='store_true', help='aktifkan link refiner CRAFT')
    ap.add_argument('--refiner_model', default='saved_models/craft_refiner_CTW1500.pth', help='weight refiner CRAFT')
    ap.add_argument('--cuda', type=lambda v: str(v).lower() in ('1','true','yes','y'), default=torch.cuda.is_available())
    ap.add_argument('--crops_folder', default='./craft_crops', help='folder untuk menyimpan crop hasil deteksi')
    ap.add_argument('--results_folder', default='./craft_results', help='folder untuk menyimpan hasil (overlay/teks)')
    ap.add_argument('--save_overlay', action='store_true', help='simpan gambar overlay deteksi ke results_folder')
    ap.add_argument('--line_tol_factor', type=float, default=0.8, help='faktor pengelompokan baris (tol = faktor * tinggi_median, min 10px)')
    # Recognition args (mirror demo.py defaults)
    ap.add_argument('--saved_model', required=True, help='path ke model pengenalan (.pth)')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--batch_max_length', type=int, default=25)
    ap.add_argument('--imgH', type=int, default=32)
    ap.add_argument('--imgW', type=int, default=100)
    ap.add_argument('--rgb', action='store_true')
    ap.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
    ap.add_argument('--sensitive', action='store_true')
    ap.add_argument('--PAD', action='store_true')
    ap.add_argument('--Transformation', type=str, default='TPS')
    ap.add_argument('--FeatureExtraction', type=str, default='ResNet')
    ap.add_argument('--SequenceModeling', type=str, default='BiLSTM')
    ap.add_argument('--Prediction', type=str, default='Attn')
    ap.add_argument('--num_fiducial', type=int, default=20)
    ap.add_argument('--input_channel', type=int, default=1)
    ap.add_argument('--output_channel', type=int, default=512)
    ap.add_argument('--hidden_size', type=int, default=256)

    opt = ap.parse_args()

    # Persiapan folder output
    in_dir = Path(opt.input_folder)
    crops_dir = Path(opt.crops_folder)
    if crops_dir.exists():
        shutil.rmtree(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)
    line_crops_dir = crops_dir / "lines"
    line_crops_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(opt.results_folder); results_dir.mkdir(parents=True, exist_ok=True)
    opt.results_folder = results_dir  # gunakan Path untuk penulisan hasil

    # Build recognizer
    if opt.sensitive:
        import string
        opt.character = string.printable[:-6]
    rec_model, converter, device = build_recognizer(opt)

    # Load CRAFT
    net = load_craft(opt.trained_model, opt.cuda)
    refine_net = None
    if opt.refine:
        try:
            from refinenet import RefineNet
            refine_net = RefineNet()
            if opt.cuda:
                refine_net.load_state_dict(copy_state_dict(torch.load(opt.refiner_model)))
                refine_net = refine_net.cuda()
                refine_net = torch.nn.DataParallel(refine_net)
            else:
                refine_net.load_state_dict(copy_state_dict(torch.load(opt.refiner_model, map_location='cpu')))
            refine_net.eval()
            opt.poly = True  # enable polygon mode when refiner is used
            print(f'Loaded refiner from {opt.refiner_model}')
        except Exception as e:
            print(f'Warning: gagal memuat refiner: {e}')
            refine_net = None

    # Gather images
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    images = [p for p in in_dir.rglob('*') if p.suffix.lower() in exts]
    print(f'Found {len(images)} images in {in_dir}')

    total_crops = 0
    box_map = defaultdict(list)  # base -> list of (idx, box)
    for img_path in images:
        img = cv.imread(str(img_path))
        if img is None:
            print(f'skip (cannot read): {img_path}')
            continue
        # Deteksi dengan refiner (opsional)
        boxes_refine, polys_refine, score_text, score_link = craft_detect(net, img, opt, refine_net=refine_net)
        # simpan peta karakter (text) dan affinity (link) sebagai heatmap
        stem = img_path.stem
        text_map_img = imgproc.cvt2HeatmapImg(score_text)
        link_map_img = imgproc.cvt2HeatmapImg(score_link)
        cv.imwrite(str(opt.results_folder / f"{stem}_textmap.png"), text_map_img)
        cv.imwrite(str(opt.results_folder / f"{stem}_linkmap.png"), link_map_img)
        if boxes_refine is None or len(boxes_refine) == 0:
            print(f'{img_path} -> 0 boxes')
            continue
        # Deteksi plain untuk urutan baca stabil
        boxes_plain, _, _, _ = craft_detect(net, img, opt, refine_net=None)
        if boxes_plain is None or len(boxes_plain) == 0:
            boxes_plain = boxes_refine
        # optional: save overlay of detections
        if getattr(opt, 'save_overlay', False):
            overlay = img.copy()
            # draw polygons if available, else boxes
            to_draw = polys_refine if (polys_refine is not None and len(polys_refine)==len(boxes_refine)) else boxes_refine
            for poly in to_draw:
                pts = np.array(poly, dtype=np.int32).reshape(-1,1,2)
                cv.polylines(overlay, [pts], True, (0,255,0), 2)
            stem = img_path.stem
            cv.imwrite(str(results_dir / f'{stem}_overlay.jpg'), overlay)
        # sort boxes in reading order (atas->bawah, kiri->kanan dalam satu baris)
        order_plain = sort_boxes_reading_order(boxes_plain, tol_factor=getattr(opt, 'line_tol_factor', 0.8))
        # mapping urutan plain -> refine via IoU (hindari duplikasi)
        used = set()
        ordered_refine_idx = []
        for i in order_plain:
            br = boxes_plain[i]
            best_j, best_iou = None, 0.0
            for j, b in enumerate(boxes_refine):
                if j in used:
                    continue
                iou = iou_rect(br, b)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j is not None:
                used.add(best_j)
                ordered_refine_idx.append(best_j)
        if not ordered_refine_idx:
            ordered_refine_idx = list(range(len(boxes_refine)))
        stem = img_path.stem
        saved = []
        
        #crop per baris, lalu crop per kata dalam baris
        for line_rank, idx_ref in enumerate(ordered_refine_idx, start=1):
            line_crop = quad_crop(img, boxes_refine[idx_ref])
            if line_crop is None:
                continue
            # simpan crop per baris
            cv.imwrite(str(line_crops_dir / f'{stem}_line{line_rank}.png'), line_crop)
            # deteksi kata di dalam crop baris menggunakan CRAFT plain (tanpa refiner)
            boxes_word, _, _, _ = craft_detect(net, line_crop, opt, refine_net=None)
            if boxes_word is None or len(boxes_word) == 0:
                # fallback: simpan seluruh baris sebagai satu crop
                out_path = crops_dir / f'{stem}_line{line_rank}_word1.png'
                cv.imwrite(str(out_path), line_crop)
                saved.append(out_path)
                box_map[stem].append((line_rank * 1000 + 1, boxes_refine[idx_ref]))
                continue

            order_word = sort_boxes_reading_order(boxes_word, tol_factor=getattr(opt, 'line_tol_factor', 0.8))
            for word_rank, w_idx in enumerate(order_word, start=1):
                word_crop = quad_crop(line_crop, boxes_word[w_idx])
                if word_crop is None:
                    continue
                out_path = crops_dir / f'{stem}_line{line_rank}_word{word_rank}.png'
                cv.imwrite(str(out_path), word_crop)
                saved.append(out_path)
                # gunakan kunci gabungan agar tetap urut saat merge (line lebih signifikan)
                box_map[stem].append((line_rank * 1000 + word_rank, boxes_word[w_idx]))
        print(f'{img_path} -> saved {len(saved)} crops')
        total_crops += len(saved)
    print(f'Total crops written: {total_crops}')

    # Recognize all crops
    recs = recognize_folder(rec_model, converter, device, crops_dir, opt)
    # Write a simple results file
    out_txt = results_dir / 'recognized.txt'
    with open(out_txt, 'w', encoding='utf-8') as f:
        for pth, txt in recs:
            f.write(f'{pth}\t{txt}\n')
    print(f'Recognition results saved to {out_txt}')

    # Group per source image (merge dengan urutan baca)
    merged = {}
    grouped = defaultdict(list)
    for pth, txt in recs:
        stem = Path(pth).stem
        # abaikan file line-only (tanpa _word) agar tidak terhitung ganda saat merge
        if '_line' in stem and '_word' not in stem:
            continue
        if '_line' in stem:
            base, rest = stem.split('_line', 1)
            # rest expects format "<line>_word<word>"
            line_no = 0
            word_no = 0
            try:
                parts = rest.split('_word')
                line_no = int(parts[0])
                if len(parts) > 1:
                    word_no = int(parts[1])
            except Exception:
                pass
            idx = line_no * 1000 + word_no  # konsisten dengan box_map
        elif '_box_' in stem:
            base, idx = stem.rsplit('_box_', 1)
            try:
                idx = int(idx)
            except Exception:
                idx = 0
        else:
            base, idx = stem, 0
        grouped[base].append((idx, txt))

    merged_master = results_dir / 'merged.txt'
    with open(merged_master, 'w', encoding='utf-8') as f:
        for base, items in grouped.items():
            items.sort(key=lambda t: t[0])
            # urutkan berdasarkan idx numerik (line*1000 + word) agar sesuai posisi
            items.sort(key=lambda t: t[0])
            lines_dict = defaultdict(list)
            for idx, txt in items:
                line_no = idx // 1000
                lines_dict[line_no].append((idx, txt))
            line_texts = []
            for line_no in sorted(lines_dict.keys()):
                words = lines_dict[line_no]
                words.sort(key=lambda t: t[0])
                line_text = ' '.join(w[1] for w in words)
                line_texts.append(line_text)
                # simpan per-baris
                with open(results_dir / f'res_{base}_line{line_no}_merged.txt', 'w', encoding='utf-8') as lf:
                    lf.write(line_text + '\n')

            paragraph = '\n'.join(line_texts)
            merged[base] = paragraph
            # one file per image (line-separated)
            with open(results_dir / f'res_{base}_merged.txt', 'w', encoding='utf-8') as imgf:
                imgf.write('\n'.join(line_texts) + '\n')
            f.write(f'{base}\t{paragraph}\n\n')
            print(f'\n[{base}]: {paragraph}')
    print(f'Merged paragraphs saved to {merged_master}')


if __name__ == '__main__':
    main()
