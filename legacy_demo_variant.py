import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.transforms as transforms
from craft import CRAFT

from PIL import Image
import math
import cv2 as cv
from skimage import io
import numpy as np
import craft_utils 
import imgproc
import file_utils
import json
import zipfile
from collections import OrderedDict

#import crnn related
import CRNN.crnn as crnn
import crnn_utils

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# for the arguments use
def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=384, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=True, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='./test_data/bahan/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
parser.add_argument('--output_folder', default='./test_results/eng/', type=str, help='output path')
parser.add_argument('--crnn_weights', default='./weights/crnn.pth', type=str, help='Where is the crnn model weights')
parser.add_argument('--crop_mode', default='center', choices=['center','full_width','none'], help='Cropping strategy: center=384x384 around finger, full_width=use full image width, none=use full image without crop')
parser.add_argument('--alphabet', default='0123456789abcdefghijklmnopqrstuvwxyz', type=str, help='Alphabet for CRNN (without CTC blank). Add punctuation here to enable them, e.g. 0123456789abcdefghijklmnopqrstuvwxyz:[]().,!?-')
parser.add_argument('--preproc', default='', type=str, help='Comma-separated preprocessing steps for CRNN crops: clahe,binarize,sharpen,denoise,gamma')
parser.add_argument('--binarize', default='otsu', choices=['otsu','adaptive'], help='Binarization method when enabled')
parser.add_argument('--denoise', default='none', choices=['none','median3','bilateral'], help='Denoise method when enabled')
parser.add_argument('--gamma', default=1.0, type=float, help='Gamma correction value (1.0 disables)')
parser.add_argument('--clahe_clip', default=2.0, type=float, help='CLAHE clip limit')
parser.add_argument('--clahe_tile', default=8, type=int, help='CLAHE tile grid size (tile x tile)')
parser.add_argument('--sharpen', default=0.0, type=float, help='Unsharp mask strength (0 disables)')
parser.add_argument('--use_finger', default=False, type=str2bool, help='Use finger points from JSON; if False, JSON is optional and image center is used')

args = parser.parse_args()

""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = args.output_folder
if not os.path.isdir(result_folder):
    os.makedirs(result_folder, exist_ok=True)

class ReaderData(object):
    def __init__(self, DataDir, use_finger=True):
        self.RootDir = DataDir
        self.AllFiles = os.listdir(self.RootDir)
        self.ImgExt = '.jpg'  # kept for backward compatibility
        self.JsonExt = '.json'
        self.use_finger = use_finger
        # Build pairs: only names that have both an image and a json
        allowed_img_exts = {'.jpg', '.jpeg', '.png', '.gif', '.pgm'}
        images = {}
        jsons = {}
        for fname in self.AllFiles:
            stem, ext = os.path.splitext(fname)
            ext = ext.lower()
            fpath = os.path.join(self.RootDir, fname)
            if ext in allowed_img_exts:
                images[stem] = fpath
            elif ext == '.json':
                jsons[stem] = fpath
        self.FileNames = sorted(images.keys())
        self.Pairs = []
        for name in self.FileNames:
            img_path = images[name]
            json_path = jsons.get(name)
            if use_finger and json_path is None:
                print(f"Warning: finger JSON not found for {name}, using image center.")
            self.Pairs.append((name, img_path, json_path))
        self.crop_height = 384 # default
        self.crop_width = 384  # default

    def setCropParas(self, w, h):
        self.crop_width = w
        self.crop_height = h

    def loadImage(self, src_path):
        img = io.imread(src_path)
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
        return img

    def saveCropImage(self, tgt_path, crop_img):
        cv.imwrite(tgt_path, crop_img)

    def findSize(self, img):
        return img.shape[0], img.shape[1]

    def loadFinger(self, json_dir):
        # input the json file name, return the array of finger's position
        with open (json_dir, "r") as load_f:
            finger_dict = json.load(load_f)
            index = 0
            for idx , item in enumerate(finger_dict['shapes']):
                if "finger" in item['label']:
                    index = idx
                    break
                else:
                    continue
            finger_pos = finger_dict['shapes'][index]['points'][0]
        load_f.close()
        return finger_pos

    def cropImage(self, img_path, json_path):
        t0 = time.time()
        img = self.loadImage(img_path)
        ori_h, ori_w = self.findSize(img)
        print("Loading the Original image:{}".format(time.time() - t0))

        t0 = time.time()
        # Determine finger position; if JSON unavailable or disabled, use image center
        if (not self.use_finger) or json_path is None:
            fin = [ori_w / 2, ori_h / 2]
        else:
            try:
                fin = self.loadFinger(json_path)
            except Exception as e:
                print(f"Warning: failed to parse finger from {json_path} ({e}). Using image center.")
                fin = [ori_w / 2, ori_h / 2]
        print("Loading the finger position:{}".format(time.time() - t0))

        t0 = time.time()
        mode = getattr(args, 'crop_mode', 'center')
        if mode == 'none':
            print("Cropping mode: none (using full image)")
            return img, [fin[0], fin[1]]
        elif mode == 'full_width':
            print("Cropping mode: full_width (use full image width and height)")
            # Use the full image without cropping to avoid losing edge text
            top = 0
            bottom = ori_h
            left = 0
            right = ori_w
            crop = img[top:bottom, left:right]
            # finger position in cropped coords (same as original here)
            crop_finger_pos_w = fin[0] - left
            crop_finger_pos_h = fin[1] - top
            print("Croping the image:{}".format(time.time() - t0))
            return crop, [crop_finger_pos_w, crop_finger_pos_h]
        else:
            print("Cropping mode: center (384x384 around finger)")
            margin_h = int(self.crop_height/2)
            margin_w = int(self.crop_width/2)
            rangeCrop = {"left":0,"right":0,"low":0,"up":0}
            # crop the image with finger at the center
            rangeCrop['left'] = fin[0] - margin_w if (fin[0] - margin_w) > 0 else 0
            rangeCrop['right'] = fin[0] + margin_w if (fin[0] + margin_w) < ori_w else ori_w
            rangeCrop['low'] = fin[1] - margin_h if fin[1] - margin_h > 0 else 0
            rangeCrop['up'] = fin[1] + margin_h if (fin[1] + margin_h) < ori_h else ori_h

            crop = img[rangeCrop['low'] : rangeCrop['up'], rangeCrop['left'] : rangeCrop['right']]
            crop = cv.resize(crop, (self.crop_width, self.crop_height))
            crop_finger_pos_w = margin_w
            crop_finger_pos_h = margin_h
            print("Croping the image:{}".format(time.time() - t0))
            return crop, [crop_finger_pos_w, crop_finger_pos_h]

class resizeNormalize(object):
    # this is used for the crnn's data preprocessing
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class resizeNormalizeKeepRatio(object):
    # resize keeping aspect ratio: set height, adjust width accordingly
    def __init__(self, height=32, min_width=16, max_width=512, interpolation=Image.BILINEAR):
        self.height = int(height)
        self.min_width = int(min_width)
        self.max_width = int(max_width)
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        w, h = img.size
        if h == 0:
            h = 1
        new_h = self.height
        new_w = int(round(w * (new_h / float(h))))
        new_w = max(self.min_width, min(self.max_width, new_w))
        img = img.resize((new_w, new_h), self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def preprocess_for_crnn(img_np, args, name=None, rank=None):
    # img_np: numpy array HxW or HxWxC (crop), returns grayscale numpy uint8 after preprocessing
    import cv2 as cv
    import numpy as np

    steps = set([s.strip().lower() for s in args.preproc.split(',') if s.strip()])
    # convert to gray
    if img_np.ndim == 3:
        if img_np.shape[2] == 3:
            gray = cv.cvtColor(img_np, cv.COLOR_BGR2GRAY)
        elif img_np.shape[2] == 4:
            gray = cv.cvtColor(img_np, cv.COLOR_BGRA2GRAY)
        else:
            gray = cv.cvtColor(img_np, cv.COLOR_BGR2GRAY)
    else:
        gray = img_np.copy()

    gray = gray.astype(np.uint8)

    # gamma
    if 'gamma' in steps and args.gamma and abs(args.gamma - 1.0) > 1e-3:
        inv = 1.0 / max(args.gamma, 1e-6)
        table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(256)]).astype("uint8")
        gray = cv.LUT(gray, table)

    # denoise
    if 'denoise' in steps and args.denoise != 'none':
        if args.denoise == 'median3':
            gray = cv.medianBlur(gray, 3)
        elif args.denoise == 'bilateral':
            gray = cv.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # clahe
    if 'clahe' in steps:
        clahe = cv.createCLAHE(clipLimit=float(args.clahe_clip), tileGridSize=(int(args.clahe_tile), int(args.clahe_tile)))
        gray = clahe.apply(gray)

    # binarize
    if 'binarize' in steps:
        if args.binarize == 'adaptive':
            gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 10)
        else:
            _, gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # sharpen
    if 'sharpen' in steps and args.sharpen and args.sharpen > 0.0:
        blur = cv.GaussianBlur(gray, (0, 0), 1.0)
        alpha = float(args.sharpen)
        gray = cv.addWeighted(gray, 1.0 + alpha, blur, -alpha, 0)

    return gray

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    #print("Input:{}\nInput shape:{}".format(x, x.shape))
    with torch.no_grad():
        y, feature = net(x)

    #print("Inference output:\n{}\nOutput shape:{}".format(y, y.shape))

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

#    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

def findNearest(finger_pos, bboxes):
    def e2_distance(p1, p2):
        import math
        # p1, p2 are both in list format
        return math.sqrt(math.pow((p1[0]-p2[0]), 2) + math.pow((p1[1]-p2[1]), 2))

    """
    input:  finger's position
            a list of boxes' coordinates
    outputs: only one box's coordinates, which is the nearest one to the finger's position
    """
    min_dis = 543 # this the dialog's length of 384*384 image's which is the longest length in the image
    nearest_id = 0
    #fin_pos = np.array(finger_pos)
    for idx, item in enumerate(bboxes):
        item_center = item.mean(axis = 0)
        #print("++++{}".format(item_center))
        item_center = item_center.tolist()
        dist = e2_distance(finger_pos, item_center)
        #print("Distance:{}".format(dist))
        if dist < min_dis:
            nearest_id = idx
            min_dis = dist
    #print("The nearest position id: {}".format(nearest_id))
    return nearest_id

if __name__ == '__main__':
    
    # load net of craft
    net = CRAFT()     # initialize
    print('Loading CRAFT weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    net.eval()

    # load the crnn related model, parameters
    model_path = args.crnn_weights
    # Build CRNN; default nclass from provided alphabet (CTC: +1 for blank)
    alphabet = args.alphabet
    nclass = len(alphabet) + 1
    crnn_net = crnn.CRNN(32, 1, nclass, 256)
    if torch.cuda.is_available():
        crnn_net = crnn_net.cuda()
    print('Loading CRNN weights from checkpoint (' + model_path + ')')
    # Robust loader for PyTorch 2.6+ (weights_only default) and legacy checkpoints
    def _torch_load_compat(path, map_location=None):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    map_loc = None if torch.cuda.is_available() else 'cpu'
    ckpt = _torch_load_compat(model_path, map_location=map_loc)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    # strip possible 'module.' prefix from DataParallel
    if isinstance(state_dict, dict) and len(state_dict) > 0:
        first_key = next(iter(state_dict))
        if first_key.startswith('module.'):
            new_state = {}
            for k, v in state_dict.items():
                new_state[k[len('module.'):]] = v
            state_dict = new_state
    # Try loading checkpoint; if classification layer shape mismatches (likely different nclass),
    # attempt to infer nclass from checkpoint and rebuild CRNN, else fall back to partial load.
    def _infer_nclass_from_state_dict(sd: dict):
        # typical key for last embedding layer in our CRNN: 'rnn.1.embedding.weight'
        candidates = [
            'rnn.1.embedding.weight',
            'module.rnn.1.embedding.weight',
        ]
        for k in sd.keys():
            if k.endswith('rnn.1.embedding.weight'):
                return int(sd[k].shape[0])
        for k in candidates:
            if k in sd:
                return int(sd[k].shape[0])
        # fallback: find any '.embedding.weight'
        for k, v in sd.items():
            if k.endswith('embedding.weight') and len(v.shape) == 2:
                return int(v.shape[0])
        return None

    try:
        crnn_net.load_state_dict(state_dict)
    except Exception:
        inferred = _infer_nclass_from_state_dict(state_dict)
        if inferred is not None and inferred != nclass:
            print(f"Info: rebuilding CRNN to match checkpoint classes (nclass={inferred}).")
            crnn_net = crnn.CRNN(32, 1, inferred, 256)
            if torch.cuda.is_available():
                crnn_net = crnn_net.cuda()
            crnn_net.load_state_dict(state_dict, strict=True)
            if inferred != len(alphabet) + 1:
                print("Warning: checkpoint nclass != alphabet length + 1. Decoding may be wrong; set --alphabet to the exact training alphabet.")
        else:
            print('Warning: partial load for CRNN due to shape mismatch (likely alphabet change).')
            model_dict = crnn_net.state_dict()
            filtered = {}
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered[k] = v
            model_dict.update(filtered)
            crnn_net.load_state_dict(model_dict, strict=False)

    converter = crnn_utils.strLabelConverter(alphabet)
    # revert to fixed-size resize (original behavior)
    transformer = resizeNormalize((100, 32))
    crnn_net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True
    
    # test the dataset's function
    data_root = args.test_folder
    if not os.path.isdir(data_root):
        print(f"Test folder not found: {data_root}")
        print("Please set --test_folder to a valid directory containing <name>.jpg and <name>.json pairs.")
        sys.exit(1)
    data = ReaderData(data_root, use_finger=args.use_finger)
    if len(data.FileNames) == 0:
        print(f"No image files found in {data_root}.")
        sys.exit(0)
    # iterate over valid (name, image_path, json_path) pairs
    for name, i_path, j_path in data.Pairs:
        print("-------------------{}---------------------".format(name))
        #crop_save_path = os.path.join(data.RootDir, ('crop_'+item+data.ImgExt))
        #t0 = time.time()
        crop, finger_new_pos = data.cropImage(i_path, j_path)
        # do the text detection:
        bboxes, polys, score_text = test_net(net, crop, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
        #print("bboxes:\n{}, type:\n{}, \ntotal:\t{}".format(bboxes, type(bboxes), len(bboxes)))

        # Save heatmap and detection overlay for reference
        mask_file = result_folder + "/res_" + name + '_mask.jpg'
        cv.imwrite(mask_file, score_text)
        file_utils.saveResult(os.path.basename(i_path), crop[:,:,::-1], polys, dirname = result_folder)

        # Recognize ALL detected boxes in reading order (top-to-bottom, left-to-right)
        if len(bboxes) == 0:
            print("No text boxes detected, skipping CRNN recognition for this item.")
            continue

        # Save heatmap and detection overlay for reference
        mask_file = result_folder + "/res_" + name + '_mask.jpg'
        cv.imwrite(mask_file, score_text)
        file_utils.saveResult(os.path.basename(i_path), crop[:,:,::-1], polys, dirname = result_folder)

        # group by baseline (average of two lowest y), then left-to-right within each line
        # compute stats per box
        stats = []  # (idx, min_x, min_y, max_x, max_y, baseline_y, height)
        for i, box in enumerate(bboxes):
            ys = box[:, 1]
            xs = box[:, 0]
            min_y = float(ys.min()); max_y = float(ys.max())
            min_x = float(xs.min()); max_x = float(xs.max())
            # baseline as average of two lowest y-values (more robust to ascenders)
            ys_sorted = np.sort(ys)
            baseline_y = float(np.mean(ys_sorted[-2:])) if ys_sorted.shape[0] >= 2 else float(max_y)
            height = max_y - min_y
            stats.append((i, min_x, min_y, max_x, max_y, baseline_y, height))

        if len(stats) == 0:
            print("No valid boxes to process.")
            continue

        median_h = float(np.median([s[6] for s in stats]))
        line_tol = max(10.0, 0.5 * median_h)

        # sort by baseline and cluster into lines
        stats.sort(key=lambda s: s[5])
        lines = []
        current = []
        current_base = None
        for s in stats:
            if current and current_base is not None and abs(s[5] - current_base) > line_tol:
                lines.append(current)
                current = [s]
                current_base = s[5]
            else:
                current.append(s)
                if current_base is None:
                    current_base = s[5]
                else:
                    current_base = (current_base * (len(current) - 1) + s[5]) / len(current)
        if current:
            lines.append(current)

        recognized_all = []
        lines_out = []
        rank = 0
        pad = 2
        for li, line in enumerate(lines):
            line.sort(key=lambda s: s[1])  # left-to-right by min_x
            for s in line:
                idx, min_x, min_y, max_x, max_y = s[0], s[1], s[2], s[3], s[4]
                H, W = crop.shape[0], crop.shape[1]
                h_0 = max(0, int(min_y) - pad)
                h_1 = min(H, int(max_y) + pad)
                w_0 = max(0, int(min_x) - pad)
                w_1 = min(W, int(max_x) + pad)
                if h_1 <= h_0 or w_1 <= w_0:
                    continue
                text_crop_np = crop[h_0:h_1, w_0:w_1]
                # save cropped area
                text_area_path = f'text_area_{name}_{rank}.jpg'
                try:
                    io.imsave(os.path.join(result_folder, text_area_path), text_crop_np)
                except Exception:
                    pass

                # CRNN inference
                t0 = time.time()
                # optional preprocessing
                if args.preproc:
                    try:
                        proc_np = preprocess_for_crnn(text_crop_np, args, name=name, rank=rank)
                        text_crop_pil = Image.fromarray(proc_np).convert('L')
                    except Exception:
                        text_crop_pil = Image.fromarray(text_crop_np).convert('L')
                else:
                    text_crop_pil = Image.fromarray(text_crop_np).convert('L')
                text_crop = transformer(text_crop_pil)
                if torch.cuda.is_available():
                    text_crop = text_crop.cuda()
                text_crop = text_crop.view(1, *text_crop.size())
                text_crop = Variable(text_crop)
                with torch.no_grad():
                    preds = crnn_net(text_crop)
                _, preds = preds.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([preds.size(0)])
                raw_pred = converter.decode(preds.data, preds_size, raw=True)
                sim_pred = converter.decode(preds.data, preds_size, raw=False)
                # print("crnn recognition time:{}".format(time.time() - t0))
                # print('%-20s => %-20s' % (raw_pred, sim_pred))
                recognized_all.append(sim_pred)
                poly = ','.join(str(int(p)) for p in bboxes[idx].reshape(-1))
                lines_out.append(f"{rank}\t{poly}\t{sim_pred}")
                rank += 1

        # write per-box and merged results
        try:
            with open(os.path.join(result_folder, f'res_{name}_texts.txt'), 'w', encoding='utf-8') as f:
                for line in lines_out:
                    f.write(line + "\n")
            with open(os.path.join(result_folder, f'res_{name}_merged.txt'), 'w', encoding='utf-8') as f:
                f.write(' '.join(recognized_all))
            print('Recognized (merged):', ' '.join(recognized_all))
        except Exception:
            pass

    #print("elapsed time : {}s".format(time.time() - t))
