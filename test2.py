# eval_merged_vs_gt.py
import argparse
import pathlib
import re

def levenshtein(a,b):
    la,lb=len(a),len(b)
    dp=list(range(lb+1))
    for i,ca in enumerate(a,1):
        prev=dp[0]; dp[0]=i
        for j,cb in enumerate(b,1):
            cur=dp[j]
            dp[j]=prev if ca==cb else 1+min(prev,dp[j],dp[j-1])
            prev=cur
    return dp[-1]


def normalize_key(stem):
    """Normalize trailing numeric suffix so bahan_01 and bahan_1 map to same key."""
    match = re.fullmatch(r"(.*_)(\d+)", stem)
    if not match:
        return stem
    return f"{match.group(1)}{int(match.group(2))}"

def normalize_text(text):
    """Treat newline, tabs, and repeated spaces as one normal space."""
    return " ".join(text.split())

def read_gt(gt_path):
    gt={}
    for line in pathlib.Path(gt_path).read_text(encoding='utf-8').splitlines():
        parts=line.strip().split(maxsplit=1)
        if len(parts)==2:
            gt[normalize_key(pathlib.Path(parts[0]).stem)] = normalize_text(parts[1])
    return gt

def read_merged(folder):
    merged={}
    for p in pathlib.Path(folder).glob("res_*_merged.txt"):
        if "_line" in p.stem:
            continue
        stem=p.stem.replace("res_","").replace("_merged","")
        txt=p.read_text(encoding='utf-8').strip()
        merged[normalize_key(stem)] = normalize_text(txt)
    return merged

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="gt.txt")
    ap.add_argument("--merged_dir", default="craft_results", help="folder res_*_merged.txt")
    args=ap.parse_args()

    gt=read_gt(args.gt)
    pred=read_merged(args.merged_dir)
    keys=sorted(set(gt.keys())&set(pred.keys()))
    if not keys:
        print("Tidak ada nama gambar yang cocok.")
        return

    def f1_score(g,p):
        g_words = g.split()
        p_words = p.split()
        if not g_words and not p_words: return 1.0
        if not g_words or not p_words: return 0.0
        from collections import Counter
        cg, cp = Counter(g_words), Counter(p_words)
        tp = sum((cg & cp).values())
        prec = tp / len(p_words)
        rec = tp / len(g_words)
        return 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)

    def bleu1(g,p):
        g_words = g.split()
        p_words = p.split()
        if not p_words: return 0.0
        overlap = sum((Counter(g_words) & Counter(p_words)).values())
        p1 = overlap / len(p_words)
        if len(p_words)==0: return 0.0
        if len(p_words) > len(g_words):
            bp = 1.0
        elif len(p_words)==0:
            bp = 0.0
        else:
            bp = pow(2.718281828, 1 - len(g_words)/len(p_words))
        return bp * p1

    def rouge_l(g,p):
        g_words = g.split()
        p_words = p.split()
        if not g_words and not p_words: return 1.0
        if not g_words or not p_words: return 0.0
        # LCS
        m,n = len(g_words), len(p_words)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                if g_words[i]==p_words[j]:
                    dp[i+1][j+1]=dp[i][j]+1
                else:
                    dp[i+1][j+1]=max(dp[i][j+1], dp[i+1][j])
        lcs = dp[m][n]
        prec = lcs / n
        rec = lcs / m
        return 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)

    from collections import Counter
    n=0; norm_ed=0
    f1_sum=0; bleu_sum=0; rouge_sum=0
    cer_sum=0; cer_den=0
    wer_sum=0; wer_den=0
    per_image_rows = []
    for k in keys:
        g,p = gt[k], pred[k]
        n+=1
        sample_norm_ed = 1 - levenshtein(p,g)/max(len(g), len(p)) if g and p else 0
        sample_f1 = f1_score(g,p)
        sample_bleu = bleu1(g,p)
        sample_rouge = rouge_l(g,p)
        sample_char_edit = levenshtein(p, g)
        sample_cer = sample_char_edit / len(g) if len(g) > 0 else 0.0
        gt_words = g.split()
        pred_words = p.split()
        sample_word_edit = levenshtein(pred_words, gt_words)
        sample_wer = sample_word_edit / len(gt_words) if gt_words else 0.0
        sample_word_accuracy = max(0.0, 1.0 - sample_wer) * 100.0

        norm_ed += sample_norm_ed
        f1_sum += sample_f1
        bleu_sum += sample_bleu
        rouge_sum += sample_rouge
        cer_sum += sample_char_edit
        cer_den += len(g)
        wer_sum += sample_word_edit
        wer_den += len(gt_words) if gt_words else 0
        per_image_rows.append(
            (
                k,
                len(gt_words),
                len(pred_words),
                sample_norm_ed,
                sample_f1,
                sample_bleu,
                sample_rouge,
                sample_word_accuracy,
                sample_cer,
                sample_wer,
            )
        )
    print("Per-image metrics:")
    print("Image\tGT Words\tPred Words\tnorm_ED\tF1\tBLEU-1\tROUGE-L\tWord Accuracy\tCER\tWER")
    for row in per_image_rows:
        (
            k,
            gt_word_count,
            pred_word_count,
            sample_norm_ed,
            sample_f1,
            sample_bleu,
            sample_rouge,
            sample_word_accuracy,
            sample_cer,
            sample_wer,
        ) = row
        print(
            f"{k}\t{gt_word_count}\t{pred_word_count}\t{sample_norm_ed:.3f}\t"
            f"{sample_f1:.3f}\t{sample_bleu:.3f}\t{sample_rouge:.3f}\t"
            f"{sample_word_accuracy:.3f}\t{sample_cer:.3f}\t{sample_wer:.3f}"
        )
    print()
    print(f"Samples: {n}")
    print(f"GT Words: {wer_den}")
    print(f"Pred Words: {sum(row[2] for row in per_image_rows)}")
    print(f"norm_ED: {norm_ed/n:.3f}")
    print(f"F1 (word-level): {f1_sum/n:.3f}")
    print(f"BLEU-1: {bleu_sum/n:.3f}")
    print(f"ROUGE-L: {rouge_sum/n:.3f}")
    cer = cer_sum/cer_den if cer_den>0 else 0.0
    wer = wer_sum/wer_den if wer_den>0 else 0.0
    word_accuracy = max(0.0, 1.0 - wer) * 100.0
    print(f"Word Accuracy: {word_accuracy:.3f}")
    print(f"CER: {cer:.3f}")
    print(f"WER: {wer:.3f}")

if __name__=="__main__":
    main()
