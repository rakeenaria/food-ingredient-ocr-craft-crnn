# eval_merged_vs_gt.py
import argparse, pathlib, re
from collections import defaultdict

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

def read_gt(gt_path):
    gt={}
    for line in pathlib.Path(gt_path).read_text(encoding='utf-8').splitlines():
        parts=line.strip().split(maxsplit=1)
        if len(parts)==2:
            gt[pathlib.Path(parts[0]).stem]=parts[1]
    return gt

def read_merged(folder):
    merged={}
    for p in pathlib.Path(folder).glob("res_*_merged.txt"):
        stem=p.stem.replace("res_","").replace("_merged","")
        txt=p.read_text(encoding='utf-8').strip()
        merged[stem]=txt
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
    n=0; acc=0; norm_ed=0
    f1_sum=0; bleu_sum=0; rouge_sum=0
    cer_sum=0; cer_den=0
    wer_sum=0; wer_den=0
    for k in keys:
        g,p = gt[k], pred[k]
        n+=1
        if g==p: acc+=1
        norm_ed += 1 - levenshtein(p,g)/max(len(g), len(p)) if g and p else 0
        f1_sum += f1_score(g,p)
        bleu_sum += bleu1(g,p)
        rouge_sum += rouge_l(g,p)
        cer_sum += levenshtein(p, g)
        cer_den += len(g)
        gt_words = g.split()
        pred_words = p.split()
        wer_sum += levenshtein(pred_words, gt_words)
        wer_den += len(gt_words) if gt_words else 0
    print(f"Samples: {n}")
    print(f"Accuracy: {acc/n*100:.3f}")
    print(f"norm_ED: {norm_ed/n:.3f}")
    print(f"F1 (word-level): {f1_sum/n:.3f}")
    print(f"BLEU-1: {bleu_sum/n:.3f}")
    print(f"ROUGE-L: {rouge_sum/n:.3f}")
    cer = cer_sum/cer_den if cer_den>0 else 0.0
    wer = wer_sum/wer_den if wer_den>0 else 0.0
    print(f"CER: {cer:.3f}")
    print(f"WER: {wer:.3f}")

if __name__=="__main__":
    main()
