import pandas as pd
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import sacrebleu




df = pd.read_csv("reports.csv")





def compute_bert(refs, cands):
    P, R, F1 = bert_score(cands, refs, lang="en", rescale_with_baseline=True)
    return float(F1.mean())

def compute_rouge(refs, cands):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for r, c in zip(refs, cands):
        scores = scorer.score(r, c)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rL.append(scores["rougeL"].fmeasure)
    return float(pd.Series(r1).mean()), float(pd.Series(r2).mean()), float(pd.Series(rL).mean())

def compute_bleu(refs, cands):
    
    bleu_scores = []
    for r, c in zip(refs, cands):
        bleu = sacrebleu.sentence_bleu(c, [r]).score
        bleu_scores.append(bleu)
    return float(pd.Series(bleu_scores).mean())





comparisons = [
    ("expert_original", "simple_original"),
    ("expert_original", "e_to_s"),
    ("direct_telugu", "restored_telugu"),
    ("direct_telugu", "lay_telugu"),
    ("direct_hindi", "restored_hindi"),
    ("direct_hindi", "lay_hindi"),
]

results = []




for ref_col, cand_col in comparisons:
    print(i)
    refs = df[ref_col].astype(str).tolist()
    cands = df[cand_col].astype(str).tolist()

    bert = compute_bert(refs, cands)
    r1, r2, rL = compute_rouge(refs, cands)
    bleu = compute_bleu(refs, cands)

    results.append({
        "reference": ref_col,
        "candidate": cand_col,
        "BERTScore": bert,
        "ROUGE-1": r1,
        "ROUGE-2": r2,
        "ROUGE-L": rL,
        "BLEU": bleu
    })




results_df = pd.DataFrame(results)
print(results_df)


results_df.to_csv("translation_evaluation_results.csv", index=False)
