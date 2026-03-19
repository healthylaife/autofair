import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sys

sys.path.append("evaluation")
from metrics import DomainSpecificity

# Run once if needed:
# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("punkt_tab")

stop_words = set(stopwords.words("english"))
domain_specificity = DomainSpecificity()
OUTCOME = "clinical decision"

DATASETS = [
    {
        "group": "Reviewer-generated",
        "method": "Original",
        "path": "dataset/vignettes_our_gpt4o.xlsx",
        "column": "Question",
        "has_ref": True,
        "ref_col": "Reference",
    },
    {
        "group": "Reviewer-generated",
        "method": "No Hint",
        "path": "../output/no_hint_reviewer_clean.csv",
        "column": "Question",
        "has_ref": True,
        "ref_col": "Reference",
    },
    {
        "group": "Reviewer-generated",
        "method": "Complex Context",
        "path": "../output/complex_context_reviewer_clean.csv",
        "column": "Question",
        "has_ref": True,
        "ref_col": "Reference",
    },
]

def distinct_tokens(text: str):
    tokens = [t.lower() for t in word_tokenize(str(text))]
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return set(tokens)

def diversity_each(texts):
    counts = [len(distinct_tokens(t)) for t in texts]
    return float(np.mean(counts)), float(np.std(counts))

def diversity_all(texts):
    all_tok = set()
    for t in texts:
        all_tok.update(distinct_tokens(t))
    return len(all_tok)

def outcome_similarity(texts, outcome):
    sims = [domain_specificity.caluate(str(t), outcome) for t in texts]
    return float(np.mean(sims)), float(np.std(sims))

def ref_similarity(texts, refs):
    sims = []
    for t, r in zip(texts, refs):
        if pd.isna(r) or str(r).strip() == "" or str(r).strip().lower() == "none":
            continue
        sims.append(domain_specificity.caluate(str(t), str(r)))
    if not sims:
        return None, None
    return float(np.mean(sims)), float(np.std(sims))

rows = []

for ds in DATASETS:
    if ds["path"].endswith(".xlsx"):
        df = pd.read_excel(ds["path"])
    else:
        df = pd.read_csv(ds["path"])

    texts = df[ds["column"]].fillna("").astype(str).tolist()

    div_mean, div_std = diversity_each(texts)
    div_all = diversity_all(texts)
    out_mean, out_std = outcome_similarity(texts, OUTCOME)

    refs = df[ds["ref_col"]].fillna("").astype(str).tolist()
    ref_mean, ref_std = ref_similarity(texts, refs)
    ref_str = "-" if ref_mean is None else f"{ref_mean:.2f} ({ref_std:.2f})"

    rows.append({
        "Group": ds["group"],
        "Method": ds["method"],
        "N": len(texts),
        "Each Vignette": f"{div_mean:.2f} ({div_std:.2f})",
        "All Vignettes": div_all,
        "Ref. Similarity": ref_str,
        "Outcome Similarity": f"{out_mean:.2f} ({out_std:.2f})",
    })

out_df = pd.DataFrame(rows)
print(out_df.to_string(index=False))
out_df.to_csv("evaluation/reviewer_new_vignette_results.csv", index=False)
print("\nSaved: evaluation/reviewer_new_vignette_results.csv")
