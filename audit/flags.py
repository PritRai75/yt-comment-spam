"""Build the label-audit files. No LLM output is used anywhere in the audit.

A comment is flagged when
  - the rules and the logistic-regression baseline both disagree with its label, or
  - another comment with the same text (data.clean.dup_key) has the other label.

Outputs (all regenerated identically from the pinned data):
  audit/flags.csv          one row per flag, with the two baselines' scores
  audit/review_sheet.csv   what the reviewer sees: text and label only, shuffled
  audit/blind_recheck.csv  flags drawn now for a blind second review days later
  audit/subtype_sample.csv 200 test spam comments for hand-labelled subtypes

    python -m audit.flags           write the files
    python -m audit.flags --check   regenerate and compare with the files on disk
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data.clean import clean_text, dup_key
from data.fetch import ROOT
from data.load import load_comments
from models import logreg, rules

SEED = 20261125
BLIND_N = 20
SUBTYPE_N = 200
AUDIT = ROOT / "audit"
FILES = ["flags.csv", "review_sheet.csv", "blind_recheck.csv", "subtype_sample.csv"]


def label_name(y: int) -> str:
    return "spam" if y == 1 else "ham"


def build() -> dict:
    df = load_comments()
    df["rules_hits"] = df.content.map(lambda t: "|".join(rules.rule_hits(t)))
    df["rules_score"] = df.rules_hits.map(lambda h: 0 if h == "" else h.count("|") + 1)
    df["lr_proba"] = logreg.score_all(df).round(6)
    rules_wrong = (df.rules_score >= 1).astype(int) != df.label
    lr_wrong = (df.lr_proba >= 0.5).astype(int) != df.label
    key = df.content.map(dup_key)
    conflict_keys = df.assign(key=key).groupby("key").label.nunique()
    conflict_keys = set(conflict_keys[conflict_keys > 1].index) - {""}
    dup_conflict = key.isin(conflict_keys)

    reasons = np.where(rules_wrong & lr_wrong & dup_conflict, "both_baselines_wrong;duplicate_conflict",
              np.where(rules_wrong & lr_wrong, "both_baselines_wrong",
              np.where(dup_conflict, "duplicate_conflict", "")))
    df["reason"] = reasons
    flagged = df[df.reason != ""].copy()

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(flagged))
    flagged = flagged.iloc[order].reset_index(drop=True)
    flagged.insert(0, "flag_id", [f"F{i + 1:03d}" for i in range(len(flagged))])

    flags = flagged[["flag_id", "row_id", "split", "video", "label", "rules_score", "rules_hits", "lr_proba", "reason"]]
    flags = flags.rename(columns={"label": "gold"})
    flags["gold"] = flags.gold.map(label_name)

    sheet = pd.DataFrame({
        "flag_id": flagged.flag_id,
        "text": flagged.content.map(clean_text),
        "current_label": flagged.label.map(label_name),
        "verdict": "",          # keep / correct / unsure
        "subtype_if_spam": "",  # scam / link / self_promotion / bulk, for the final label
        "reason": "",
    })

    n_blind = min(BLIND_N, len(flagged))
    blind_idx = np.sort(rng.choice(len(flagged), size=n_blind, replace=False))
    blind = pd.DataFrame({"flag_id": flagged.flag_id.iloc[blind_idx].values})

    test_spam = df[(df.split == "test") & (df.label == 1)].reset_index(drop=True)
    pick = np.sort(rng.choice(len(test_spam), size=SUBTYPE_N, replace=False))
    sample = test_spam.iloc[pick].reset_index(drop=True)
    subtype = pd.DataFrame({
        "sample_id": [f"S{i + 1:03d}" for i in range(len(sample))],
        "row_id": sample.row_id,
        "text": sample.content.map(clean_text),
        "subtype": "",  # scam / link / self_promotion / bulk / not_spam
        "notes": "",
    })
    return {"flags.csv": flags, "review_sheet.csv": sheet, "blind_recheck.csv": blind,
            "subtype_sample.csv": subtype}


def write(tables: dict) -> None:
    for name, t in tables.items():
        # utf-8-sig so Excel shows emoji and accents correctly
        t.to_csv(AUDIT / name, index=False, encoding="utf-8-sig", lineterminator="\n")


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(AUDIT / name, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def check(tables: dict) -> int:
    bad = 0
    for name, t in tables.items():
        on_disk = read(name)
        fresh = t.astype(str).replace("nan", "").reset_index(drop=True)
        # probabilities may differ in the last printed digit across platforms' maths libraries
        num = ["lr_proba"] if "lr_proba" in fresh.columns else []
        text_cols = [c for c in fresh.columns if c not in num]
        same = (list(on_disk.columns) == list(fresh.columns) and len(on_disk) == len(fresh)
                and on_disk[text_cols].equals(fresh[text_cols])
                and all(np.allclose(on_disk[c].astype(float), t[c].astype(float), rtol=0, atol=2e-6) for c in num))
        print(("OK      " if same else "DIFFERS ") + f"audit/{name} ({len(t)} rows)")
        bad += not same
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    tables = build()
    if args.check:
        return check(tables)
    write(tables)
    f = tables["flags.csv"]
    print(f"{len(f)} flags: " + ", ".join(f"{k} {v}" for k, v in f.reason.value_counts().items()))
    print("by split: " + ", ".join(f"{k} {v}" for k, v in f.split.value_counts().items()))
    print("by current label: " + ", ".join(f"{k} {v}" for k, v in f.gold.value_counts().items()))
    print(f"blind re-check: {len(tables['blind_recheck.csv'])} flags; subtype sample: {len(tables['subtype_sample.csv'])} test spam")
    return 0


if __name__ == "__main__":
    sys.exit(main())
