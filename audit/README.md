# Label audit

The audit checks the dataset's labels against `policy/spam_policy.md` before the test set is used. No LLM output is used anywhere in it, so the LLM's single test run stays independent of the audit.

## Which comments are flagged

`python -m audit.flags` flags a comment when

- the keyword rules (`models/rules.py`) and the logistic-regression baseline (`models/logreg.py`) both disagree with its label (train comments are scored by a model fitted on the other train video), or
- another comment with the same text, after cleaning and normalising, carries the other label. On this dataset there are none.

Both baselines were fixed before any dev or test score was looked at.

## Files

| File | What it holds |
|---|---|
| `flags.csv` | One row per flag: split, video, current label, both baselines' scores, reason |
| `review_sheet.csv` | What the reviewer sees: cleaned text and current label only, in a shuffled order, with empty answer columns |
| `blind_recheck.csv` | 20 flags drawn by seed before any flag was discussed, for a blind second review days later |
| `subtype_sample.csv` | 200 test spam comments drawn by seed for hand-labelled subtypes |

All four regenerate with identical content from the pinned data (`python -m audit.flags --check`, also run by the tests). `python -m audit.workbook <folder>` writes Excel working copies with drop-down answer columns.

## How a flag is reviewed

1. Read the text and decide what the policy says it is, before looking at the current label.
2. Verdict: **keep** (current label is right), **correct** (it is wrong), or **unsure** (the policy does not settle it; say why).
3. For anything whose final label is spam, give the subtype by the policy's precedence order.
4. A one-line reason, naming the policy section or edge case.

The reviewer does not see the baselines' scores while reviewing.

## What follows the review

Corrections go into `corrections.csv` in one commit. `data/splits.json` and the raw data never change: results are reported on the original labels and on the corrected labels. The 20 blind re-check flags are reviewed again a few days later without the first verdicts, and agreement with the first review is reported.
