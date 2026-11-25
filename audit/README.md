# Label audit

The audit checks the dataset's labels against `policy/spam_policy.md` before the test set is used. No LLM output is used anywhere in it, so the LLM's single test run stays independent of the audit.

It has three parts, all labelled by hand against the policy:

1. **Flags**: comments where both baselines disagree with the label (63).
2. **Subtype sample**: 200 test spam comments drawn at random. Because they are random, the share marked not spam estimates the label error rate among test spam.
3. **Full review**: every remaining dev and test spam comment, plus 100 random ham comments from each of dev and test (614), labelled blind.

## Which comments are flagged

`python -m audit.flags` flags a comment when

- the keyword rules (`models/rules.py`) and the logistic-regression baseline (`models/logreg.py`) both disagree with its label (train comments are scored by a model fitted on the other train video), or
- another comment with the same text, after cleaning and normalising, carries the other label. On this dataset there are none.

Both baselines were fixed before any dev or test score was looked at.

## Files

Built from the pinned data (regenerate identically: `python -m audit.flags --check`, `python -m audit.full_review --check`, both also run by the tests):

| File | What it holds |
|---|---|
| `flags.csv` | One row per flag: split, video, current label, both baselines' scores, reason |
| `review_sheet.csv` | What the reviewer sees: cleaned text and current label only, in a shuffled order, with empty answer columns |
| `blind_recheck.csv` | 20 flags drawn by seed before any flag was discussed, for a blind second review days later |
| `subtype_sample.csv` | 200 test spam comments drawn by seed for hand-labelled subtypes |
| `full_review.csv` | The 614 comments of the full review: 160 dev and 254 test spam not already reviewed, and 100 ham from each split, drawn by seed and shuffled |

Written from the filled workbooks by `audit/apply_review.py`:

| File | What it holds |
|---|---|
| `review_verdicts.csv` | The verdict, subtype and reason for each flag |
| `subtype_labels.csv` | The subtype (or not_spam, or unsure) and note for each sampled comment |
| `full_review_labels.csv` | The answer and note for each full-review comment |
| `recheck_verdicts.csv` | The blind re-check answers beside the first ones, and whether the final label agrees |
| `unsure_resolution.csv` | The second-pass answer and note on each first-pass `unsure` |
| `corrections.csv` | One row per comment whose label changes: before, after, subtype, which review(s) found it, reason |
| `scoring_exclusions.csv` | Comments left out of scoring: `Eminem:269`, one source row that holds six comments |

## Workbooks

`python -m audit.workbook <folder> --set s1|full|recheck` writes Excel working copies with drop-down answer columns and a sheet of the five videos. It never overwrites a workbook that already exists. Filled workbooks live outside git (`labels/`, excluded locally).

- `s1`: `audit_review.xlsx` (the flags, current label shown) and `subtype_labels.xlsx` (the sample).
- `full`: `review_dev.xlsx` and `review_test.xlsx`. The current label is hidden and spam and ham are shuffled together, so every answer is blind. Answer with a subtype for spam, `not_spam` for ham, or `unsure`.
- `recheck`: `blind_recheck.xlsx`, the 20 re-check flags in a new order under new IDs (R01–R20), without the first answers.
- `unsure`: `unsure.xlsx`, a second pass on every `unsure` answer from the flags and the full review (U01–U15), current label hidden, first-pass note shown. Every answer needs a note. The first-pass record is not changed, so the blind re-check still compares with the first answers.

Then:

    python -m audit.apply_review --audit labels/audit_review.xlsx --subtypes labels/subtype_labels.xlsx
        [--full-dev labels/review_dev.xlsx --full-test labels/review_test.xlsx]
        [--recheck labels/blind_recheck.xlsx] [--unsure labels/unsure.xlsx] [--check]

It stops, writing nothing, if a row or a given column was changed, an answer is blank or not a drop-down value, or the two S1 sheets disagree on a comment they share. `--check` compares with the files already in `audit/`.

## How a comment is reviewed

1. Read the text and decide what the policy says it is (before looking at the current label, where it is shown).
2. Flags: verdict **keep** (current label is right), **correct** (it is wrong) or **unsure** (the policy does not settle it; say why), and a one-line reason naming the policy section or edge case.
3. For anything whose final label is spam, and for unsure, give the subtype by the policy's order.
4. Subtype sample and full review: an answer that disagrees with the current label, and any unsure, needs a note.

The reviewer never sees the baselines' scores.

## What follows the review

An unsure answer never changes a label; its second-pass answer can (source `unsure_resolution` in `corrections.csv`). `data/splits.json` and the raw data never change: results are reported on the original labels and on the corrected labels, with `Eminem:269` left out of scoring. Agreement between the blind re-check and the first review is reported.
