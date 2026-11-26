---
license: cc-by-4.0
language:
- en
pretty_name: YouTube Spam Collection, audited labels
task_categories:
- text-classification
size_categories:
- 1K<n<10K
tags:
- spam
- youtube
- content-moderation
- label-noise
- data-quality
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.csv
  - split: validation
    path: data/validation.csv
  - split: test
    path: data/test.csv
---

# YouTube Spam Collection, audited labels

The 1,956 comments of the [YouTube Spam Collection](https://archive.ics.uci.edu/dataset/380/) (Alberto & Lochter, UCI) with their original spam/ham labels, a corrected label for each, the answer a hand review under a written spam policy gave, and the reason for every one of the **88 corrections** (87 spam → ham, 1 ham → spam).

The audit was done for the project [yt-comment-spam](https://github.com/PritRai75/yt-comment-spam), which compares keyword rules, two classical classifiers and an LLM on these comments. The policy, the review sheets, the code that builds this dataset and the tests that check it are all in that repository.

## Why corrected labels

Under the project's policy (`policy/spam_policy.md` in the repository), 68 of the 481 test comments labelled spam are not spam (14.1%, 95% Wilson interval 11.3%–17.5%). A spam detector scored on the original labels is marked wrong for calling those comments ham.

The rule cited most in the correction reasons is edge case 3: asking for likes on the comment itself ("1 like = 1 prayer") is ham, since it promotes nothing outside the thread (32 of the 88 reasons). Rules cited, by number of corrections: EC3 32, EC8 8, P1 8, EC10 8, EC4 8, EC9 5 (see the policy).

The corrected labels follow one policy, written for this project. They are not ground truth for other definitions of spam: keep `label_original` beside them, and read `correction_reason` before relying on a correction.

## Splits

Split by video, so a model is evaluated on videos it never saw. `validation` is the project's dev split.

| Split | Videos | Comments | Spam (original) | Spam (corrected) | Corrections |
|---|---|---:|---:|---:|---:|
| train | KatyPerry, Psy | 700 | 350 | 348 | 2 |
| validation | Shakira | 370 | 174 | 157 | 17 |
| test | Eminem, LMFAO | 886 | 481 | 414 | 69 |

## Columns

| Column | Meaning |
|---|---|
| `row_id` | `<video>:<row>`, the row's 0-based position in its CSV as published; unique (`comment_id` is not: three rows appear twice) |
| `comment_id` | `COMMENT_ID` from the original files; join on it (with the video) for author and date, left out here |
| `video` | Psy, KatyPerry, LMFAO, Eminem or Shakira |
| `split` | train, dev or test, as in the project (dev is this dataset's `validation`) |
| `content` | The comment exactly as published, HTML and invisible characters included |
| `label_original` | 1 spam, 0 ham, as published |
| `label_corrected` | 1 spam, 0 ham, after the audit |
| `corrected` | `label_corrected` differs from `label_original` |
| `review` | The reviewer's answer under the policy: `scam`, `link`, `self_promotion`, `bulk`, `not_spam`, `unsure`, or empty if the comment was not reviewed |
| `correction_source` | Which part of the audit led to the correction (below); empty if not corrected |
| `correction_reason` | The reviewer's reason, citing the policy's edge cases (`EC…`) and rules (`P…`) |
| `share_text` | The comment is in YouTube's "Check out this video on YouTube" share-text family |
| `scored` | False for the 1 row(s) the project leaves out of scoring (Eminem:269: one source row holding several comments) |

## How the labels were audited

One reviewer, against the written policy, before any LLM was run on the comments, in three parts:

1. **Flags:** comments where a keyword-rule baseline and a logistic-regression baseline both disagreed with the label.
2. **Subtype sample:** 200 test comments labelled spam, drawn at random, each given a subtype or `not_spam`.
3. **Full review:** every remaining dev and test comment labelled spam, plus 100 random ham comments from each of dev and test, with the label hidden.

Every dev and test comment labelled spam was reviewed; 855 dev and test comments were reviewed in all. Of the 200 reviewed ham, 1 was judged spam. Unsure answers had a second pass under the final policy. Corrections by source (a correction can have two): audit_flag 38, full_review 30, subtype_sample 26, unsure_resolution 3.

Spam subtypes on the corrected labels of the scored dev and test comments: self_promotion 299, bulk 159, scam 62, link 50.

**Consistency:** a blind re-check of 20 flags two days after the first review gave the same label as the first answers on 16 of 20; 4 of the other 4 had been answered unsure the first time. This is one reviewer's consistency, not agreement between reviewers.

## Limitations

- One reviewer, who also wrote the policy. No second labeller.
- Train comments were not reviewed apart from the flags, so train keeps almost all of its original labels.
- Ham was sampled (100 per split), not reviewed in full, so spam mislabelled as ham is estimated, not corrected everywhere.
- Five music videos from 2013–2015, English only; today's spam looks different.
- The policy makes choices other definitions would not, for example likes asked for the comment itself are ham, and YouTube's share text is spam (bulk).

## Licence and credit

CC BY 4.0, as the original dataset. Original data: Alberto, T. & Lochter, J. (2015). YouTube Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58885. The comment text here is unmodified.

The corrected labels, review answers and reasons: Prit Raithatha, https://github.com/PritRai75/yt-comment-spam.
