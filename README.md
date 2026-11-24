# YouTube Comment Spam Detection with LLMs

A small, complete enforcement-detection workflow for YouTube comment spam: a written spam policy, audited labels, four classifiers (keyword rules, TF-IDF + logistic regression, and two LLM prompt versions), a two-threshold enforcement rule (auto-remove, human review, allow), a pre-registered experiment and adversarial evasion tests.

**Status:** in progress. Data is pinned; results will be added here once the single test run is done.

## Data

[YouTube Spam Collection](https://archive.ics.uci.edu/dataset/380/) from the UCI Machine Learning Repository: 1,956 real comments from five music videos, collected in 2015, each labelled spam or ham.

| Split | Videos | Comments | Spam | Ham |
|---|---|---:|---:|---:|
| Train | Psy, KatyPerry | 700 | 350 | 350 |
| Dev | Shakira | 370 | 174 | 196 |
| Test | LMFAO, Eminem | 886 | 481 | 405 |

Splitting by video means every evaluation comment comes from a video the classifiers never saw, as a live enforcement system would face.

The zip is stored as downloaded in `data/raw/`, with its size and SHA-256 (and those of each CSV) in `data/manifest.json`. To re-download and re-pin:

```
python data/fetch.py
```

To verify the stored copy without a download:

```
python data/fetch.py --check
```

**What the raw data contains.** 1,548 comments end in an invisible U+FEFF character whose presence tracks the label on some videos (on Eminem: every ham comment, 21% of spam); the Eminem video has no dates on any spam comment; 106 comments contain HTML tags and 263 HTML entities; three rows appear twice. Every classifier therefore sees the same cleaned text (`data/clean.py`: the text a viewer saw, without U+FEFF), and none uses the date or author.

## Policy and label audit

`policy/spam_policy.md` defines spam, four subtypes (scam or giveaway, link spam, self-promotion, bulk) and the edge cases. The label audit (`audit/`) flags comments where a keyword-rule baseline and a logistic-regression baseline both disagree with the label, and every flag is hand-reviewed against the policy. See `audit/README.md`.

**Credit and licence.** Alberto, T. & Lochter, J. (2015). YouTube Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58885. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the copy in `data/raw/` is unmodified.

## Quick start

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Licence

Code: MIT (see `LICENSE`). Data: CC BY 4.0, as above.
