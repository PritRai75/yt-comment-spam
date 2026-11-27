# Report rehearsal on dev

Written by `python -m eval.report --split dev` from the result files; the README's blocks are rendered the same way from the test files.

## Results

**Dev: 370 scored comments from videos no system was trained on, 157 spam under the corrected labels.** Each system at its own spam cut-off; 95% intervals (Wilson; F1 by bootstrap).

| System | Spam if | Precision | Recall | F1 | False-positive rate | Avg. precision | TP / FP / FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Keyword rules | ≥ 1 rule | 0.930 (0.880–0.961) | 0.936 (0.887–0.965) | 0.933 (0.903–0.959) | 0.052 (0.029–0.090) | 0.928 | 147 / 11 / 10 |
| TF-IDF + logistic regression | p ≥ 0.5 | 0.957 (0.910–0.980) | 0.854 (0.790–0.900) | 0.902 (0.864–0.935) | 0.028 (0.013–0.060) | 0.963 | 134 / 6 / 23 |
| Gradient boosting, readable features | p ≥ 0.5 | 0.905 (0.847–0.943) | 0.854 (0.790–0.900) | 0.879 (0.837–0.914) | 0.066 (0.040–0.107) | 0.962 | 134 / 14 / 23 |
| LLM v1 (policy only) | ≥ 3 of 5 votes | 0.994 (0.965–0.999) | 0.994 (0.965–0.999) | 0.994 (0.983–1.000) | 0.005 (0.001–0.026) | 0.999 | 156 / 1 / 1 |
| LLM v2 (+ 8 examples, normalisation) | ≥ 3 of 5 votes | 1.000 (0.976–1.000) | 0.994 (0.965–0.999) | 0.997 (0.989–1.000) | 0.000 (0.000–0.018) | 1.000 | 156 / 0 / 1 |

**F1 on other label and comment sets** (secondary; same cut-offs).

| System | Corrected labels (370) | Original labels (370) | Corrected, without share text (360) | Corrected, without the comments the policy quotes (368) |
|---|---:|---:|---:|---:|
| Keyword rules | 0.933 | 0.946 | 0.932 | 0.936 |
| TF-IDF + logistic regression | 0.902 | 0.885 | 0.899 | 0.905 |
| Gradient boosting, readable features | 0.879 | 0.888 | 0.874 | 0.881 |
| LLM v1 (policy only) | 0.994 | 0.943 | 0.993 | 0.994 |
| LLM v2 (+ 8 examples, normalisation) | 0.997 | 0.945 | 0.997 | 0.997 |

**Hypotheses** (exact McNemar on paired predictions, Holm's adjustment across the two, α = 0.05). Discordant: comments only the first system got right / only the second.

| Hypothesis | Systems | Discordant | p | Holm p | Reject H0 | F1 difference (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| H1 | LLM v2 (+ 8 examples, normalisation) vs LLM v1 (policy only) | 1 / 0 | 1.000 | 1.000 | no | +0.003 (+0.000 to +0.011) |
| H2 | LLM v2 (+ 8 examples, normalisation) vs TF-IDF + logistic regression | 28 / 0 | 7.5e-09 | 1.5e-08 | yes | +0.094 (+0.062 to +0.131) |
| H1 (original labels) | LLM v2 (+ 8 examples, normalisation) vs LLM v1 (policy only) | 1 / 0 | 1.000 | 1.000 | no | +0.003 (+0.000 to +0.010) |
| H2 (original labels) | LLM v2 (+ 8 examples, normalisation) vs TF-IDF + logistic regression | 23 / 5 | 9.1e-04 | 0.002 | yes | +0.060 (+0.027 to +0.096) |

**Enforcement tiers** (thresholds chosen on dev, corrected labels; `eval/thresholds.json`).

| System | Auto-removed | Auto-removal precision | Spam auto-removed | Sent to review | Review per 1,000 comments | Spam removed or reviewed |
|---|---:|---:|---:|---:|---:|---:|
| Keyword rules | 67 | 1.000 (0.946–1.000) | 42.7% | 303 | 819 | 100.0% |
| TF-IDF + logistic regression | 74 | 0.986 (0.927–0.998) | 46.5% | 122 | 330 | 96.2% |
| Gradient boosting, readable features | 88 | 0.989 (0.938–0.998) | 55.4% | 105 | 284 | 95.5% |
| LLM v1 (policy only) | 157 | 0.994 (0.965–0.999) | 99.4% | 0 | 0 | 99.4% |
| LLM v2 (+ 8 examples, normalisation) | 156 | 1.000 (0.976–1.000) | 99.4% | 0 | 0 | 99.4% |

**LLM running cost** at US$1.00 per million input and US$5.00 per million output tokens (Claude Haiku 4.5 on Amazon Bedrock, Standard tier, global cross-region inference; source: Anthropic model pricing list (Amazon Bedrock section), effective 31 Aug 2026, https://assets.claude.com/9e03129acc36d31970777d336f969bb53dc84355.pdf, read 2026-11-26). Five votes per comment; comments with the same text under one video share their votes.

| System | Calls | Invalid | Input tokens per 1,000 comments | Output tokens per 1,000 | Model minutes per 1,000 | US$ per 1,000 |
|---|---:|---:|---:|---:|---:|---:|
| LLM v1 (policy only) | 1,620 | 0 | 16,576,189 | 430,819 | 79 | $18.73 |
| LLM v2 (+ 8 examples, normalisation) | 1,620 | 0 | 19,421,230 | 428,835 | 77 | $21.57 |

## One comment, as each system classified it

Chosen by a fixed rule: the first dev comment, in data order, whose label the audit corrected and on which the keyword rules and LLM v2 disagree, leaving out the comments the policy quotes. `Shakira:132`, as every system received it after cleaning:

```
Help shakira's waka waka be the first song by a female artist to reach 1 billion views.
(dark horse is ahead by roughly 100 million more views, and roar has only 50 million more views)
https://www.youtube.com/watch?v=pRpeEdMmmQ0
```

Dataset label **spam**; corrected label **ham** (audit reason: EC1/EC8: links this video; view campaign for this artist (P1)).

| System | Spam if | Score | Decision | First vote's stated reason |
|---|---|---|---|---|
| Keyword rules | ≥ 1 rule | 1 rule family (link) | **spam** |  |
| TF-IDF + logistic regression | p ≥ 0.5 | p = 0.332 | **ham** |  |
| Gradient boosting, readable features | p ≥ 0.5 | p = 1.000 | **spam** |  |
| LLM v1 (policy only) | ≥ 3 of 5 votes | 0 of 5 votes spam | **ham** | "Fan activity for this artist: view campaign with no request for personal benefit." |
| LLM v2 (+ 8 examples, normalisation) | ≥ 3 of 5 votes | 0 of 5 votes spam | **ham** | "Fan activity for this video's artist with no request for commenter benefit." |

## Red team

**Dev: 98 spam comments that every attacked system caught** (drawn with seed 20261125 from 98 distinct texts; attacked: Keyword rules; TF-IDF + logistic regression; Gradient boosting, readable features). Evaded = scored below the system's own cut-off. Cells: evaded / variants, as is → with normalisation in front.

| Transform | Keyword rules | TF-IDF + logistic regression | Gradient boosting, readable features |
|---|---:|---:|---:|
| Broken-up links | 3/17 (18%) → 0/17 (0%) | 2/17 (12%) → 0/17 (0%) | 5/17 (29%) → 0/17 (0%) |
| Leetspeak | 76/98 (78%) → 76/98 (78%) | 36/98 (37%) → 36/98 (37%) | 80/98 (82%) → 80/98 (82%) |
| Look-alike letters | 80/98 (82%) → 6/98 (6%) | 83/98 (85%) → 2/98 (2%) | 81/98 (83%) → 7/98 (7%) |
| Zero-width characters | 80/98 (82%) → 0/98 (0%) | 83/98 (85%) → 0/98 (0%) | 81/98 (83%) → 0/98 (0%) |
| Benign wrapper | 0/98 (0%) → 0/98 (0%) | 70/98 (71%) → 70/98 (71%) | 0/98 (0%) → 0/98 (0%) |
| Injection note | 0/98 (0%) → 0/98 (0%) | 19/98 (19%) → 19/98 (19%) | 0/98 (0%) → 0/98 (0%) |
| **All six** | 239/507 (47%) → 82/507 (16%) | 293/507 (58%) → 127/507 (25%) | 247/507 (49%) → 87/507 (17%) |

Pooled evasion with a bootstrap 95% interval over comments: Keyword rules as is 47.1% (41.6%–51.7%); Keyword rules normalised 16.2% (13.8%–18.2%); TF-IDF + logistic regression as is 57.8% (51.3%–63.9%); TF-IDF + logistic regression normalised 25.0% (20.8%–29.2%); Gradient boosting, readable features as is 48.7% (43.4%–53.0%); Gradient boosting, readable features normalised 17.2% (14.8%–19.1%).

## Label audit

**88 label corrections** (`audit/corrections.csv`), each with the policy reason:

| Split | Spam → ham | Ham → spam |
|---|---:|---:|
| train | 2 | 0 |
| dev | 17 | 0 |
| test | 68 | 1 |
| **All** | 87 | 1 |

- Test comments labelled spam that the review judged not spam under the policy: 68 of 481 (14.1%, 95% CI 11.3%–17.5%). Every dev and test spam comment was reviewed.
- Ham comments reviewed (100 drawn from each of dev and test): 1 of 200 judged spam.
- Blind re-check of 20 audit flags two days later: 16 of 20 gave the same label as the first review; the other 4 had all been answered unsure the first time. One labeller, so this is consistency, not agreement between raters.
