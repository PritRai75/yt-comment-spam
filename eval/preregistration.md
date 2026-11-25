# Pre-registration: the single test run

Written 26 November 2026, before any system is scored on the test split. The commit that adds this file freezes everything below; `eval/lock.py` refuses the test split until this file (by its SHA-256), the thresholds, the analysis code and the label-audit record are all committed and unchanged, and every test command also needs `--final`.

## 1. What is tested

The test split is every comment under two videos no system was built or tuned on: LMFAO (438 scored) and Eminem (447 scored), 886 comments in all. `Eminem:269` is left out of scoring because one source row holds six comments (decision 28), so **885 comments are scored**. Spam among them: **413 on the corrected labels, 480 on the original labels**.

Two label sets are reported side by side:

- **Corrected labels (primary).** The dataset's labels after the label audit in `audit/` (88 corrections; on test, 69). Every test comment labelled spam was reviewed against `policy/spam_policy.md`, plus 100 random test ham and the audit flags.
- **Original labels.** The labels as published with the dataset.

The primary label set was fixed before any test score (decision 46). The corrected labels and the LLM prompts both come from the same policy and the same author, so the corrected labels may favour the LLM; the original labels are always shown beside them.

## 2. Systems and their spam decision

Five systems score every comment. Each one's spam/ham decision uses its own cut-off, fixed before it was scored on dev (decision 48):

| System | Score | Spam when |
|---|---|---|
| Keyword rules (`models/rules.py`) | rule families hit, 0 to 6 | at least 1 |
| TF-IDF + logistic regression (`models/logreg.py`) | probability of spam | at least 0.5 |
| Gradient boosting on 27 features (`models/tree.py`) | probability of spam | at least 0.5 |
| LLM v1 (`models/prompts/v1.md`, config `v1-f9699d88f4b0`) | 5 votes + 0.99 × mean p_spam | at least 3 of 5 votes |
| LLM v2 (`models/prompts/v2.md`, config `v2-646be57ed779`) | as v1 | at least 3 of 5 votes |

The baselines are fitted on the train split with its original labels. Both LLM versions use `global.anthropic.claude-haiku-4-5-20251001-v1:0` on Amazon Bedrock (ap-south-1) at temperature 1.0, 5 votes per comment, the answer checks and invalid-vote rule of `models/llm.py` (decision 40), and the configuration hashes above. Those hashes cover the prompt text, the policy sections in it, v2's examples and normalisation, the model, region, temperature, output limit and the pinned data file (SHA-256 `bd6182891adb3cfc8334b82c062176dfbebc563bf0ba07e31c2645f916865a0a`).

## 3. Hypotheses

- **H1.** LLM v2's test error rate differs from LLM v1's.
- **H2.** LLM v2's test error rate differs from logistic regression's.

Each null hypothesis is that the two systems are equally likely to be the one that errs on a comment where exactly one of them errs. An error is a spam/ham decision (section 2) that differs from the label.

- **Test:** exact two-sided McNemar test on the paired decisions over the 885 scored comments: p = min(1, 2 × P(X ≤ min(b, c))) with X ~ Binomial(b + c, ½), where b and c count the comments only one system gets right (`eval/mcnemar.py`, checked against statsmodels).
- **Primary analysis:** corrected labels, all 885 scored comments.
- **Multiple tests:** Holm's adjustment across H1 and H2 (decision 49). A null hypothesis is rejected when its Holm-adjusted p is below **α = 0.05**. Raw p-values are reported beside the adjusted ones.
- **Effect size:** F1 of each system (primary metric) and the F1 difference (v2 minus the other) with a paired percentile bootstrap 95% interval (2,000 resamples of comments, seed 20261125).
- A null result is reported as found. No hypothesis, label set, comment set, threshold or test is changed after the test scores are seen.

**What dev suggests, stated so the result can be read against it.** On dev (370 comments), v1 and v2 disagree on 1 comment (v2 right), so H1 is unlikely to reach significance on test: with Holm, H1 needs about 6 or more discordant comments, nearly all in one direction (6 to 0 gives p = 0.031). On dev, v2 and logistic regression disagree on 28 comments, all with v2 right (p = 7 × 10⁻⁹).

## 4. Secondary results (descriptive, not adjusted)

The same paired tests and F1 differences are reported, unadjusted and labelled secondary, for:

- the original labels;
- the comments without YouTube's share-text family (`data/share_text.py`; 153 of the scored test comments, decision 14);
- the comments without the 4 test comments the policy text quotes (`LMFAO:349`, `LMFAO:278`, `LMFAO:360`, `LMFAO:181`; decision 42).

For every system, on each label set and comment set: true and false positives and negatives; precision, recall and false-positive rate with Wilson 95% intervals; F1 with a bootstrap 95% interval; average precision. On all scored comments also recall by subtype (from the audit's review answers) and by video, each with a Wilson interval.

**Enforcement tiers** (`eval/thresholds.json`, chosen on dev with the corrected labels and applied unchanged; decisions 7, 8, 45, 46):

| System | Auto-remove at score ≥ | Allow below |
|---|---|---|
| Rules | 1.5 (2 or more hits) | 0.0 (no allow tier) |
| Logistic regression | 0.7213070285874905 | 0.3608377571055401 |
| Gradient boosting | 0.9999994502428498 | 0.00030871908290460294 |
| LLM v1 | 3.0 | 3.0 (no review tier) |
| LLM v2 | 3.0 | 3.0 (no review tier) |

On test, for each system: the share of spam auto-removed, auto-removal precision with its Wilson interval, recall of remove plus review, and auto-removals and review-queue size per 1,000 comments. The 98% precision target was met on dev; the test precision is reported as found, and missing the target is a result, not a reason to move a threshold.

**Running cost:** input and output tokens and model time per 1,000 comments for each LLM version, from the vote caches; US dollars per 1,000 comments at the Bedrock price for this model on the run date, with the price and its source stated.

## 5. Procedure for the single run

1. This file, `eval/lock.py` and the analysis code are committed; CI passes.
2. `python -m models.llm run --prompt v1 --split test --final`, then the same with `--prompt v2`. Votes go to `results/llm/<config>/r0/calls-test.jsonl`; the dev caches do not change. A run that stops part-way is resumed with the same command (same configuration, missing votes only); that is one run, not a repeat. Votes that fail the answer checks twice are invalid and handled by decision 40; no vote is re-asked by hand.
3. The two caches are committed.
4. `python -m eval.final --split test --final` writes `results/final/test.json` and `results/final/test_predictions.csv`, which are committed unchanged.
5. Nothing that judges the test split (labels, policy, prompts, thresholds, cut-offs, analysis code) changes after step 2. A bug found later is fixed in a new commit and both results are reported, with the reason.

The same analysis has been run on dev (`results/final/dev.json`), so the code that produces the test results is the code already checked on dev.

## 6. What was known before this run

- **Dev results.** Every system was scored on dev, and the thresholds were chosen there (`results/final/dev.json`).
- **Baselines on test during the audit.** The rules and logistic regression were run on test to flag comments where both disagreed with the label (38 test flags). Their test metrics were not computed. Flags led to label corrections, so on the corrected labels the audit can favour those two baselines; on test, every comment labelled spam was reviewed whatever the flags said, which limits that for spam judged not spam.
- **Policy examples.** The policy quotes 4 test comments, written while reviewing test labels (decision 42); hence the secondary result without them.
- **The LLMs have never been run on test**, and the gradient-boosting model's test scores have never been looked at.
- **Label consistency.** A blind re-check of 20 audit flags, 2 days after the first review, gave the same final label as the settled answers on 20 of 20 (Wilson 0.84 to 1.00). That is one labeller's consistency, not agreement between labellers.

## 7. Changes from the original plan

- The plan's "F1 at the dev-chosen threshold" predates the three-tier rule. The hypotheses use each system's own spam cut-off (decision 48); the dev-chosen thresholds set the enforcement tiers. For the LLMs both are 3 votes (decision 45).
- Holm's adjustment across H1 and H2 was added (decision 49).
- A gradient-boosting baseline was added to the systems (decision 25); it is in the secondary results only.

Any deviation from this file is listed in the README with its reason.
