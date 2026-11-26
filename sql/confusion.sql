-- Confusion counts and metrics for every system, label set and split, from the
-- predictions table (results/final/<split>_predictions.csv, loaded into BigQuery as
-- yt_comment_spam.predictions). Each system's prediction is at its own spam cut-off.
-- Written to run unchanged in BigQuery and SQLite.
WITH long AS (
  SELECT split, 'rules' AS system, rules_pred AS pred, label_corrected, label_original
  FROM yt_comment_spam.predictions
  UNION ALL
  SELECT split, 'logreg', logreg_pred, label_corrected, label_original FROM yt_comment_spam.predictions
  UNION ALL
  SELECT split, 'tree', tree_pred, label_corrected, label_original FROM yt_comment_spam.predictions
  UNION ALL
  SELECT split, 'llm_v1', llm_v1_pred, label_corrected, label_original FROM yt_comment_spam.predictions
  UNION ALL
  SELECT split, 'llm_v2', llm_v2_pred, label_corrected, label_original FROM yt_comment_spam.predictions
),
labelled AS (
  SELECT split, system, pred, 'corrected' AS label_set, label_corrected AS label FROM long
  UNION ALL
  SELECT split, system, pred, 'original', label_original FROM long
),
counts AS (
  SELECT
    split,
    label_set,
    system,
    SUM(CASE WHEN pred = 1 AND label = 1 THEN 1 ELSE 0 END) AS tp,
    SUM(CASE WHEN pred = 1 AND label = 0 THEN 1 ELSE 0 END) AS fp,
    SUM(CASE WHEN pred = 0 AND label = 1 THEN 1 ELSE 0 END) AS fn,
    SUM(CASE WHEN pred = 0 AND label = 0 THEN 1 ELSE 0 END) AS tn
  FROM labelled
  GROUP BY split, label_set, system
)
SELECT
  split,
  label_set,
  system,
  tp,
  fp,
  fn,
  tn,
  1.0 * tp / NULLIF(tp + fp, 0) AS precision,
  1.0 * tp / NULLIF(tp + fn, 0) AS recall,
  2.0 * tp / NULLIF(2 * tp + fp + fn, 0) AS f1,
  1.0 * fp / NULLIF(fp + tn, 0) AS fpr
FROM counts
ORDER BY split, label_set, system;
