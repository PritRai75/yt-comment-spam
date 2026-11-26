-- Recall and precision by video for every system, on the corrected labels (the
-- primary label set), from yt_comment_spam.predictions.
-- Written to run unchanged in BigQuery and SQLite.
WITH long AS (
  SELECT split, video, 'rules' AS system, rules_pred AS pred, label_corrected AS label
  FROM yt_comment_spam.predictions
  UNION ALL
  SELECT split, video, 'logreg', logreg_pred, label_corrected FROM yt_comment_spam.predictions
  UNION ALL
  SELECT split, video, 'tree', tree_pred, label_corrected FROM yt_comment_spam.predictions
  UNION ALL
  SELECT split, video, 'llm_v1', llm_v1_pred, label_corrected FROM yt_comment_spam.predictions
  UNION ALL
  SELECT split, video, 'llm_v2', llm_v2_pred, label_corrected FROM yt_comment_spam.predictions
)
SELECT
  split,
  video,
  system,
  COUNT(*) AS comments,
  SUM(label) AS spam,
  SUM(pred) AS flagged,
  SUM(CASE WHEN pred = 1 AND label = 1 THEN 1 ELSE 0 END) AS caught,
  1.0 * SUM(CASE WHEN pred = 1 AND label = 1 THEN 1 ELSE 0 END) / NULLIF(SUM(label), 0) AS recall,
  1.0 * SUM(CASE WHEN pred = 1 AND label = 1 THEN 1 ELSE 0 END) / NULLIF(SUM(pred), 0) AS precision
FROM long
GROUP BY split, video, system
ORDER BY split, video, system;
