"""TF-IDF + logistic regression baseline.

Fixed settings, chosen before any dev or test score was seen and not tuned:
word 1-2 grams, min_df 2, sublinear tf, lower-cased cleaned text; liblinear logistic
regression with C = 1.0. The score is the predicted probability of spam; the label is
spam when it is at least 0.5.

Train comments are scored out of fold, one video at a time (the model that scores
Psy was fitted on KatyPerry only, and the reverse), so no comment is scored by a model
that saw it. Dev and test are scored by the model fitted on all of train.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from data.clean import clean_text


def make_model():
    return make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=0),
    )


def fit(texts, labels):
    model = make_model()
    model.fit([clean_text(t) for t in texts], np.asarray(labels))
    return model


def spam_proba(model, texts) -> np.ndarray:
    return model.predict_proba([clean_text(t) for t in texts])[:, 1]


def score_all(df: pd.DataFrame) -> pd.Series:
    """Spam probability for every row: out of fold by video on train, full-train model elsewhere."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    train = df[df.split == "train"]
    for video in sorted(train.video.unique()):
        fit_on = train[train.video != video]
        score_on = train[train.video == video]
        out.loc[score_on.index] = spam_proba(fit(fit_on.content, fit_on.label), score_on.content)
    full = fit(train.content, train.label)
    rest = df[df.split != "train"]
    out.loc[rest.index] = spam_proba(full, rest.content)
    return out
