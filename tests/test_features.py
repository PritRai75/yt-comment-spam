import pytest

from data.load import load_comments
from models import rules
from models.features import FEATURES, RULE_COLS, feature_frame, features

DF = load_comments()


def test_27_named_features():
    assert len(FEATURES) == 27
    assert len(set(FEATURES)) == 27
    assert list(features("hello")) == FEATURES


def test_a_typical_plug():
    f = features("Check out my channel http://x.com/abc !!!")
    assert f["n_urls"] == 1
    assert f["n_exclaim"] == 3
    assert f["first_singular"] == 1
    assert f["request_verbs"] == 1
    assert f["rule_self_promo"] == 1 and f["rule_link"] == 1 and f["rule_money"] == 0


@pytest.mark.parametrize("text,name,value", [
    ("love it 😍😍", "n_emoji", 2),
    ("I ♥ this", "n_emoji", 1),
    ("+Shakira you rock", "n_mentions", 1),
    ("@katy hi", "n_mentions", 1),
    ("mail me at a@b.com", "n_mentions", 0),
    ("#love #music", "n_hashtags", 2),
    ("call 555-123-4567", "has_long_number", 1),
    ("watching in 2015", "has_long_number", 0),
    ("earn $50 now", "has_currency", 1),
    ("ABC def", "caps_share", 0.5),
    ("sooooo good", "max_char_run", 5),
    ("line one<br />line two", "n_lines", 2),
    ("www.site.com and http://a.org/x", "n_urls", 2),
    ("we love our fans", "first_plural", 2),
    ("u should see ur face", "second_person", 2),
    ("great song, nice beat", "song_words", 2),
])
def test_single_features(text, name, value):
    assert features(text)[name] == pytest.approx(value)


def test_cleaning_comes_first():
    # entities are decoded before counting: &#39; is an apostrophe, not a hashtag
    f = features("I&#39;m here &lt;3")
    assert f["n_hashtags"] == 0
    assert f["first_singular"] == 1


def test_empty_comment():
    f = features("")
    assert all(v == 0 for v in f.values())


def test_rule_columns_match_the_rules_baseline():
    X = feature_frame(DF.content)
    for col, name in zip(RULE_COLS, rules.RULES):
        expected = DF.content.map(lambda t: name in rules.rule_hits(t)).astype(float).to_numpy()
        assert (X[col].to_numpy() == expected).all(), col


def test_feature_frame_shape():
    X = feature_frame(DF.content.head(10))
    assert X.shape == (10, 27)
    assert list(X.columns) == FEATURES
    assert (X.dtypes == float).all()
