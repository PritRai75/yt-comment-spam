import numpy as np
import pytest
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.multitest import multipletests

from eval import mcnemar as M
from eval.labels import label_table, labels, scored
from eval.metrics import f1_from, predict
from models import llm


def _table(y, a, b):
    ra, rb = a == y, b == y
    return [[(ra & rb).sum(), (ra & ~rb).sum()], [(~ra & rb).sum(), (~ra & ~rb).sum()]]


@pytest.mark.parametrize("seed", range(40))
def test_mcnemar_matches_statsmodels_exact(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 300))
    y = rng.integers(0, 2, n)
    a = np.where(rng.random(n) < rng.uniform(0.6, 1.0), y, 1 - y)
    b = np.where(rng.random(n) < rng.uniform(0.6, 1.0), y, 1 - y)
    r = M.mcnemar(y, a, b)
    expected = sm_mcnemar(_table(y, a, b), exact=True).pvalue
    assert r["p"] == pytest.approx(expected, abs=1e-12)
    assert r["both_right"] + r["a_only_right"] + r["b_only_right"] + r["both_wrong"] == n


@pytest.mark.parametrize("b,c,p", [(0, 0, 1.0), (1, 0, 1.0), (5, 0, 0.0625), (6, 0, 0.03125), (3, 3, 1.0), (0, 6, 0.03125)])
def test_exact_p_known_values(b, c, p):
    assert M.exact_p(b, c) == pytest.approx(p, abs=1e-12)


def test_discordant_counts_are_directional():
    y = np.array([1, 1, 0, 0, 1])
    a = np.array([1, 1, 0, 1, 0])  # wrong on 3, 4
    b = np.array([1, 0, 1, 1, 1])  # wrong on 1, 2, 3
    r = M.mcnemar(y, a, b)
    assert (r["both_right"], r["a_only_right"], r["b_only_right"], r["both_wrong"]) == (1, 2, 1, 1)
    assert (r["errors_a"], r["errors_b"]) == (2, 3)


@pytest.mark.parametrize("seed", range(20))
def test_holm_matches_statsmodels(seed):
    p = np.random.default_rng(seed).random(int(seed % 5) + 1) ** 3
    assert M.holm(p) == pytest.approx(list(multipletests(p, method="holm")[1]), abs=1e-12)


def test_holm_two_tests():
    assert M.holm([0.03, 0.01]) == pytest.approx([0.03, 0.02])
    assert M.holm([0.6, 0.01]) == pytest.approx([0.6, 0.02])


def test_f1_difference_point_and_paired_interval():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    a = np.where(rng.random(300) < 0.9, y, 1 - y)
    b = np.where(rng.random(300) < 0.75, y, 1 - y)
    r = M.f1_difference(y, a, b)
    def f1(p):
        return f1_from(((p == 1) & (y == 1)).sum(), ((p == 1) & (y == 0)).sum(), ((p == 0) & (y == 1)).sum())
    assert r["difference"] == pytest.approx(f1(a) - f1(b))
    assert r["lo"] < r["difference"] < r["hi"]
    same = M.f1_difference(y, a, a)
    assert same["difference"] == same["lo"] == same["hi"] == 0.0  # paired: identical systems never differ
    assert M.f1_difference(y, a, b) == r  # seeded


def test_v1_and_v2_on_dev_differ_on_one_comment():
    rows = scored(label_table(), "dev")
    y = labels(rows, "corrected")
    dev = llm._rows("dev")
    pred = {v: predict(llm.scores(dev, llm.load_prompt(v)).score.reindex(rows.index), 3.0) for v in ("v1", "v2")}
    r = M.mcnemar(y, pred["v2"], pred["v1"])
    assert (r["a_only_right"], r["b_only_right"], r["p"]) == (1, 0, 1.0)  # Shakira:003
