"""The committed test results: recomputed from the committed inputs, and pinned.

The test split was scored once (eval/preregistration.md); these tests only read the
committed result files and recompute them, so any change to the code, the data or a
result file that would alter a reported test number fails here.
"""
import json

from adversarial import redteam as R
from eval import final


def test_committed_test_analysis_matches_a_fresh_analysis():
    assert final.main(["--split", "test", "--final", "--check"]) == 0


def test_committed_red_team_matches_a_fresh_analysis():
    assert R.main(["analyse", "--split", "test", "--check"]) == 0


def test_test_results_as_reported():
    r = json.loads((final.OUT / "test.json").read_text(encoding="utf-8"))
    assert r["comments_scored"] == 885 and r["primary_label_set"] == "corrected"
    main = r["results"]["corrected"]["all"]
    assert main["spam"] == 413
    m = {k: main["systems"][k]["metrics"] for k in final.SYSTEMS}
    assert [(m[k]["tp"], m[k]["fp"], m[k]["fn"]) for k in final.SYSTEMS] == [
        (404, 33, 9), (397, 28, 16), (404, 39, 9), (410, 16, 3), (407, 10, 6)]
    h1, h2 = main["hypotheses"]["H1"], main["hypotheses"]["H2"]
    assert (h1["a_only_right"], h1["b_only_right"], h1["reject_at_alpha"]) == (6, 3, False)
    assert (h2["a_only_right"], h2["b_only_right"], h2["reject_at_alpha"]) == (36, 8, True)
    orig = r["results"]["original"]["all"]["hypotheses"]
    assert not orig["H1"]["reject_at_alpha"] and not orig["H2"]["reject_at_alpha"]


def test_red_team_as_reported():
    r = json.loads((R.OUT / "test.json").read_text(encoding="utf-8"))
    assert r["sampled"] == 100 and sum(r["variants_by_transform"].values()) == 509
    evaded = {k: v["all_transforms"]["evaded"] for k, v in r["results"].items()}
    assert evaded == {"rules_as_is": 256, "rules_normalised": 86, "logreg_as_is": 287, "logreg_normalised": 114,
                      "tree_as_is": 260, "tree_normalised": 93, "llm_v2_normalised": 9}
    assert r["llm_usage"]["calls"] == 1840 and r["llm_usage"]["invalid"] == 0
