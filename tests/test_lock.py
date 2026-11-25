import json
import shutil
import subprocess

import pytest

from data.fetch import ROOT
from eval import final, lock


def test_preregistration_hash_is_pinned():
    # Editing eval/preregistration.md locks the test split until this pin is changed too.
    assert lock.preregistration_sha256() == lock.PREREGISTRATION_SHA256


def test_every_locked_file_exists():
    assert all((ROOT / f).exists() for f in lock.COMMITTED)


def test_require_needs_final_then_no_problems(monkeypatch):
    with pytest.raises(lock.TestLocked, match="--final"):
        lock.require(False)
    monkeypatch.setattr(lock, "problems", lambda: ["x is not committed"])
    with pytest.raises(lock.TestLocked, match="x is not committed"):
        lock.require(True)
    monkeypatch.setattr(lock, "problems", lambda: [])
    lock.require(True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    (tmp_path / "eval").mkdir()
    pre = tmp_path / "eval" / "preregistration.md"
    pre.write_text("plan\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "start")
    monkeypatch.setattr(lock, "ROOT", tmp_path)
    monkeypatch.setattr(lock, "PREREGISTRATION", pre)
    monkeypatch.setattr(lock, "PREREGISTRATION_SHA256", lock.preregistration_sha256())
    monkeypatch.setattr(lock, "COMMITTED", ["eval/preregistration.md", "a.txt"])
    return tmp_path, git


def test_problems_track_the_commits(repo):
    path, git = repo
    assert lock.problems() == ["eval/preregistration.md is not committed", "a.txt is not committed"]
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "freeze")
    assert lock.problems() == []
    (path / "a.txt").write_text("changed\n", encoding="utf-8")
    assert lock.problems() == ["a.txt has uncommitted changes"]
    git("checkout", "--", "a.txt")
    lock.PREREGISTRATION.write_text("plan, edited\n", encoding="utf-8")
    p = lock.problems()
    assert "eval/preregistration.md differs from the pinned SHA-256" in p
    assert "eval/preregistration.md has uncommitted changes" in p


def test_staged_but_uncommitted_still_locks(repo):
    _, git = repo
    git("add", ".")
    assert lock.problems() == ["eval/preregistration.md has uncommitted changes", "a.txt has uncommitted changes"]


def test_hash_ignores_windows_line_endings(repo):
    path, _ = repo
    lock.PREREGISTRATION.write_bytes(b"plan\r\n")
    assert lock.preregistration_sha256() == lock.PREREGISTRATION_SHA256


# The analysis that runs on test, rehearsed on dev

def test_final_refuses_test_without_the_open_lock(capsys):
    assert final.main(["--split", "test"]) == 2
    assert "--final" in capsys.readouterr().err


def test_committed_dev_rehearsal_matches_a_fresh_analysis():
    assert final.main(["--split", "dev", "--check"]) == 0


def test_dev_rehearsal_reproduces_the_dev_results():
    r = json.loads((final.OUT / "dev.json").read_text(encoding="utf-8"))
    assert r["comments_scored"] == 370 and r["primary_label_set"] == "corrected"
    main = r["results"]["corrected"]["all"]
    m = {k: main["systems"][k]["metrics"] for k in final.SYSTEMS}
    assert [(m[k]["tp"], m[k]["fp"], m[k]["fn"]) for k in final.SYSTEMS] == [
        (147, 11, 10), (134, 6, 23), (134, 14, 23), (156, 1, 1), (156, 0, 1)]
    h1, h2 = main["hypotheses"]["H1"], main["hypotheses"]["H2"]
    assert (h1["a_only_right"], h1["b_only_right"], h1["p"], h1["reject_at_alpha"]) == (1, 0, 1.0, False)
    assert (h2["a_only_right"], h2["b_only_right"]) == (28, 0)
    assert h2["p_holm"] == pytest.approx(2 * h2["p"]) and h2["reject_at_alpha"]
    assert set(r["results"]["corrected"]) == {"all", "no_share_text", "no_quoted"}
    assert r["results"]["corrected"]["no_quoted"]["comments"] == 368
    assert r["results"]["corrected"]["no_share_text"]["comments"] == 360
    for k in final.SYSTEMS:
        assert main["systems"][k]["enforcement"]["comments"] == 370
