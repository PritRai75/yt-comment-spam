"""The test lock: nothing scores the test split until the rules that judge it are committed.

The test split is opened only when all of these hold:
  1. the caller passes `final=True` (on the command line, `--final`);
  2. `eval/preregistration.md` has the SHA-256 pinned in PREREGISTRATION_SHA256 (text
     read with universal newlines, so a Windows checkout hashes the same);
  3. every file in COMMITTED is tracked by git and unchanged from the last commit:
     the pre-registration, the thresholds, the analysis code and the audit record.
Anything else raises TestLocked, naming what is missing.
"""
import hashlib
import subprocess

from data.fetch import ROOT

PREREGISTRATION = ROOT / "eval" / "preregistration.md"
PREREGISTRATION_SHA256 = "d541b505150de5a3aa03909ac6b9849bc841bf1d9b397d3813497878a48ec498"
COMMITTED = [
    "eval/preregistration.md",
    "eval/thresholds.json",
    "eval/thresholds.py",
    "eval/final.py",
    "eval/mcnemar.py",
    "eval/metrics.py",
    "eval/labels.py",
    "audit/corrections.csv",
    "audit/review_verdicts.csv",
    "audit/subtype_labels.csv",
    "audit/full_review_labels.csv",
    "audit/unsure_resolution.csv",
    "audit/recheck_verdicts.csv",
    "audit/scoring_exclusions.csv",
]


class TestLocked(Exception):
    __test__ = False  # not a pytest test class


def preregistration_sha256() -> str:
    return hashlib.sha256(PREREGISTRATION.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def problems() -> list:
    """Why the test split is still locked (empty when it may be opened with --final)."""
    out = []
    if not PREREGISTRATION.exists():
        out.append("eval/preregistration.md is missing")
    elif preregistration_sha256() != PREREGISTRATION_SHA256:
        out.append("eval/preregistration.md differs from the pinned SHA-256")
    try:
        tracked = set(_git("ls-files", "--", *COMMITTED).stdout.split())
        changed = set(_git("diff", "--name-only", "HEAD", "--", *COMMITTED).stdout.split())
    except FileNotFoundError:
        return out + ["git is not available, so the commits cannot be checked"]
    out += [f"{f} is not committed" for f in COMMITTED if f not in tracked]
    out += [f"{f} has uncommitted changes" for f in COMMITTED if f in changed]
    return out


def require(final: bool) -> None:
    if not final:
        raise TestLocked("the test split needs --final (it is scored once, after the pre-registration)")
    p = problems()
    if p:
        raise TestLocked("test split locked: " + "; ".join(p))
