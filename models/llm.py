"""LLM classifier on Amazon Bedrock: prompt, structured answer, 5-vote score, cache.

One call sends the prompt (`models/prompts/<version>.md`, with the spam policy in the
system prompt) and one cleaned comment, and forces the model to answer through a tool,
`record_verdict`, with four fields: label (spam or ham), subtype, confidence (50-100)
and a one-line reason. The answer is checked in code before it is used:

  - label is "spam" or "ham"; subtype is one of the four for spam and "none" for ham;
  - confidence is a whole number from 50 to 100 (below 50 would mean the model thinks
    the other label is more likely, so the fields disagree);
  - reason is one non-empty line of at most 300 characters;
  - the model answered through the tool, once, with no other fields.
An answer that fails any check is asked again once (a fresh sample at temperature 1.0);
if the second also fails, that vote is recorded as invalid with the failed check.

Score (decision 6, with invalid votes per decision 40): each comment gets 5 votes at
temperature 1.0. p_spam is confidence/100 for a spam vote and 1 - confidence/100 for a
ham vote. Over the valid votes,

    score = 5 x (share of valid votes that are spam) + 0.99 x (mean p_spam)

which is spam votes + 0.99 x mean p_spam when all 5 are valid, so it runs from 0 to
5.99 and every extra spam vote outranks any confidence. A comment with no valid vote
scores 0 and is counted in the run summary.

One configuration, hashed to 12 characters, is carried by every call: prompt version
and the SHA-256 of the rendered system prompt (so the policy text is in it), model ID,
region, temperature, maximum output tokens and the SHA-256 of the pinned data zip.

Cache: one JSON line per vote in results/llm/<version>-<hash>/r<replicate>/calls.jsonl,
keyed by (configuration, video, cleaned text, vote index). A run that stops part-way
resumes where it stopped and pays only for missing votes. Comments with the same
cleaned text under the same video get the same prompt, so they share their votes
(decision 41): the score is a function of what the model is shown, as for the other
four systems (on the pinned data, 243 comments in 52 such groups, mostly share text). A line holds IDs, the answer
fields, token counts and latency, never the comment text; the reason is the model's
own line and may quote the comment. `--replicate 1` repeats a run with new samples
(the noise check before comparing prompt versions).

Comment text is untrusted data: it is passed as a JSON string inside the user message,
and the prompt says that nothing in it is an instruction.

    python -m models.llm run --split dev --limit 5      # try a few first
    python -m models.llm run --split dev                # all of dev, 5 votes each
    python -m models.llm run --split dev --fake         # offline, no AWS
    python -m models.llm summary --split dev

The test split is refused until the test lock exists (step 6).
"""
import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from data.clean import clean_text
from data.fetch import ROOT
from data.load import load_comments
from models import rules

MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"  # decision 19
REGION = "ap-south-1"
TEMPERATURE = 1.0  # decision 6
MAX_TOKENS = 300
N_VOTES = 5
RPM = 40  # under the 50-a-minute quota, which is shared per model across the account
TOOL_NAME = "record_verdict"
SUBTYPES = ["scam", "link", "self_promotion", "bulk"]
MAX_REASON = 300

PROMPTS = ROOT / "models" / "prompts"
POLICY = ROOT / "policy" / "spam_policy.md"
VIDEOS = ROOT / "data" / "videos.json"
MANIFEST = ROOT / "data" / "manifest.json"
RUNS = ROOT / "results" / "llm"
FAKE_RUNS = ROOT / "results" / "llm-fake"  # offline runs, not committed
POLICY_SECTIONS = ["Definition", "How to judge", "Subtypes", "Which links count", "Edge cases"]
ALLOWED_SPLITS = ["dev"]  # the test split is added with the test lock (step 6)


# Prompt

def policy_text(path: Path = POLICY) -> str:
    """The policy sections the prompts carry, verbatim, in the policy's order."""
    parts = re.split(r"(?m)^## ", path.read_text(encoding="utf-8"))
    found = {p.split("\n", 1)[0].strip(): "## " + p.rstrip() for p in parts[1:]}
    missing = [s for s in POLICY_SECTIONS if s not in found]
    if missing:
        raise ValueError(f"policy sections missing: {missing}")
    return "\n\n".join(found[s] for s in POLICY_SECTIONS)


@dataclass(frozen=True)
class Prompt:
    version: str
    system: str
    user_template: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256((self.system + "\x00" + self.user_template).encode("utf-8")).hexdigest()

    def user(self, video: str, text: str) -> str:
        return (self.user_template.replace("{{VIDEO}}", describe_video(video))
                .replace("{{COMMENT_JSON}}", json.dumps(text, ensure_ascii=False)))


def load_prompt(version: str = "v1", policy: Path = POLICY) -> Prompt:
    raw = (PROMPTS / f"{version}.md").read_text(encoding="utf-8")
    try:
        _, rest = raw.split("=== SYSTEM ===\n", 1)
        system, user = rest.split("=== USER ===\n", 1)
    except ValueError:
        raise ValueError(f"{version}.md needs '=== SYSTEM ===' and '=== USER ===' lines") from None
    if system.count("{{POLICY}}") != 1 or "{{COMMENT_JSON}}" not in user or "{{VIDEO}}" not in user:
        raise ValueError(f"{version}.md placeholders missing")
    return Prompt(version, system.strip().replace("{{POLICY}}", policy_text(policy)), user.strip())


def describe_video(video: str) -> str:
    v = json.loads(VIDEOS.read_text(encoding="utf-8"))["videos"][video]
    return f"{v['artist']} - {v['song']} (video \"{video}\" in the policy's list, YouTube ID {v['youtube_id']})"


def tool_spec() -> dict:
    return {"toolSpec": {
        "name": TOOL_NAME,
        "description": "Record the spam verdict for the comment under the policy.",
        "inputSchema": {"json": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": ["spam", "ham"]},
                "subtype": {"type": "string", "enum": SUBTYPES + ["none"]},
                "confidence": {"type": "integer", "minimum": 50, "maximum": 100},
                "reason": {"type": "string", "description": "One line, at most 25 words"},
            },
            "required": ["label", "subtype", "confidence", "reason"],
            "additionalProperties": False,
        }},
    }}


# Configuration

@dataclass(frozen=True)
class Config:
    prompt_version: str
    prompt_sha256: str
    model_id: str
    region: str
    temperature: float
    max_tokens: int
    data_sha256: str

    @property
    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

    @property
    def name(self) -> str:
        return f"{self.prompt_version}-{self.hash}"


def make_config(prompt: Prompt) -> Config:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))["zip"]["sha256"]
    return Config(prompt.version, prompt.sha256, MODEL_ID, REGION, TEMPERATURE, MAX_TOKENS, data)


def request(config: Config, prompt: Prompt, video: str, text: str) -> dict:
    """Keyword arguments for bedrock-runtime `converse`. Temperature only: Claude models
    on Bedrock refuse temperature and top_p together."""
    return {
        "modelId": config.model_id,
        "system": [{"text": prompt.system}],
        "messages": [{"role": "user", "content": [{"text": prompt.user(video, text)}]}],
        "inferenceConfig": {"temperature": config.temperature, "maxTokens": config.max_tokens},
        "toolConfig": {"tools": [tool_spec()], "toolChoice": {"tool": {"name": TOOL_NAME}}},
    }


# Answers

class InvalidAnswer(Exception):
    pass


def parse(response: dict) -> dict:
    """The checked answer from a converse response; InvalidAnswer names the failed check."""
    content = response.get("output", {}).get("message", {}).get("content", [])
    uses = [c["toolUse"] for c in content if "toolUse" in c]
    if len(uses) != 1:
        raise InvalidAnswer(f"tool_calls={len(uses)}")
    if uses[0].get("name") != TOOL_NAME:
        raise InvalidAnswer("tool_name")
    a = uses[0].get("input")
    if not isinstance(a, dict):
        raise InvalidAnswer("input_not_object")
    if set(a) != {"label", "subtype", "confidence", "reason"}:
        raise InvalidAnswer("fields")
    label, subtype, conf, reason = a["label"], a["subtype"], a["confidence"], a["reason"]
    if label not in ("spam", "ham"):
        raise InvalidAnswer("label")
    if label == "spam" and subtype not in SUBTYPES:
        raise InvalidAnswer("spam_without_subtype")
    if label == "ham" and subtype != "none":
        raise InvalidAnswer("ham_with_subtype")
    if isinstance(conf, bool) or not isinstance(conf, int):
        raise InvalidAnswer("confidence_not_integer")
    if not 50 <= conf <= 100:
        raise InvalidAnswer("confidence_range")
    if not isinstance(reason, str) or not reason.strip():
        raise InvalidAnswer("reason_empty")
    if "\n" in reason or len(reason) > MAX_REASON:
        raise InvalidAnswer("reason_not_one_line")
    return {"label": label, "subtype": subtype, "confidence": conf, "reason": reason.strip()}


def p_spam(label: str, confidence: int) -> float:
    return confidence / 100 if label == "spam" else 1 - confidence / 100


def vote_score(votes) -> dict:
    """Score from a comment's votes (dicts with valid, label, confidence)."""
    valid = [v for v in votes if v["valid"]]
    if not valid:
        return {"n_votes": len(votes), "n_valid": 0, "spam_votes": 0, "mean_p_spam": float("nan"), "score": 0.0}
    spam = sum(v["label"] == "spam" for v in valid)
    mean_p = sum(p_spam(v["label"], v["confidence"]) for v in valid) / len(valid)
    return {"n_votes": len(votes), "n_valid": len(valid), "spam_votes": spam, "mean_p_spam": mean_p,
            "score": N_VOTES * spam / len(valid) + 0.99 * mean_p}


# Cache and run

def call_key(config: Config, video: str, text: str, vote: int) -> str:
    h = hashlib.sha256()
    for part in (config.hash, video, text, str(vote)):
        h.update(part.encode("utf-8") + b"\x00")
    return h.hexdigest()[:20]


class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.records = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.records[r["key"]] = r

    def __contains__(self, key):
        return key in self.records

    def add(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        self.records[record["key"]] = record


class Pacer:
    """At most `rpm` call starts a minute."""

    def __init__(self, rpm: float, clock=time.monotonic, sleep=time.sleep):
        self.interval = 60.0 / rpm if rpm else 0.0
        self.clock, self.sleep, self.next = clock, sleep, None

    def wait(self):
        now = self.clock()
        if self.next is not None and now < self.next:
            self.sleep(self.next - now)
            now = self.next
        self.next = now + self.interval


def run_dir(config: Config, replicate: int = 0) -> Path:
    return RUNS / config.name / f"r{replicate}"


def one_vote(client, config, prompt, video, text):
    """Ask once, and once more if the answer fails a check. Returns the vote record fields."""
    tokens_in = tokens_out = latency = 0
    failed = None
    for attempt in (1, 2):
        resp = client.converse(**request(config, prompt, video, text))
        usage = resp.get("usage", {})
        tokens_in += usage.get("inputTokens", 0)
        tokens_out += usage.get("outputTokens", 0)
        latency += resp.get("metrics", {}).get("latencyMs", 0)
        try:
            answer = parse(resp)
        except InvalidAnswer as e:
            failed = str(e)
            continue
        return dict(answer, valid=True, attempts=attempt, error="",
                    input_tokens=tokens_in, output_tokens=tokens_out, latency_ms=latency)
    return {"label": "", "subtype": "", "confidence": None, "reason": "", "valid": False, "attempts": 2,
            "error": failed, "input_tokens": tokens_in, "output_tokens": tokens_out, "latency_ms": latency}


def run(rows: pd.DataFrame, client, prompt: Prompt, *, votes: int = N_VOTES, replicate: int = 0,
        rpm: float = RPM, limit: int = None, cache: Cache = None, pacer: Pacer = None, log=print) -> dict:
    """Fill the cache for `rows` (row_id, video, split, content). Returns counts for the run."""
    bad = sorted(set(rows.split) - set(ALLOWED_SPLITS))
    if bad:
        raise ValueError(f"split(s) {bad} refused: only {ALLOWED_SPLITS} until the test lock exists")
    config = make_config(prompt)
    cache = cache or Cache(run_dir(config, replicate) / "calls.jsonl")
    pacer = pacer or Pacer(rpm)
    if limit is not None:
        rows = rows.head(limit)
    made = cached = invalid = 0  # cached: votes already in the cache, incl. repeated texts
    started = time.monotonic()
    for i, r in enumerate(rows.itertuples(), 1):
        text = clean_text(r.content)
        for vote in range(votes):
            key = call_key(config, r.video, text, vote + N_VOTES * replicate)
            if key in cache:
                cached += 1
                continue
            pacer.wait()
            rec = one_vote(client, config, prompt, r.video, text)
            rec.update(key=key, row_id=r.row_id, vote=vote, replicate=replicate, config=config.hash)
            cache.add(rec)
            made += 1
            invalid += not rec["valid"]
        if i % 25 == 0:
            log(f"{i}/{len(rows)} comments, {made} calls made, {cached} cached, {invalid} invalid")
    return {"config": config.name, "comments": len(rows), "calls_made": made, "cached": cached,
            "invalid": invalid, "seconds": round(time.monotonic() - started, 1), "path": str(cache.path)}


def scores(rows: pd.DataFrame, prompt: Prompt, *, replicate: int = 0, votes: int = N_VOTES,
           cache: Cache = None) -> pd.DataFrame:
    """One row per comment with its votes and score, from the cache."""
    config = make_config(prompt)
    cache = cache or Cache(run_dir(config, replicate) / "calls.jsonl")
    out = []
    for r in rows.itertuples():
        text = clean_text(r.content)
        recs = [cache.records.get(call_key(config, r.video, text, v + N_VOTES * replicate)) for v in range(votes)]
        if any(x is None for x in recs):
            raise ValueError(f"{r.row_id}: votes missing from the cache; run first")
        s = vote_score(recs)
        spam_subtypes = [x["subtype"] for x in recs if x["valid"] and x["label"] == "spam"]
        s["subtype"] = max(SUBTYPES, key=spam_subtypes.count) if spam_subtypes else ""
        out.append(dict(row_id=r.row_id, **s))
    return pd.DataFrame(out).set_index("row_id")


def usage(cache: Cache) -> dict:
    recs = list(cache.records.values())
    return {"calls": len(recs), "attempts": sum(r["attempts"] for r in recs),
            "invalid": sum(not r["valid"] for r in recs),
            "input_tokens": sum(r["input_tokens"] for r in recs),
            "output_tokens": sum(r["output_tokens"] for r in recs),
            "latency_s": round(sum(r["latency_ms"] for r in recs) / 1000, 1)}


# Clients

def make_client():
    """The real Bedrock client (needs AWS credentials, e.g. `aws login`)."""
    import boto3
    from botocore.config import Config as BotoConfig
    return boto3.client("bedrock-runtime", region_name=REGION,
                        config=BotoConfig(retries={"max_attempts": 10, "mode": "adaptive"}))


class FakeClient:
    """Offline stand-in for tests and dry runs: answers from the rules baseline, with a
    vote flipped now and then so the 5 votes are not all alike. `answer` can be replaced
    to return any tool input (or a whole response) for a given call."""

    def __init__(self, answer=None):
        self.calls = []
        self.answer = answer or self.default_answer

    @staticmethod
    def default_answer(text: str, n: int) -> dict:
        hits = rules.rule_hits(text)
        h = int(hashlib.sha256(f"{text}|{n}".encode()).hexdigest(), 16)
        spam = bool(hits) != (h % 7 == 0)
        order = {"money": "scam", "link": "link", "self_promo": "self_promotion", "subscribe": "self_promotion"}
        subtype = next((order[x] for x in order if x in hits), "bulk") if spam else "none"
        return {"label": "spam" if spam else "ham", "subtype": subtype,
                "confidence": 55 + h % 45, "reason": "fake answer from the rules baseline"}

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        user = kwargs["messages"][0]["content"][0]["text"]
        text = json.loads(user.rsplit("\n", 1)[1])
        out = self.answer(text, len(self.calls))
        if "output" in out:
            return out
        n_in = (len(kwargs["system"][0]["text"]) + len(user)) // 4
        return {
            "output": {"message": {"role": "assistant", "content": [
                {"toolUse": {"toolUseId": f"tooluse_{len(self.calls)}", "name": TOOL_NAME, "input": out}}]}},
            "stopReason": "tool_use",
            "usage": {"inputTokens": n_in, "outputTokens": 40, "totalTokens": n_in + 40},
            "metrics": {"latencyMs": 0},
        }


# Command line

def _rows(split: str) -> pd.DataFrame:
    df = load_comments()
    return df[df.split == split].reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "summary"):
        p = sub.add_parser(name)
        p.add_argument("--prompt", default="v1")
        p.add_argument("--split", choices=ALLOWED_SPLITS, required=True)
        p.add_argument("--replicate", type=int, default=0)
        p.add_argument("--votes", type=int, default=N_VOTES)
        p.add_argument("--fake", action="store_true", help="offline fake client; writes under results/llm-fake/")
        if name == "run":
            p.add_argument("--limit", type=int, default=None, help="only the first N comments")
            p.add_argument("--rpm", type=float, default=RPM)
    args = ap.parse_args(argv)
    if not 1 <= args.votes <= N_VOTES:
        ap.error(f"--votes must be 1 to {N_VOTES}")
    prompt = load_prompt(args.prompt)
    config = make_config(prompt)
    base = FAKE_RUNS if args.fake else RUNS
    path = base / config.name / f"r{args.replicate}" / "calls.jsonl"
    cache = Cache(path)
    rows = _rows(args.split)
    if args.cmd == "run":
        cfg_file = path.parent.parent / "config.json"
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(json.dumps(dict(asdict(config), hash=config.hash), indent=2) + "\n", encoding="utf-8")
        client = FakeClient() if args.fake else make_client()
        print(json.dumps(run(rows, client, prompt, votes=args.votes, replicate=args.replicate, rpm=args.rpm,
                             limit=args.limit, cache=cache, pacer=Pacer(0 if args.fake else args.rpm))))
    u = usage(cache)
    print(f"{config.name} r{args.replicate}: {u['calls']} calls cached ({u['attempts']} attempts, "
          f"{u['invalid']} invalid), {u['input_tokens']:,} input and {u['output_tokens']:,} output tokens, "
          f"{u['latency_s']} s of model time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
