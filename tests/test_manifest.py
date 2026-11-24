import json

import pytest

from data import fetch

MANIFEST = json.loads(fetch.MANIFEST.read_text(encoding="utf-8"))
EXPECTED_MEMBERS = {
    "Youtube01-Psy.csv", "Youtube02-KatyPerry.csv", "Youtube03-LMFAO.csv",
    "Youtube04-Eminem.csv", "Youtube05-Shakira.csv",
}


def test_source_is_uci_with_licence():
    src = MANIFEST["source"]
    assert src["url"] == "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip"
    assert src["licence"] == "CC BY 4.0"
    assert src["doi"] == "10.24432/C58885"


def test_zip_matches_manifest():
    desc = fetch.describe_zip(fetch.RAW_ZIP)
    assert desc["zip"] == MANIFEST["zip"]


def test_csv_members_match_manifest():
    desc = fetch.describe_zip(fetch.RAW_ZIP)
    assert desc["members"] == MANIFEST["members"]
    assert {m["name"].rsplit("/", 1)[-1] for m in desc["members"]} == EXPECTED_MEMBERS


def test_check_mode_passes():
    assert fetch.main(["--check"]) == 0


def test_download_saves_bytes_unchanged(tmp_path, monkeypatch):
    body = b"PK\x03\x04 raw bytes\r\n kept as served"

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return body

    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        return FakeResponse()

    monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "raw" / "x.zip"
    fetch.download("https://example.org/x.zip", dest)
    assert dest.read_bytes() == body
    assert seen["url"] == "https://example.org/x.zip"


@pytest.mark.parametrize("name,expected", [
    ("Youtube01-Psy.csv", True),
    ("YouTube-Spam-Collection-v1/Youtube05-Shakira.csv", True),
    ("__MACOSX/Youtube01-Psy.csv", False),
    ("__MACOSX/._Youtube01-Psy.csv", False),
    ("._Youtube01-Psy.csv", False),
    ("README.txt", False),
])
def test_is_data_member(name, expected):
    assert fetch.is_data_member(name) is expected
