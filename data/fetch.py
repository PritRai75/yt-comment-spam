"""Download the UCI YouTube Spam Collection and pin it.

The zip is saved exactly as served to data/raw/youtube-spam-collection.zip, and
data/manifest.json records the source, licence, size and SHA-256 of the zip and of
each CSV inside it.

    python data/fetch.py            download (if missing) and write the manifest
    python data/fetch.py --check    verify the saved zip against the manifest, no network
"""
import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_ZIP = ROOT / "data" / "raw" / "youtube-spam-collection.zip"
MANIFEST = ROOT / "data" / "manifest.json"

SOURCE = {
    "name": "YouTube Spam Collection",
    "url": "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip",
    "landing_page": "https://archive.ics.uci.edu/dataset/380/",
    "creators": ["T.C. Alberto", "J.V. Lochter"],
    "licence": "CC BY 4.0",
    "doi": "10.24432/C58885",
    "citation": "Alberto, T. & Lochter, J. (2015). YouTube Spam Collection [Dataset]. "
                "UCI Machine Learning Repository. https://doi.org/10.24432/C58885",
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def is_data_member(name: str) -> bool:
    """CSV members of the zip, ignoring macOS metadata folders."""
    base = name.rsplit("/", 1)[-1]
    return (not name.startswith("__MACOSX/") and not base.startswith("._")
            and base.startswith("Youtube0") and base.endswith(".csv"))


def describe_zip(path: Path) -> dict:
    data = path.read_bytes()
    members = []
    with zipfile.ZipFile(path) as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            if info.is_dir() or not is_data_member(info.filename):
                continue
            b = zf.read(info.filename)
            members.append({"name": info.filename, "bytes": len(b), "sha256": sha256_bytes(b)})
    return {
        "zip": {"path": path.relative_to(ROOT).as_posix(), "bytes": len(data), "sha256": sha256_bytes(data)},
        "members": members,
    }


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "yt-comment-spam data fetch"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)


def write_manifest() -> dict:
    if RAW_ZIP.exists():
        print(f"{RAW_ZIP.relative_to(ROOT).as_posix()} already present; not downloading again")
    else:
        print(f"downloading {SOURCE['url']}")
        download(SOURCE["url"], RAW_ZIP)
    desc = describe_zip(RAW_ZIP)
    if len(desc["members"]) != 5:
        sys.exit(f"expected 5 CSV files in the zip, found {len(desc['members'])}")
    manifest = {
        "source": SOURCE,
        "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **desc,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"zip {desc['zip']['bytes']} bytes sha256 {desc['zip']['sha256']}")
    for m in desc["members"]:
        print(f"  {m['name']}: {m['bytes']} bytes sha256 {m['sha256']}")
    print(f"wrote {MANIFEST.relative_to(ROOT).as_posix()}")
    return manifest


def check() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    desc = describe_zip(RAW_ZIP)
    problems = []
    if desc["zip"] != manifest["zip"]:
        problems.append(f"zip differs: {desc['zip']} vs manifest {manifest['zip']}")
    if desc["members"] != manifest["members"]:
        problems.append("CSV members differ from the manifest")
    for p in problems:
        print("MISMATCH:", p)
    if not problems:
        print(f"OK: zip and {len(desc['members'])} CSV files match data/manifest.json")
    return 1 if problems else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify the saved zip against the manifest")
    args = ap.parse_args(argv)
    if args.check:
        return check()
    write_manifest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
