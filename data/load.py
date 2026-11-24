"""Load the pinned comments with their video and split.

Text is returned exactly as served (HTML entities, tags and stray U+FEFF included);
any cleaning is a separate, documented step. COMMENT_ID is not unique in the source
(three rows appear twice), so each row gets a `row_id` of the form "<video>:<row>",
where <row> is its 0-based position in its CSV as served.
"""
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

from data.fetch import RAW_ZIP, ROOT, is_data_member

SPLITS = ROOT / "data" / "splits.json"
COLUMNS = ["COMMENT_ID", "AUTHOR", "DATE", "CONTENT", "CLASS"]


def video_of(member_name: str) -> str:
    """'Youtube01-Psy.csv' -> 'Psy'."""
    base = member_name.rsplit("/", 1)[-1]
    return base.split("-", 1)[1].rsplit(".", 1)[0]


def load_splits() -> dict:
    return json.loads(SPLITS.read_text(encoding="utf-8"))["splits"]


def load_comments(zip_path: Path = RAW_ZIP) -> pd.DataFrame:
    video_to_split = {v: s for s, vids in load_splits().items() for v in vids}
    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(n for n in zf.namelist() if is_data_member(n)):
            raw = zf.read(name)
            df = pd.read_csv(io.BytesIO(raw), encoding="utf-8", dtype=str, keep_default_na=False)
            if list(df.columns) != COLUMNS:
                raise ValueError(f"{name}: unexpected columns {list(df.columns)}")
            video = video_of(name)
            df.insert(0, "row_id", [f"{video}:{i:03d}" for i in range(len(df))])
            df["video"] = video
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"COMMENT_ID": "comment_id", "AUTHOR": "author", "DATE": "date",
                              "CONTENT": "content", "CLASS": "label"})
    out["label"] = out["label"].astype(int)
    out["split"] = out["video"].map(video_to_split)
    if out["split"].isna().any():
        missing = sorted(out.loc[out["split"].isna(), "video"].unique())
        raise ValueError(f"videos without a split: {missing}")
    return out[["row_id", "video", "split", "comment_id", "author", "date", "content", "label"]]
