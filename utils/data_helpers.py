# utils/get_data.py

import pandas as pd
import pathlib
from pathlib import Path
from datetime import datetime, timedelta
import json
import os
from datetime import datetime

#=======================================================
# Load Reddit data from JSONL files in a specified directory within a date range
#=======================================================
def load_reddit_range(root="data/daily_reddit", days=7):
    now, cutoff = datetime.now(), datetime.now() - timedelta(days=days)
    frames = []
    for p in Path(root).glob("reddit_posts_parallel_*.jsonl"): #FIXME
        try:
            parts = p.name.strip().split("_")              # ..._DD_MM_YY.jsonl
            d, m, y = int(parts[3]), int(parts[4]), int(parts[5].split(".")[0])
            dt = datetime(2000 + y, m, d)
            if cutoff <= dt <= now:
                df = pd.read_json(p, lines=True)
                frames.append(df.assign(_file=p.name, _file_date=dt.date().isoformat()))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def clean_null_df(df,field="text"):
    
    #TODO drop duplicates based on 'title' or 'text' if needed

    """
    Clean the DataFrame by removing rows with null or empty 'text' fields.
    """
    return df[df[field].notnull() & df[field].str.strip().ne("")]

#=======================================================
# Parse Output
# ======================================================
def _extract_json_blocks(text: str):
    """Yield JSON substrings that start with { and have balanced braces."""
    brace_stack, start = [], None
    for i, ch in enumerate(text):
        if ch == "{":
            if not brace_stack:     # fresh object starts
                start = i
            brace_stack.append("{")
        elif ch == "}":
            if brace_stack:
                brace_stack.pop()
                if not brace_stack:  # object closed
                    yield text[start : i + 1]

def _parse_trends(raw_text: str):
    """Return a flat list of TrendBrief dicts from concatenated JSON blocks."""
    trends = []
    for block in _extract_json_blocks(raw_text):
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and "trends" in obj:
                trends.extend(obj["trends"])
        except json.JSONDecodeError:
            # skip malformed blocks
            continue
    return trends

def _renumber_trends(trends):
    """Overwrite trend_id with sequential ints starting at 0."""
    for new_id, brief in enumerate(trends):
        brief["trend_id"] = new_id
    return trends

def parse_chunk_json(SOURCE_FILE, OUT_FILE):
    raw = pathlib.Path(SOURCE_FILE).read_text(encoding="utf-8")
    all_trends = _renumber_trends(_parse_trends(raw))

    combined_obj = {"trends": all_trends}
    pathlib.Path(OUT_FILE).write_text(
        json.dumps(combined_obj, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"✅ Wrote {len(all_trends)} TrendBriefs → {OUT_FILE}")



if __name__ == "__main__":
    # Example usage
    df = load_reddit_range(days=7)
    print(df.head())
    print(f"Loaded {len(df)} posts from recent files.")