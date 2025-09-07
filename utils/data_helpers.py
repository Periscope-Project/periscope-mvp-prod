# utils/get_data.py

import pandas as pd
import pathlib
from pathlib import Path
from datetime import datetime, timedelta
import json
from json import JSONDecodeError
import os
from datetime import datetime

#=======================================================
# Load Reddit data from JSONL files in a specified directory within a date range
#=======================================================
import re
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

DATE_PAT_ISO = re.compile(r"(\d{4}-\d{2}-\d{2})")      # e.g., 2025-09-07
DATE_PAT_DMY = re.compile(r"_(\d{2})_(\d{2})_(\d{2})") # e.g., _07_09_25

def _extract_dt_from_name(name: str):
    """Return a datetime.date parsed from filename or None."""
    # Try ISO first
    m = DATE_PAT_ISO.search(name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass
    # Then DD_MM_YY
    m = DATE_PAT_DMY.search(name)
    if m:
        d, mth, y2 = map(int, m.groups())
        try:
            y4 = 2000 + y2
            return datetime(y4, mth, d).date()
        except ValueError:
            pass
    return None

def load_reddit_range(root="public/files/source_data/reddit",
                      _glob="reddit_daily_*.ndjson",
                      days=7,
                      verbose=True):
    now = datetime.now()
    cutoff = now - timedelta(days=days)
    frames = []
    root_path = Path(root)

    matched = list(root_path.glob(_glob))
    if verbose:
        print(f"Found {len(matched)} files for pattern '{_glob}' in '{root}'")

    for p in matched:
        dt_date = _extract_dt_from_name(p.name)
        if dt_date is None:
            if verbose:
                print(f"Skipping (no date parsed): {p.name}")
            continue

        dt = datetime(dt_date.year, dt_date.month, dt_date.day)
        if cutoff <= dt <= now:
            try:
                df = pd.read_json(p, lines=True)
                frames.append(df.assign(_file=p.name, _file_date=dt_date.isoformat()))
                if verbose:
                    print(f"Loaded {len(df)} rows from {p.name}")
            except Exception as e:
                if verbose:
                    print(f"Error reading {p.name}: {e}")
        else:
            if verbose:
                print(f"Skipping (out of range {cutoff.date()}–{now.date()}): {p.name}")

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



def load_json_or_jsonl(path: str):
    """Return a Python list of dicts from .json / .jsonl / .ndjson."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)  # works for normal .json (array or dict)
    except JSONDecodeError:
        # Likely NDJSON/JSONL → parse lines
        try:
            df = pd.read_json(path, lines=True)
            return df.to_dict(orient="records")
        except Exception:
            # Manual fallback (very robust)
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
            return records


if __name__ == "__main__":
    # Example usage
    import os
    import glob
    os.chdir("C:/Users/Avika/OneDrive - Hogarth Worldwide/Documents/Work/Periscope/periscope-mvp-prod")
    
    files = glob.glob("public/files/source_data/reddit/reddit_daily_all*.ndjson")
    print(files)
    
    df = load_reddit_range(days=7)
    print(df.head())
    print(f"Loaded {len(df)} posts from recent files.")