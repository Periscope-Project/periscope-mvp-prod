# load_raw.py
import os, re, json, glob, hashlib, datetime
import mysql.connector as mysql
from dotenv import load_dotenv
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
os.chdir(REPO_ROOT)

ROOT = "public/files/source_data"  # adjust if needed
REDDIT_DIR = os.path.join(REPO_ROOT, ROOT, "reddit")
POLY_DIR = os.path.join(REPO_ROOT, ROOT, "polymarket")

print(REDDIT_DIR)
load_dotenv()

DB = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB", "periscope"),
}

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def reddit_source_id_from_url(url: str) -> str | None:
    m = re.search(r"/comments/([a-z0-9]+)/", url or "", re.I)
    return m.group(1) if m else None

def upsert(conn, source, source_id, created_at, url, payload_json, content_hash):
    sql = """
    INSERT INTO raw_post (source, source_id, created_at, url, payload_json, content_hash, fetched_at)
    VALUES (%s,%s,%s,%s,%s,%s,UTC_TIMESTAMP())
    ON DUPLICATE KEY UPDATE
      created_at=VALUES(created_at),
      url=VALUES(url),
      payload_json=VALUES(payload_json),
      content_hash=VALUES(content_hash),
      fetched_at=VALUES(fetched_at);
    """
    with conn.cursor() as cur:
        cur.execute(sql, (source, source_id, created_at, url, payload_json, content_hash))
    conn.commit()

import datetime as dt

SNAPSHOT_BLOCKLIST = ("categories_snapshot", "popular_snapshot", "cache", "tags_cache")

def to_mysql_dt(s: str | None) -> str | None:
    """Normalize '2025-09-03 03:35:53 UTC' or '2025-09-09T09:46:58Z' to 'YYYY-MM-DD HH:MM:SS'."""
    if not s:
        return None
    s = s.strip()
    s = s.replace(" UTC", "").replace("Z", "").replace("T", " ")
    # keep only 'YYYY-MM-DD HH:MM:SS'
    s = s[:19]
    try:
        return dt.datetime.fromisoformat(s).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None  # fall back to NULL if weird

def is_snapshot_file(path: str) -> bool:
    name = os.path.basename(path)
    return any(tag in name for tag in SNAPSHOT_BLOCKLIST)

def read_json_records(path: str):
    """Robust loader: handles JSON arrays, proper NDJSON, and messy lines."""
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()

    text = data.lstrip()
    # Case 1: JSON array file
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            # Ensure all items are dictionaries
            return [item for item in parsed if isinstance(item, dict)]
        except Exception as e:
            print("Array parse fail:", path, e)
            return []

    # Case 2: NDJSON / mixed formatting
    rows, bad = [], 0
    for i, line in enumerate(data.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        # tolerate trailing commas
        if line.endswith(","):
            line = line[:-1]
        try:
            parsed = json.loads(line)
            # Only add if it's a dictionary (not string or other types)
            if isinstance(parsed, dict):
                rows.append(parsed)
            else:
                bad += 1
                print(f"Skipping non-dict data on line {i}: {type(parsed).__name__}")
        except json.JSONDecodeError as e:
            bad += 1
            print(f"JSON parse error on line {i}: {str(e)[:100]}")
            continue
    if bad:
        print(f"Note: skipped {bad} bad line(s) in {path}")
    return rows

def load_reddit(conn):
    # accept *.jsonl / *.ndjson / *.json (line-delimited or array)
    paths = sorted(glob.glob(os.path.join(REDDIT_DIR, "*.*json*")))
    
    # Filter out snapshot files first
    filtered_paths = []
    for path in paths:
        if any(blocked in path for blocked in SNAPSHOT_BLOCKLIST):
            print(f"Skipping snapshot file: {os.path.basename(path)}")
            continue
        filtered_paths.append(path)
    
    total_processed = 0
    total_failed = 0
    
    # Progress bar for files
    for path in tqdm(filtered_paths, desc="Processing Reddit files", unit="file"):
        print(f"\nProcessing: {os.path.basename(path)}")
        
        # Use the robust JSON reader
        rows = read_json_records(path)
        
        if not rows:
            print(f"No valid JSON records found in {path}")
            continue
            
        file_processed = 0
        file_failed = 0
        
        # Progress bar for rows within each file
        for row in tqdm(rows, desc=f"Processing rows", unit="row", leave=False):
            try:
                # Additional safety check - ensure row is a dict
                if not isinstance(row, dict):
                    file_failed += 1
                    total_failed += 1
                    continue
                    
                payload = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                h = sha256(payload)
                url = row.get("url")
                sid = reddit_source_id_from_url(url) or sha256(url or row.get("title", ""))
                
                # FIX: Use the to_mysql_dt function to normalize datetime
                created_iso = to_mysql_dt(row.get("created_iso"))
                
                upsert(conn, "reddit", sid, created_iso, url, payload, h)
                file_processed += 1
                total_processed += 1
                
            except Exception as e:
                file_failed += 1
                total_failed += 1
                if file_failed <= 5:  # Only show first 5 errors per file
                    print(f"\nError: {str(e)[:100]}")
        
        print(f"File complete: {file_processed} processed, {file_failed} failed")
    
    print(f"\nReddit loading complete: {total_processed} total processed, {total_failed} total failed")

def load_polymarket(conn):
    paths = sorted(glob.glob(os.path.join(POLY_DIR, "*.json*")))
    
    # Filter out snapshot and cache files
    filtered_paths = []
    for path in paths:
        if any(blocked in path for blocked in SNAPSHOT_BLOCKLIST):
            print(f"Skipping cache/snapshot file: {os.path.basename(path)}")
            continue
        filtered_paths.append(path)
    
    total_processed = 0
    total_failed = 0
    
    # Progress bar for files
    for path in tqdm(filtered_paths, desc="Processing Polymarket files", unit="file"):
        print(f"\nProcessing: {os.path.basename(path)}")
        
        # Use the robust JSON reader
        rows = read_json_records(path)
        
        if not rows:
            print(f"No valid JSON records found in {path}")
            continue
            
        file_processed = 0
        file_failed = 0
        
        # Progress bar for rows within each file
        for row in tqdm(rows, desc=f"Processing rows", unit="row", leave=False):
            try:
                if row.get("record_type") != "market_snapshot":
                    continue
                    
                market = row.get("market", {})
                sid = str(market.get("id"))
                
                # FIX: Use the to_mysql_dt function for polymarket too
                created_at = to_mysql_dt(row.get("collected_at"))
                
                payload = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                h = sha256(payload)
                upsert(conn, "polymarket", sid, created_at, None, payload, h)
                file_processed += 1
                total_processed += 1
                
            except Exception as e:
                file_failed += 1
                total_failed += 1
                if file_failed <= 5:  # Only show first 5 errors per file
                    print(f"\nError: {str(e)[:100]}")
        
        print(f"File complete: {file_processed} processed, {file_failed} failed")
    
    print(f"\nPolymarket loading complete: {total_processed} total processed, {total_failed} total failed")

def main():
    try:
        conn = mysql.connect(**DB)
        print("Database connected successfully")
        
        load_reddit(conn)
        load_polymarket(conn)
        
        conn.close()
        print("Done loading raw files.")
        
    except mysql.Error as e:
        print(f"Database connection error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()