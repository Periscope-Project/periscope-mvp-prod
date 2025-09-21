# src/database_utils/aligned_data_to_sql.py
from __future__ import annotations
import os, json, hashlib, zlib, math
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional, Set

from dotenv import load_dotenv
import mysql.connector as mysql

# ---------------- env / db ----------------
load_dotenv()
DB = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB", "periscope"),
}

# ---------------- sanitizers ----------------
def _num(x):
    """Return float(x) if finite else None (so MySQL gets NULL, not 'nan')."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None

def _clean_jsonable(x):
    """Recursively replace NaN/Inf with None inside dicts/lists/primitives."""
    if isinstance(x, dict):
        return {k: _clean_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean_jsonable(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, str) and x.strip().lower() in {"nan", "inf", "+inf", "-inf"}:
        return None
    return x

def _to_json(x: Any) -> str:
    """JSON dump with NaN/Inf sanitized and allow_nan=False so it never emits NaN."""
    cleaned = _clean_jsonable(x)
    return json.dumps(cleaned, ensure_ascii=False, allow_nan=False)

# ---------------- helpers ----------------
def _mk_src_uid(obj: Dict[str, Any]) -> str:
    slug = obj.get("slug")
    if slug:
        return f"slug:{slug}"
    hkey = f"{obj.get('label','')}|{obj.get('question','')}|{obj.get('Top_n_words','')}"
    return "h:" + hashlib.sha1(hkey.encode("utf-8")).hexdigest()[:24]

def _mk_topic_id_int(item: Dict[str, Any], group_id: int) -> int:
    """Deterministic positive 31-bit int for (group_id, slug/label/question)."""
    basis = f"{group_id}|{item.get('slug','')}|{item.get('label','')}|{item.get('question','')}"
    return zlib.crc32(basis.encode("utf-8")) & 0x7FFFFFFF

def _safe_dt(v: Any) -> Optional[str]:
    if v is None: return None
    s = str(v).strip()
    if s.lower() in {"nat","nan",""}: return None
    try:
        if s.endswith("Z"):
            s = s.replace("Z","+00:00")
        return datetime.fromisoformat(s).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

def _iter_items(payload: Any):
    if isinstance(payload, list):
        for it in payload:
            if isinstance(it, dict): yield it
        return
    if isinstance(payload, dict):
        for k in ("topics","trends","items","data"):
            arr = payload.get(k)
            if isinstance(arr, list):
                for it in arr:
                    if isinstance(it, dict): yield it
                return
        yield payload

def _get_nested(item: Dict[str, Any], *path, default=None):
    cur = item
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default

def _first(*vals):
    for v in vals:
        if v not in (None, "", [], {}):
            return v
    return None

def _collect_feat_blob(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if isinstance(k, str) and k.startswith("feat_")}

# ---------------- SQL ----------------

# Insert trend groups (reference table)
UPSERT_TREND_GROUP = """
INSERT INTO trend_group (group_id, label, created_at)
VALUES (%s, %s, %s)
ON DUPLICATE KEY UPDATE
  label = VALUES(label)
"""

UPSERT_TOPIC = """
INSERT INTO topic (
  group_id, topic_id, source,
  group_similarity, label, keywords_json, size, date_range_json,
  volume_velocity, average_sentiment_score, average_odds, google_score, src_uid
) VALUES (
  %(group_id)s, %(topic_id)s, %(source)s,
  %(group_similarity)s, %(label)s, CAST(%(keywords_json)s AS JSON), %(size)s, CAST(%(date_range_json)s AS JSON),
  %(volume_velocity)s, %(average_sentiment_score)s, %(average_odds)s, %(google_score)s, %(src_uid)s
)
ON DUPLICATE KEY UPDATE
  source                     = VALUES(source),
  group_similarity           = VALUES(group_similarity),
  label                      = VALUES(label),
  keywords_json              = VALUES(keywords_json),
  size                       = VALUES(size),
  date_range_json            = VALUES(date_range_json),
  volume_velocity            = VALUES(volume_velocity),
  average_sentiment_score    = VALUES(average_sentiment_score),
  average_odds               = VALUES(average_odds),
  google_score               = VALUES(google_score),
  src_uid                    = VALUES(src_uid)
"""

INSERT_GROUP_TOPIC = """
INSERT IGNORE INTO group_topic (group_id, topic_id, data_source, topic_uid)
VALUES (%s, %s, %s, %s)
"""

UPSERT_POLY = """
INSERT INTO polymarket_doc (
  topic_id, group_id, doc_index, source, text, probability,
  question, slug, outcomes_json, created_ts, created_iso, feat_collected_ts,
  feat_json, volume, liquidity, turnover,
  tag_json, market_json, odds_json, trades_json,
  scraped_at
) VALUES (
  %(topic_id)s, %(group_id)s, %(doc_index)s, 'polymarket', %(text)s, %(probability)s,
  %(question)s, %(slug)s, CAST(%(outcomes_json)s AS JSON), %(created_ts)s, %(created_iso)s, %(feat_collected_ts)s,
  CAST(%(feat_json)s AS JSON), %(volume)s, %(liquidity)s, %(turnover)s,
  CAST(%(tag_json)s AS JSON), CAST(%(market_json)s AS JSON), CAST(%(odds_json)s AS JSON), CAST(%(trades_json)s AS JSON),
  %(scraped_at)s
)
ON DUPLICATE KEY UPDATE
  text = VALUES(text),
  probability = VALUES(probability),
  question = VALUES(question),
  slug = VALUES(slug),
  outcomes_json = VALUES(outcomes_json),
  created_ts = VALUES(created_ts),
  created_iso = VALUES(created_iso),
  feat_collected_ts = VALUES(feat_collected_ts),
  feat_json = VALUES(feat_json),
  volume = VALUES(volume),
  liquidity = VALUES(liquidity),
  turnover = VALUES(turnover),
  tag_json = VALUES(tag_json),
  market_json = VALUES(market_json),
  odds_json = VALUES(odds_json),
  trades_json = VALUES(trades_json),
  scraped_at = VALUES(scraped_at)
"""

UPSERT_REDDIT = """
INSERT INTO reddit_doc (
  topic_id, group_id, doc_index, source, title, text, probability, is_representative,
  created_utc, created_iso, created_dt,
  subreddit, subreddit_id, subreddit_subs,
  author, author_fullname, author_premium,
  score, upvote_ratio, num_comments, flair, is_ad, content_type,
  media_url, url, top_comments_json, raw_json, scraped_date
) VALUES (
  %(topic_id)s, %(group_id)s, %(doc_index)s, 'reddit', %(title)s, %(text)s, %(probability)s, %(is_representative)s,
  %(created_utc)s, %(created_iso)s, %(created_dt)s,
  %(subreddit)s, %(subreddit_id)s, %(subreddit_subs)s,
  %(author)s, %(author_fullname)s, %(author_premium)s,
  %(score)s, %(upvote_ratio)s, %(num_comments)s, %(flair)s, %(is_ad)s, %(content_type)s,
  %(media_url)s, %(url)s, CAST(%(top_comments_json)s AS JSON), CAST(%(raw_json)s AS JSON), %(scraped_date)s
)
ON DUPLICATE KEY UPDATE
  title = VALUES(title),
  text = VALUES(text),
  probability = VALUES(probability),
  is_representative = VALUES(is_representative),
  created_utc = VALUES(created_utc),
  created_iso = VALUES(created_iso),
  created_dt  = VALUES(created_dt),
  subreddit   = VALUES(subreddit),
  subreddit_id = VALUES(subreddit_id),
  subreddit_subs = VALUES(subreddit_subs),
  author = VALUES(author),
  author_fullname = VALUES(author_fullname),
  author_premium = VALUES(author_premium),
  score = VALUES(score),
  upvote_ratio = VALUES(upvote_ratio),
  num_comments = VALUES(num_comments),
  flair = VALUES(flair),
  is_ad = VALUES(is_ad),
  content_type = VALUES(content_type),
  media_url = VALUES(media_url),
  url = VALUES(url),
  top_comments_json = VALUES(top_comments_json),
  raw_json = VALUES(raw_json),
  scraped_date = VALUES(scraped_date)
"""

# ---------------- core loader ----------------
def load_aligned_into_erd(path: str, default_group_id: Optional[int] = None) -> Tuple[int,int,int,int,int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cn = mysql.connect(**DB); cn.autocommit = False
    cur = cn.cursor()

    # Step 1: Collect all unique group_ids and create trend_group entries
    group_data: Dict[int, str] = {}
    for item in _iter_items(data):
        group_id = item.get("group_id") or default_group_id
        if group_id is None:
            continue
        
        group_id = int(group_id)
        if group_id not in group_data:
            # Try to get a meaningful label for the group
            label = _first(
                item.get("group_label"),
                item.get("label"),
                _get_nested(item, "reddit_topic", "label"),
                _get_nested(item, "polymarket_topic", "label"),
                f"Group {group_id}"
            )
            group_data[group_id] = str(label)[:500]  # Truncate to fit VARCHAR(500)

    # Insert trend_groups first
    n_groups = 0
    for group_id, label in group_data.items():
        cur.execute(UPSERT_TREND_GROUP, (group_id, label, datetime.now()))
        n_groups += 1
    
    print(f"Created/updated {n_groups} trend groups")

    # Step 2: Load topics and documents
    n_topic = n_gt = n_poly = n_rd = 0

    for item in _iter_items(data):
        # group
        group_id = item.get("group_id") or default_group_id
        if group_id is None:
            continue
        
        group_id = int(group_id)

        # nested topics
        rt = item.get("reddit_topic") or {}
        pm = item.get("polymarket_topic") or {}

        # label + keywords
        top_n_words = _first(item.get("Top_n_words"), rt.get("Top_n_words"))
        keywords = _first(item.get("keywords"), rt.get("keywords"), pm.get("keywords")) or []
        if not keywords and isinstance(top_n_words, str):
            keywords = [w.strip() for w in top_n_words.split("-") if w.strip()]
        label = _first(item.get("label"), rt.get("label"), pm.get("label"))

        # date range (explicit or derived from docs)
        date_range = _first(item.get("date_range"), rt.get("date_range"), pm.get("date_range"))
        if not isinstance(date_range, dict):
            dates: List[str] = []
            docs = _first(_get_nested(rt, "documents"), rt.get("docs"),
                          item.get("Representative_Docs"), item.get("reddit_docs"), item.get("docs")) or []
            if isinstance(docs, list):
                for rd in docs:
                    iso = _first(rd.get("created_iso"), rd.get("created_dt"))
                    s = _safe_dt(iso)
                    if s: dates.append(s)
            date_range = {"min": (min(dates) if dates else None), "max": (max(dates) if dates else None)}

        # metrics
        volume_velocity = _num(_first(item.get("system_velocity"), item.get("topical_velocity"),
                                      pm.get("system_velocity"), pm.get("topical_velocity")))
        avg_sent = _num(_first(item.get("adapted_sentiment_score"), rt.get("average_sentiment_score")))
        avg_odds = _num(_first(item.get("probability"), pm.get("probability")))
        google_score = _num(_first(item.get("topical_google_score"), item.get("google_score"), rt.get("google_score")))

        # ids + source
        src_uid_val = _first(pm.get("slug"), item.get("slug"))
        src_uid = f"slug:{src_uid_val}" if src_uid_val else _mk_src_uid(item)
        topic_id = _mk_topic_id_int(
            {"slug": _first(pm.get("slug"), item.get("slug")), "label": label, "question": _first(pm.get("question"), item.get("question"))},
            group_id,
        )
        source_val = (_first(item.get("source"), "Merged") or "Merged").title()
        if source_val not in {"Merged","Reddit","Polymarket","Other"}:
            source_val = "Other"

        # Topic upsert
        topic_row = {
            "group_id": group_id,
            "topic_id": topic_id,
            "source": source_val,
            "group_similarity": _num(_first(item.get("group_similarity"), item.get("similarity"))),
            "label": label,
            "keywords_json": _to_json(keywords),
            "size": int(_num(_first(item.get("size"), rt.get("size"), pm.get("size"))) or 0) if _first(item.get("size"), rt.get("size"), pm.get("size")) is not None else None,
            "date_range_json": _to_json(date_range),
            "volume_velocity": volume_velocity,
            "average_sentiment_score": avg_sent,
            "average_odds": avg_odds,
            "google_score": google_score,
            "src_uid": src_uid,
        }
        cur.execute(UPSERT_TOPIC, topic_row)
        n_topic += 1

        # group_topic link
        cur.execute(INSERT_GROUP_TOPIC, (group_id, topic_id, "both", src_uid))
        n_gt += 1

        # polymarket_doc (now includes group_id)
        feat_blob = {**_collect_feat_blob(pm), **_collect_feat_blob(item)}  # merge nested + root feat_*
        poly_row = {
            "topic_id": topic_id,
            "group_id": group_id,  # Added group_id
            "doc_index": 0,
            "text": _first(item.get("text"), pm.get("text")),
            "probability": _num(_first(pm.get("probability"), item.get("probability"))),
            "question": _first(pm.get("question"), item.get("question")),
            "slug": _first(pm.get("slug"), item.get("slug")),
            "outcomes_json": _to_json(_first(pm.get("outcomes"), item.get("outcomes")) or []),
            "created_ts": _num(_first(pm.get("created_utc"), pm.get("created_ts"), item.get("created_utc"))),
            "created_iso": _first(pm.get("created_iso"), item.get("created_iso")),
            "feat_collected_ts": _num(_first(pm.get("feat_collected_ts"), item.get("feat_collected_ts"))),
            "feat_json": _to_json(feat_blob),
            "volume": _num(_first(pm.get("feat_mkt_volume"), item.get("feat_mkt_volume"))),
            "liquidity": _num(_first(pm.get("feat_mkt_liquidity"), item.get("feat_mkt_liquidity"))),
            "turnover": _num(_first(pm.get("feat_mkt_turnover"), item.get("feat_mkt_turnover"))),
            "tag_json": _to_json(_first(pm.get("tag"), item.get("tag")) or {}),
            "market_json": _to_json(_first(pm.get("market"), item.get("market")) or {}),
            "odds_json": _to_json(_first(pm.get("odds"), item.get("odds")) or {}),
            "trades_json": _to_json(_first(pm.get("trades_summary"), item.get("trades_summary")) or {}),
            "scraped_at": _safe_dt(_first(pm.get("scraped_date"), pm.get("scraped_dt"), item.get("scraped_date"), item.get("scraped_dt"))),
        }
        cur.execute(UPSERT_POLY, poly_row)
        n_poly += 1

        # reddit_doc(s) (now includes group_id)
        reddit_blocks = _first(_get_nested(rt, "documents"), rt.get("docs"),
                               item.get("Representative_Docs"), item.get("reddit_docs"), item.get("docs"))
        if not isinstance(reddit_blocks, list):
            reddit_blocks = []

        for idx, r in enumerate(reddit_blocks):
            rd = {
                "topic_id": topic_id,
                "group_id": group_id,  # Added group_id
                "doc_index": idx,
                "title": r.get("title"),
                "text": _first(r.get("text"), r.get("body")),
                "probability": _num(r.get("probability")),
                "is_representative": int(bool(_first(r.get("Representative_document"), r.get("representative")))) if _first(r.get("Representative_document"), r.get("representative")) is not None else None,
                "created_utc": _num(_first(r.get("created_utc"), r.get("created_ts"))),
                "created_iso": r.get("created_iso"),
                "created_dt": _safe_dt(_first(r.get("created_dt"), r.get("created_iso"))),
                "subreddit": r.get("subreddit"),
                "subreddit_id": r.get("subreddit_id"),
                "subreddit_subs": int(_num(_first(r.get("subreddit_subscribers"), r.get("subreddit_subs"))) or 0) if _first(r.get("subreddit_subscribers"), r.get("subreddit_subs")) is not None else None,
                "author": r.get("author"),
                "author_fullname": r.get("author_fullname"),
                "author_premium": int(bool(r.get("author_premium"))) if r.get("author_premium") is not None else None,
                "score": int(_num(r.get("score")) or 0) if r.get("score") is not None else None,
                "upvote_ratio": _num(r.get("upvote_ratio")),
                "num_comments": int(_num(r.get("num_comments")) or 0) if r.get("num_comments") is not None else None,
                "flair": r.get("flair"),
                "is_ad": int(bool(r.get("is_ad"))) if r.get("is_ad") is not None else None,
                "content_type": r.get("content_type"),
                "media_url": r.get("media_url"),
                "url": r.get("url"),
                "top_comments_json": _to_json(r.get("top_comments") or []),
                "raw_json": _to_json(r),
                "scraped_date": _safe_dt(r.get("scraped_date")),
            }
            cur.execute(UPSERT_REDDIT, rd)
            n_rd += 1

    cn.commit()
    cur.close(); cn.close()
    return n_groups, n_topic, n_gt, n_poly, n_rd

# ---------------- run from repo root, ignore CLI ----------------
if __name__ == "__main__":
    # compute repo root from this file: src/database_utils/aligned_data_to_sql.py -> repo root is two levels up
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
    os.chdir(REPO_ROOT)

    JSON_PATH = os.path.join(REPO_ROOT, "public", "files", "nlp_outputs", "aligned_topics_full_08_09_25.json")

    g, t, gt, p, r = load_aligned_into_erd(JSON_PATH, default_group_id=None)
    print(f"✅ Trend groups:        {g}")
    print(f"✅ Topic upserts:       {t}")
    print(f"✅ group_topic links:    {gt}")
    print(f"✅ polymarket_doc rows:  {p}")
    print(f"✅ reddit_doc rows:      {r}")