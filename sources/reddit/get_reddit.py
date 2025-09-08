#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TODO add more detailed prints on things being fetched for r/popular so we can see progress

"""
periscope_reddit_combined.py

One script, two pipelines, one output file:

1) Categories pipeline (your logic):
   • Discovery per category (recommended + largest); dedupe per-seed → union per category
   • Cross-category dedupe (fetch each subreddit once)
   • Fetch per-sub with mode: top_day | hot | both (per-sub cap)
   • Multi-account: shard sub list across accounts; one executor per account
   • Append NDJSON as you go
   • 429/5xx logs show which ACCOUNT label hit it
   • Snapshot saved separately

2) r/popular pipeline:
   • Modes: hot | top_day | new | rising | both | all
   • Own total cap + URL-dedupe (scoped to popular only)
   • Multi-account: shard modes across accounts; one executor per account
   • Append NDJSON as you go (same file as categories)
   • 429/5xx logs show which ACCOUNT label hit it
   • Snapshot saved separately

Env (any of these styles work):
  # Single-account fallback
  REDDIT_ID / REDDIT_CLIENT_ID
  REDDIT_SECRET / REDDIT_CLIENT_SECRET
  REDDIT_USER_AGENT

  # JSON list (preferred for many accounts)
  REDDIT_ACCOUNTS='[
    {"client_id":"...","client_secret":"...","user_agent":"periscope/acc1","label":"ACC1"},
    {"client_id":"...","client_secret":"...","user_agent":"periscope/acc2","label":"ACC2"},
    {"client_id":"...","client_secret":"...","user_agent":"periscope/acc3","label":"ACC3"},
    {"client_id":"...","client_secret":"...","user_agent":"periscope/acc4","label":"ACC4"}
  ]'

  # or named envs (any suffix)
  REDDIT_ID_GLORIA, REDDIT_SECRET_GLORIA, [REDDIT_USER_AGENT_GLORIA]
  REDDIT_ID_AVIKA,  REDDIT_SECRET_AVIKA,  [REDDIT_USER_AGENT_AVIKA]
  ...

  # or numbered envs
  REDDIT_ID_1, REDDIT_SECRET_1, [REDDIT_USER_AGENT_1]
  REDDIT_ID_2, REDDIT_SECRET_2, [REDDIT_USER_AGENT_2]
  REDDIT_ID_3, REDDIT_SECRET_3, [REDDIT_USER_AGENT_3]
  REDDIT_ID_4, REDDIT_SECRET_4, [REDDIT_USER_AGENT_4]

Install:
  pip install praw python-dotenv tqdm
"""

import os, json, time, random, math, datetime as dt, argparse, ast, threading
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
import praw
from prawcore.exceptions import TooManyRequests, RequestException, ResponseException, ServerError
from tqdm.auto import tqdm

# ───────────────────────────
# Load env
# ───────────────────────────
load_dotenv()

# =========================
# Config (shared)
# =========================
DEBUG_LOG              = True  # --debug can toggle
REQUEST_TIMEOUT_SECS   = 45
POLITE_DELAY_SEC       = 2.0
MAX_RETRY_ATTEMPTS     = 6
WORKERS                = 16
OUT_DIR                = "public/files/source_data/reddit"

# Timezone-aware UTC (py3.11+ has dt.UTC; fallback to timezone.utc)
UTC = getattr(dt, "UTC", dt.timezone.utc)

# Single NDJSON target + separate snapshots
def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def out_paths_for(date_utc: dt.date) -> Tuple[str, str, str]:
    """
    Returns:
      - ndjson_path (shared output for both pipelines)
      - categories_snapshot_json
      - popular_snapshot_json
    """
    base = os.path.join(OUT_DIR, f"reddit_daily_all_{date_utc.isoformat()}")
    ndjson = base + ".ndjson"
    cats   = base + ".categories_snapshot.json"
    pop    = base + ".popular_snapshot.json"
    return ndjson, cats, pop

# Logging
def _log(msg: str):
    if DEBUG_LOG:
        print(msg, flush=True)

# Thread-safe appends
APPEND_LOCK = threading.Lock()
def append_ndjson(path: str, rows: List[Dict[str, Any]]):
    if not rows: return
    with APPEND_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Timestamp helper (no deprecation warning)
def to_iso(ts_utc: float) -> str:
    if ts_utc is None:
        return ""
    t = dt.datetime.fromtimestamp(float(ts_utc), tz=UTC)
    return t.strftime("%Y-%m-%d %H:%M:%S UTC")

# =========================
# Auth (multi-account)
# =========================

def _load_multi_reddit_creds() -> List[Dict[str, str]]:
    """
    Returns a list of {"client_id","client_secret","user_agent","label"} from:
      1) REDDIT_ACCOUNTS (JSON/py-list)
      2) Named envs with arbitrary suffix:
         - REDDIT_ID_<SUF> or REDDIT_CLIENT_ID_<SUF>
         - REDDIT_SECRET_<SUF> or REDDIT_CLIENT_SECRET_<SUF>
         - [optional] REDDIT_USER_AGENT_<SUF>
      3) Numbered envs: ..._1, ..._2, ...
      4) Single fallback: REDDIT_ID/CLIENT_ID + REDDIT_SECRET/CLIENT_SECRET
    """
    creds: List[Dict[str, str]] = []

    # 1) JSON block
    raw = os.getenv("REDDIT_ACCOUNTS")
    if raw:
        try:
            parsed = json.loads(raw)
        except Exception:
            import ast as _ast
            parsed = _ast.literal_eval(raw)
        for idx, c in enumerate(parsed, start=1):
            if c.get("client_id") and c.get("client_secret"):
                label = c.get("label") or c.get("user_agent") or f"ACC{idx}"
                creds.append({
                    "client_id": c["client_id"],
                    "client_secret": c["client_secret"],
                    "user_agent": c.get("user_agent", f"periscope/{label.lower()}"),
                    "label": label,
                })

    # 2) Named envs (accept ID_* or CLIENT_ID_* as the key)
    if not creds:
        id_prefixes  = ("REDDIT_ID_", "REDDIT_CLIENT_ID_")
        sec_prefixes = ("REDDIT_SECRET_", "REDDIT_CLIENT_SECRET_")
        ua_prefix    = "REDDIT_USER_AGENT_"

        # collect suffixes that have *both* an id and a secret (in any of the accepted forms)
        suffixes: set[str] = set()
        for k, v in os.environ.items():
            for p in id_prefixes:
                if k.startswith(p) and v:
                    suf = k[len(p):]
                    # Must have a matching secret key in either style
                    if any(os.getenv(sp + suf) for sp in sec_prefixes):
                        suffixes.add(suf)

        for suf in sorted(suffixes):
            cid = os.getenv("REDDIT_ID_" + suf) or os.getenv("REDDIT_CLIENT_ID_" + suf)
            sec = os.getenv("REDDIT_SECRET_" + suf) or os.getenv("REDDIT_CLIENT_SECRET_" + suf)
            if not (cid and sec):
                continue
            ua  = os.getenv(ua_prefix + suf) or f"periscope/{suf.lower()}"
            creds.append({"client_id": cid, "client_secret": sec, "user_agent": ua, "label": suf})

    # 3) Numbered envs
    if not creds:
        i = 1
        while True:
            cid = os.getenv(f"REDDIT_ID_{i}") or os.getenv(f"REDDIT_CLIENT_ID_{i}")
            sec = os.getenv(f"REDDIT_SECRET_{i}") or os.getenv(f"REDDIT_CLIENT_SECRET_{i}")
            if not cid or not sec:
                break
            ua  = os.getenv(f"REDDIT_USER_AGENT_{i}") or f"periscope/acc{i}"
            creds.append({"client_id": cid, "client_secret": sec, "user_agent": ua, "label": f"ACC{i}"})
            i += 1

    # 4) Single fallback
    if not creds:
        cid = os.getenv("REDDIT_ID") or os.getenv("REDDIT_CLIENT_ID")
        sec = os.getenv("REDDIT_SECRET") or os.getenv("REDDIT_CLIENT_SECRET")
        ua  = os.getenv("REDDIT_USER_AGENT") or "periscope/combined 0.1"
        if cid and sec:
            creds.append({"client_id": cid, "client_secret": sec, "user_agent": ua, "label": "SINGLE"})

    return creds


def init_reddit(client_id=None, client_secret=None, user_agent=None) -> praw.Reddit:
    client_id     = client_id or os.getenv("REDDIT_ID") or os.getenv("REDDIT_CLIENT_ID")
    client_secret = client_secret or os.getenv("REDDIT_SECRET") or os.getenv("REDDIT_CLIENT_SECRET")
    user_agent    = user_agent or os.getenv("REDDIT_USER_AGENT") or "periscope/combined 0.1"
    if not client_id or not client_secret:
        raise RuntimeError("Set REDDIT_ID/REDDIT_SECRET (or *_CLIENT_*)")
    r = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        requestor_kwargs={"timeout": REQUEST_TIMEOUT_SECS},
    )
    r.read_only = True
    return r



def init_reddit_with(creds: Dict[str, str]) -> praw.Reddit:
    r = praw.Reddit(
        client_id=creds["client_id"],
        client_secret=creds["client_secret"],
        user_agent=creds.get("user_agent", f"periscope/{creds.get('label','acc').lower()}"),
        requestor_kwargs={"timeout": REQUEST_TIMEOUT_SECS},
    )
    r.read_only = True
    return r

# =========================
# Retry / listing (account-aware)
# =========================
def _sleep_with_retry_after(exc: BaseException, attempt: int, *, label: str, what: str):
    def _exp():
        base = min(60, 2 ** attempt)
        dur = base + random.uniform(0, 0.5 * base)
        _log(f"[{label}] backoff({what}) attempt {attempt} → sleep {dur:.1f}s")
        time.sleep(dur)

    resp = getattr(exc, "response", None)
    headers = None
    if resp and getattr(resp, "headers", None):
        headers = {k.lower(): v for k, v in resp.headers.items()}

    if headers and "retry-after" in headers:
        try:
            ra = float(headers["retry-after"])
            dur = max(1.0, ra) + random.uniform(0.25, 0.75)
            _log(f"[{label}] {type(exc).__name__} on {what} (attempt {attempt}) Retry-After={ra:.2f} → sleep {dur:.2f}s")
            time.sleep(dur); return
        except Exception:
            pass

    if headers:
        try:
            remaining = float(headers.get("x-ratelimit-remaining", "1"))
            reset     = float(headers.get("x-ratelimit-reset", "2"))
            used      = headers.get("x-ratelimit-used")
            _log(f"[{label}] rate headers on {what}: remaining={remaining}, reset={reset}, used={used}")
            if remaining <= 0.01:
                dur = max(1.0, reset) + random.uniform(0.25, 0.75)
                _log(f"[{label}] quota exhausted on {what} → sleep {dur:.2f}s")
                time.sleep(dur); return
        except Exception:
            pass

    _log(f"[{label}] {type(exc).__name__} on {what} (attempt {attempt}) → exp backoff")
    _exp()

def _retry_call(make_call, *, what: str, label: str, max_attempts: int = MAX_RETRY_ATTEMPTS):
    time.sleep(random.uniform(0.05, 0.3))  # tiny jitter
    for i in range(1, max_attempts + 1):
        try:
            return make_call()
        except TooManyRequests as e:
            _sleep_with_retry_after(e, i, label=label, what=what)
        except (RequestException, ResponseException, ServerError) as e:
            _sleep_with_retry_after(e, i, label=label, what=what)
    return make_call()

def _iter_listing(generator, *, label: str, what: str):
    it = iter(generator)
    attempt = 0
    while True:
        try:
            item = next(it); attempt = 0
            if POLITE_DELAY_SEC: time.sleep(POLITE_DELAY_SEC)
            yield item
        except StopIteration:
            return
        except TooManyRequests as e:
            attempt += 1
            _sleep_with_retry_after(e, attempt, label=label, what=what)
        except (RequestException, ResponseException, ServerError) as e:
            attempt += 1
            if attempt > MAX_RETRY_ATTEMPTS:
                _log(f"[{label}] giving up on {what} after {attempt} attempts ({type(e).__name__})")
                return
            _sleep_with_retry_after(e, attempt, label=label, what=what)
        except Exception as e:
            _log(f"[{label}] unexpected error on {what}: {type(e).__name__}: {e}")
            return

# =========================
# Record builder
# =========================
def _detect_media_and_type(post) -> Tuple[Optional[str], str]:
    try:
        if getattr(post, "is_self", False): return None, "text"
        url = getattr(post, "url", None) or None
        if getattr(post, "is_gallery", False) and getattr(post, "gallery_data", None):
            items = post.gallery_data.get("items", [])
            if items:
                media_id = items[0].get("media_id")
                if media_id and getattr(post, "media_metadata", None):
                    meta = post.media_metadata.get(media_id, {})
                    p = meta.get("p") or meta.get("s")
                    if isinstance(p, list) and p: return p[0].get("u"), "gallery"
                    if isinstance(p, dict):       return p.get("u"), "gallery"
            return url, "gallery"
        if getattr(post, "is_video", False):
            if post.media and isinstance(post.media, dict):
                rv = post.media.get("reddit_video")
                if rv and rv.get("fallback_url"): return rv["fallback_url"], "video"
            return url, "video"
        if url and (url.endswith((".jpg",".jpeg",".png",".gif",".webp")) or "i.redd.it" in url):
            return url, "image"
        if url: return url, "link"
    except Exception:
        pass
    return None, "unknown"

def _gather_top_comments(post, n: int) -> List[str]:
    if n <= 0 or not getattr(post, "num_comments", 0): return []
    try:
        post.comments.replace_more(limit=0)
        return [(c.body or "").strip() for c in post.comments[:n] if (c.body or "").strip()]
    except Exception:
        return []

def _record_from_post(post, include_comments: int) -> Dict[str, Any]:
    media_url, content_type = _detect_media_and_type(post)
    top_comments = _gather_top_comments(post, include_comments)
    return {
        "source": "reddit",
        "title": post.title,
        "text": f"{post.title}\n{post.selftext or ''}\n" + "\n".join(top_comments),
        "created_utc": post.created_utc,
        "created_iso": to_iso(post.created_utc),
        "subreddit": post.subreddit.display_name,
        "subreddit_id": post.subreddit_id,
        "subreddit_subscribers": getattr(post.subreddit, "subscribers", None),
        "author": str(post.author) if post.author else None,
        "author_fullname": getattr(post, "author_fullname", None),
        "author_premium": getattr(post, "author_premium", None),
        "score": post.score,
        "upvote_ratio": getattr(post, "upvote_ratio", None),
        "num_comments": post.num_comments,
        "flair": getattr(post, "link_flair_text", None),
        "is_ad": getattr(post, "is_created_from_ads_ui", None),
        "content_type": content_type,
        "media_url": media_url,
        "url": f"https://www.reddit.com{post.permalink}",
        "top_comments": top_comments
    }

# =========================
# Categories pipeline (unchanged behavior)
# =========================
# Config for categories
ALLOW_NSFW            = False
LIMIT_SEARCH          = 100
PICKS_PER_SEED        = 3
TOP_LIMIT_PER_SUB     = 20
INCLUDE_COMMENTS_CAT  = 0  # can override via CLI (shared flag maps to both)

CATEGORIES: Dict[str, List[str]] = {
    "Film & TV": ["movies","television","film","box office","hollywood"],
    "Music": ["music","hiphop","kpop","pop","indie","jpop","edm","music news"],
    "Sports": ["sports","soccer","nba","nfl","formula 1","tennis","cricket","football","premier league","college football", "wnba","mlb","cfb"],
    "Fashion": ["fashion","streetwear","menswear","womenswear"],
    "Beauty": ["makeup","skincare","beauty"],
    "Health": ["fitness","nutrition","wellness","health"],
    "Business & Finance": ["investing","stocks","personal finance","crypto"],
    "Science & Tech": ["technology","science"],
    "Gaming": ["gaming","playstation","nintendo"],
    "Politics": ["politics","world news","policy"],
    "AI/ML": ["AI","machine learning","openai","llama","genai","ai art"],
    "Internet Culture": ["pop culture","internet culture","fandom","influencers"],
    "Others": ["interesting","oddly specific"],
    "Humour": ["funny","memes","gag"],
}

def _top_by_size_for_seed(reddit: praw.Reddit, seed: str, picks: int, limit_search: int, allow_nsfw: bool) -> List[str]:
    hits = []
    for s in _retry_call(lambda: reddit.subreddits.search(seed, limit=limit_search), what=f"search[{seed}]", label="DISCOVERY"):
        name = s.display_name
        if name.lower() == "popular": continue
        if (not allow_nsfw) and getattr(s, "over18", False): continue
        subs = int(getattr(s, "subscribers", 0) or 0)
        hits.append((name, subs))
    hits.sort(key=lambda t: t[1], reverse=True)
    return [n for (n, _subs) in hits[:max(0, picks)]]

def _top_recommended_for_seed(reddit: praw.Reddit, seed: str, picks: int, limit_search: int, allow_nsfw: bool) -> List[str]:
    recs = _retry_call(lambda: reddit.subreddits.search(seed, limit=limit_search), what=f"search[{seed}]", label="DISCOVERY") or []
    out: List[str] = []
    for r in recs:
        nm = getattr(r, "display_name", None) or (r.get("sr_name") if isinstance(r, dict) else None)
        if not nm or nm.lower() == "popular": continue
        if (not allow_nsfw) and bool(getattr(r, "over18", False)): continue
        if nm not in out:
            out.append(nm)
            if len(out) >= picks: break
    return out

def discover_category_simple(reddit: praw.Reddit, seeds: List[str], *, picks_per_seed: int, limit_search: int, allow_nsfw: bool) -> List[str]:
    final_ordered: List[str] = []
    seen = set()
    for seed in seeds:
        rec_n  = _top_recommended_for_seed(reddit, seed, picks_per_seed, limit_search, allow_nsfw)
        _log(f"[DISCOVERY] [{seed}] rec: {rec_n}")
        top_n  = _top_by_size_for_seed(reddit,   seed, picks_per_seed, limit_search, allow_nsfw)
        _log(f"[DISCOVERY] [{seed}] top-by-size: {top_n}")
        per_seed = []
        for nm in rec_n + top_n:
            if nm not in per_seed: per_seed.append(nm)
        for nm in per_seed:
            key = nm.lower()
            if key not in seen:
                seen.add(key); final_ordered.append(nm)
    _log(f"[DISCOVERY] seeds done → {len(final_ordered)} subs")
    return final_ordered

def discover_all_categories_simple(reddit: praw.Reddit, categories: Dict[str, List[str]], *, picks_per_seed: int, limit_search: int, allow_nsfw: bool) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for cat, seeds in categories.items():
        out[cat] = discover_category_simple(reddit, seeds, picks_per_seed=picks_per_seed, limit_search=limit_search, allow_nsfw=allow_nsfw)
    return out

def fetch_top_day(reddit: praw.Reddit, sub: str, limit: int, include_comments: int, *, label: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    what = f"top(day): r/{sub}"
    gen = _retry_call(lambda: reddit.subreddit(sub).top(time_filter="day", limit=limit), what=what, label=label)
    for post in _iter_listing(gen, label=label, what=what):
        rows.append(_record_from_post(post, include_comments))
        if len(rows) >= limit: break
    return rows

def fetch_hot_now(reddit: praw.Reddit, sub: str, limit: int, include_comments: int, *, label: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    what = f"hot: r/{sub}"
    gen = _retry_call(lambda: reddit.subreddit(sub).hot(limit=limit), what=what, label=label)
    for post in _iter_listing(gen, label=label, what=what):
        rows.append(_record_from_post(post, include_comments))
        if len(rows) >= limit: break
    return rows

def _worker_fetch_sub(sub: str, limit: int, include_comments: int, mode: str, creds: Optional[Dict[str,str]]) -> Tuple[str, List[Dict[str, Any]], Optional[str], str]:
    label = (creds or {}).get("label", "SINGLE")
    try:
        r = init_reddit_with(creds) if creds else init_reddit()
        if mode == "top_day":
            rows = fetch_top_day(r, sub, limit, include_comments, label=label)
        elif mode == "hot":
            rows = fetch_hot_now(r, sub, limit, include_comments, label=label)
        elif mode == "both":
            a = fetch_top_day(r, sub, limit, include_comments, label=label)
            b = fetch_hot_now(r, sub, limit, include_comments, label=label)
            seen_urls = {x["url"] for x in a}
            rows = (a + [x for x in b if x["url"] not in seen_urls])[:limit]
        else:
            return sub, [], f"ValueError: unknown mode '{mode}'", label
        return sub, rows, None, label
    except Exception as e:
        return sub, [], f"{type(e).__name__}: {e} [account={label}]", label

def _shard_round_robin(items: List[Any], k: int) -> List[List[Any]]:
    if k <= 1: return [items]
    shards = [[] for _ in range(k)]
    for i, x in enumerate(items):
        shards[i % k].append(x)
    return shards

def run_categories_pipeline(*, out_ndjson: str, mode: str, limit_per_sub: int, include_comments: int, picks_per_seed: int, workers: int, categories: Dict[str, List[str]], allow_nsfw: bool, limit_search: int, cats_snapshot_path: str) -> int:
    # Discovery (single account is OK)
    base_reddit = init_reddit()
    print("▶ Discovery...")
    category_map = discover_all_categories_simple(base_reddit, categories, picks_per_seed=picks_per_seed, limit_search=limit_search, allow_nsfw=allow_nsfw)

    # Unique subs + mapping
    unique_subs: List[str] = []
    seen = set()
    sub_to_categories: Dict[str, List[str]] = {}
    for cat, subs in category_map.items():
        for s in subs:
            if s.lower() not in seen:
                seen.add(s.lower()); unique_subs.append(s)
            sub_to_categories.setdefault(s, [])
            if cat not in sub_to_categories[s]: sub_to_categories[s].append(cat)

    # Snapshot
    try:
        snap = {
            "ts_utc": dt.datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pipeline": "categories",
            "mode": mode,
            "limit_per_sub": limit_per_sub,
            "picks_per_seed": picks_per_seed,
            "categories": category_map,
            "unique_subs": unique_subs,
            "sub_to_categories": sub_to_categories,
        }
        with open(cats_snapshot_path, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved categories snapshot → {cats_snapshot_path}")
    except Exception as e:
        print(f"WARN: failed to save categories snapshot: {type(e).__name__}: {e}")

    # Multi-account parallel fetch (true parallel across accounts)
    multi_creds = _load_multi_reddit_creds()
    if not multi_creds:
        print("! No multi-creds found; using single account from env.")
        multi_creds = [None]

    k = len(multi_creds)
    shards = _shard_round_robin(unique_subs, k)
    total_written = 0

    total_workers = max(1, workers)
    per_acc_workers = max(1, total_workers // k)

    executors: List[ThreadPoolExecutor] = []
    futures = []
    try:
        for creds, sub_shard in zip(multi_creds, shards):
            if not sub_shard: continue
            label = (creds or {}).get("label", "SINGLE")
            _log(f"[{label}] categories executor with {per_acc_workers} workers for {len(sub_shard)} subs")
            ex = ThreadPoolExecutor(max_workers=per_acc_workers)
            executors.append(ex)
            for sub in sub_shard:
                futures.append(ex.submit(_worker_fetch_sub, sub, limit_per_sub, include_comments, mode, creds))

        with tqdm(total=len(futures), desc="Categories subs", dynamic_ncols=True) as pbar:
            for fut in as_completed(futures):
                sub, rows, err, label = fut.result()
                if err:
                    pbar.write(f"ERR r/{sub} [{label}]: {err}")
                else:
                    append_ndjson(out_ndjson, rows)
                    total_written += len(rows)
                    pbar.write(f"✓ r/{sub} [{label}]: wrote {len(rows)} rows (cat total {total_written}) [{mode}]")
                pbar.update(1)
    finally:
        for ex in executors:
            ex.shutdown(wait=True)

    print(f"✅ Categories done. Wrote {total_written} rows → {out_ndjson}")
    return total_written

# =========================
# r/popular pipeline (unchanged behavior: own cap + URL dedupe)
# =========================
def _fetch_popular_mode(reddit: praw.Reddit, mode: str, limit: int, include_comments: int, *, label: str) -> List[Dict[str, Any]]:
    sub = reddit.subreddit("popular")
    if mode == "hot":
        what = "popular.hot";   gen = _retry_call(lambda: sub.hot(limit=limit), what=what, label=label)
    elif mode == "top_day":
        what = "popular.top(day)"; gen = _retry_call(lambda: sub.top(time_filter="day", limit=limit), what=what, label=label)
    elif mode == "new":
        what = "popular.new";   gen = _retry_call(lambda: sub.new(limit=limit), what=what, label=label)
    elif mode == "rising":
        what = "popular.rising";gen = _retry_call(lambda: sub.rising(limit=limit), what=what, label=label)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    rows: List[Dict[str, Any]] = []
    for post in _iter_listing(gen, label=label, what=what):
        rows.append(_record_from_post(post, include_comments))
        if len(rows) >= limit: break
    return rows

def _worker_popular(mode: str, per_stream_limit: int, include_comments: int, creds: Optional[Dict[str,str]]) -> Tuple[str, List[Dict[str, Any]], Optional[str], str]:
    label = (creds or {}).get("label", "SINGLE")
    try:
        r = init_reddit_with(creds) if creds else init_reddit()
        rows = _fetch_popular_mode(r, mode, per_stream_limit, include_comments, label=label)
        return mode, rows, None, label
    except Exception as e:
        return mode, [], f"{type(e).__name__}: {e} [account={label}]", label

class PopularCap:
    """State for popular-only total cap + URL-dedupe."""
    def __init__(self, total_limit: int):
        self.total_limit = max(1, total_limit)
        self.total_written = 0
        self.url_seen = set()
        self.lock = threading.Lock()

def append_ndjson_capped_dedupe(path: str, rows: List[Dict[str, Any]], cap: PopularCap) -> int:
    if not rows: return 0
    wrote = 0
    with cap.lock:  # cap + dedupe confined to popular pipeline
        if cap.total_written >= cap.total_limit:
            return 0
        with open(path, "a", encoding="utf-8") as f:
            for r in rows:
                if cap.total_written >= cap.total_limit:
                    break
                url = r.get("url")
                if url and url in cap.url_seen:
                    continue
                if url:
                    cap.url_seen.add(url)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                cap.total_written += 1
                wrote += 1
    return wrote

def run_popular_pipeline(*, out_ndjson: str, modes: List[str], total_limit: int, include_comments: int, workers: int, pop_snapshot_path: str) -> int:
    # Snapshot
    try:
        snap = {
            "ts_utc": dt.datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pipeline": "popular",
            "modes": modes,
            "total_limit": total_limit,
            "include_comments": include_comments,
        }
        with open(pop_snapshot_path, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved popular snapshot → {pop_snapshot_path}")
    except Exception as e:
        print(f"WARN: failed to save popular snapshot: {type(e).__name__}: {e}")

    multi_creds = _load_multi_reddit_creds()
    if not multi_creds:
        print("! No multi-creds found; using single account from env.")
        multi_creds = [None]

    tasks = list(modes)  # one mode per task
    k = len(multi_creds)
    shards = _shard_round_robin(tasks, k)

    total_workers = max(1, workers)
    per_acc_workers = max(1, total_workers // k)

    # Per-stream limit (we'll overshoot; cap will enforce final total)
    per_stream_limit = max(1, math.ceil(total_limit / max(1, len(tasks))) * 2)

    cap = PopularCap(total_limit)
    executors: List[ThreadPoolExecutor] = []
    futures = []
    try:
        for creds, mode_shard in zip(multi_creds, shards):
            if not mode_shard: continue
            label = (creds or {}).get("label", "SINGLE")
            _log(f"[{label}] popular executor with {per_acc_workers} workers for modes: {mode_shard}")
            ex = ThreadPoolExecutor(max_workers=per_acc_workers)
            executors.append(ex)
            for m in mode_shard:
                futures.append(ex.submit(_worker_popular, m, per_stream_limit, include_comments, creds))

        with tqdm(total=len(futures), desc="r/popular modes", dynamic_ncols=True) as pbar:
            for fut in as_completed(futures):
                mode, rows, err, label = fut.result()
                if err:
                    pbar.write(f"ERR popular [{mode}] [{label}]: {err}")
                else:
                    wrote = append_ndjson_capped_dedupe(out_ndjson, rows, cap)
                    pbar.write(f"✓ popular [{mode}] [{label}]: wrote {wrote} (popular total {cap.total_written}/{cap.total_limit})")
                pbar.update(1)
                if cap.total_written >= cap.total_limit:
                    pbar.write(f"✋ Reached popular cap {cap.total_limit}. Skipping remaining writes.")
                    break
    finally:
        for ex in executors:
            ex.shutdown(wait=True)

    print(f"✅ Popular done. Wrote {cap.total_written} unique posts → {out_ndjson}")
    return cap.total_written

# =========================
# Orchestrator / CLI
# =========================
def fetch_reddit_data():
    parser = argparse.ArgumentParser()
    # Category pipeline args
    parser.add_argument("--categories-mode", choices=["top_day","hot","both"], default="both",
                        help="Mode for per-subreddit fetch in categories pipeline")
    parser.add_argument("--limit-per-sub", type=int, default=TOP_LIMIT_PER_SUB,
                        help="Max posts per subreddit per stream (categories)")
    parser.add_argument("--picks-per-seed", type=int, default=PICKS_PER_SEED,
                        help="How many 'recommended' and 'largest' subs to take per seed")
    parser.add_argument("--include-comments", type=int, default=3,
                        help="Top N comments per post (applies to both pipelines)")

    # Popular pipeline args
    parser.add_argument("--popular-mode", default="both",
                        help="Popular modes: hot|top_day|new|rising|both|all|comma-list")
    parser.add_argument("--popular-limit", type=int, default=1500,
                        help="TOTAL cap across all popular modes (unique URLs, popular-only)")

    # Execution / misc
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help="Total threads (split across accounts)")
    parser.add_argument("--order", choices=["categories_first","popular_first"], default="categories_first",
                        help="Which pipeline to run first")
    parser.add_argument("--skip-categories", action="store_true", help="Skip categories pipeline")
    parser.add_argument("--skip-popular", action="store_true", help="Skip popular pipeline")
    parser.add_argument("--debug", action="store_true", help="Verbose logs incl. 429 account labels")
    args = parser.parse_args()

    global DEBUG_LOG
    if args.debug:
        DEBUG_LOG = True

    # Expand popular mode spec
    m = args.popular_mode.strip().lower()
    if m == "both":
        popular_modes = ["top_day", "hot"]
    elif m == "all":
        popular_modes = ["top_day", "hot", "new", "rising"]
    else:
        allowed = {"top_day","hot","new","rising"}
        parts = [x.strip() for x in m.split(",") if x.strip()]
        if not parts or any(p not in allowed for p in parts):
            raise SystemExit("--popular-mode must be one of: hot|top_day|new|rising|both|all|comma-list of these")
        popular_modes = parts

    # Output paths
    ensure_outdir()
    today = dt.datetime.now(tz=UTC).date()
    out_ndjson, cats_snapshot, pop_snapshot = out_paths_for(today)

    # Run in requested order; both append to SAME NDJSON file
    total_cat = total_pop = 0
    try:
        if args.order == "categories_first":
            if not args.skip_categories:
                total_cat = run_categories_pipeline(
                    out_ndjson=out_ndjson,
                    mode=args.categories_mode,
                    limit_per_sub=max(1, args.limit_per_sub),
                    include_comments=max(0, args.include_comments),
                    picks_per_seed=max(1, args.picks_per_seed),
                    workers=max(1, args.workers),
                    categories=CATEGORIES,
                    allow_nsfw=ALLOW_NSFW,
                    limit_search=LIMIT_SEARCH,
                    cats_snapshot_path=cats_snapshot,
                )
            if not args.skip_popular:
                total_pop = run_popular_pipeline(
                    out_ndjson=out_ndjson,
                    modes=popular_modes,
                    total_limit=max(1, args.popular_limit),
                    include_comments=max(0, args.include_comments),
                    workers=max(1, args.workers),
                    pop_snapshot_path=pop_snapshot,
                )
        else:  # popular_first
            if not args.skip_popular:
                total_pop = run_popular_pipeline(
                    out_ndjson=out_ndjson,
                    modes=popular_modes,
                    total_limit=max(1, args.popular_limit),
                    include_comments=max(0, args.include_comments),
                    workers=max(1, args.workers),
                    pop_snapshot_path=pop_snapshot,
                )
            if not args.skip_categories:
                total_cat = run_categories_pipeline(
                    out_ndjson=out_ndjson,
                    mode=args.categories_mode,
                    limit_per_sub=max(1, args.limit_per_sub),
                    include_comments=max(0, args.include_comments),
                    picks_per_seed=max(1, args.picks_per_seed),
                    workers=max(1, args.workers),
                    categories=CATEGORIES,
                    allow_nsfw=ALLOW_NSFW,
                    limit_search=LIMIT_SEARCH,
                    cats_snapshot_path=cats_snapshot,
                )
    except KeyboardInterrupt:
        print("\nInterrupted — partial results already written to:", out_ndjson)

    print(f"\n✅ Combined done. Categories wrote {total_cat} rows; Popular wrote {total_pop} rows → {out_ndjson}")

if __name__ == "__main__":
    fetch_reddit_data()
