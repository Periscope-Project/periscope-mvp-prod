#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Polymarket LIVE exporter → ONE raw JSONL (+ optional parsed JSONL)

Only pulls **live** markets:
  • active=True, archived=False, closed=False at the API
  • local guard with is_live_market()

Speed-ups:
  • Concurrent fetch of live event tags (/events/{id}/tags) with short TTL cache
  • Prefetch ALL token midpoints/prices/spreads per tag (one batch per endpoint)
  • Optional: skip histories and/or trades entirely (default keeps both ON)
  • Concurrent histories (per token) and trades (per market) when enabled
  • Per-run caches (token -> history, conditionId -> trades_summary)
"""

import os, re, json, time, random, requests, argparse
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import List, Dict, Any, Tuple, Optional, Union
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# -------------------- CONFIG (tweak via CLI) --------------------

BASE_GAMMA = "https://gamma-api.polymarket.com"
BASE_CLOB  = "https://clob.polymarket.com"
BASE_DATA  = "https://data-api.polymarket.com"

# Defaults keep everything ON; use --skip-* flags to disable at runtime
HISTORY_DAYS     = 60         # smaller window for speed (change via --history-days)
PRICES_INTERVAL  = "1d"       # "1h"/"1d"/"max" or None → startTs/endTs window
INCLUDE_BOOKS    = False      # books are expensive; keep off unless needed
TRADES_PAGES     = 1          # pages of 1000 trades per market (keep tiny for speed)
TRADE_SAMPLE_MAX = 0

# Only process these tag labels (case-insensitive). Empty set → ALL live tags.
TAG_LABEL_ALLOWLIST: set[str] = set()

# De-dup markets across tags (write once)
DEDUPE_MARKETS = True

# Concurrency knobs (safe vs. limits)
MAX_WORKERS_HISTORY = 24
MAX_WORKERS_TRADES  = 16
MAX_WORKERS_EVENT_TAGS = 24

HTTP_TIMEOUT_SEC    = 30

# Tag cache (short TTL to reflect live movement)
TAGS_CACHE_PATH = "polymarket_tags_cache_live.json"
TAGS_CACHE_TTL_SEC = 10 * 60  # 10 minutes
USE_TAG_CACHE = True

# Rate-limit buckets (requests per 10s)
RL_LIMITS = {
    # CLOB
    "CLOB:prices_history": (100, 10.0),
    "CLOB:prices":         (100, 10.0),
    "CLOB:midpoints":      (100, 10.0),
    "CLOB:spreads":        (100, 10.0),
    "CLOB:books":          (50,  10.0),
    # GAMMA
    "GAMMA:tags":          (45,  10.0),
    "GAMMA:markets":       (45,  10.0),
    "GAMMA:events":        (45,  10.0),
    # DATA API
    "DATA:trades":         (45,  10.0),
}

# -------------------- HTTP + BACKOFF --------------------

_buckets: Dict[str, deque] = defaultdict(deque)
_bucket_locks: Dict[str, Lock] = defaultdict(Lock)

def _acquire_bucket(key: str):
    limit, window = RL_LIMITS.get(key, (40, 10.0))
    while True:
        with _bucket_locks[key]:
            q = _buckets[key]
            now = time.monotonic()
            while q and (now - q[0]) > window:
                q.popleft()
            if len(q) < limit:
                q.append(now)
                return
            oldest = q[0]
        sleep_for = window - (time.monotonic() - oldest) + 0.01
        if sleep_for > 0:
            time.sleep(sleep_for)

def _parse_retry_after(val: str):
    try:
        return float(val)
    except Exception:
        return None

def _build_session():
    s = requests.Session()
    retries = Retry(
        total=6,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "polymarket-live-export/1.0 (+contact@example.com)"})
    return s

SESSION = _build_session()

def _request(method: str, url: str, *, params=None, json_body=None, bucket=None, max_tries=8, timeout=HTTP_TIMEOUT_SEC):
    tries = 0
    while True:
        if bucket:
            _acquire_bucket(bucket)
        try:
            if method == "GET":
                resp = SESSION.get(url, params=params, timeout=timeout)
            else:
                resp = SESSION.post(url, params=params, json=json_body, timeout=timeout)
        except requests.RequestException:
            if tries < max_tries:
                time.sleep((2 ** tries) * 0.5 + random.uniform(0.05, 0.35))
                tries += 1
                continue
            raise
        if resp.status_code < 400:
            return resp
        if resp.status_code == 429 and tries < max_tries:
            ra = _parse_retry_after(resp.headers.get("Retry-After", "")) or ((2 ** tries) * 0.6 + random.uniform(0.05, 0.35))
            time.sleep(ra); tries += 1; continue
        if 500 <= resp.status_code < 600 and tries < max_tries:
            time.sleep((2 ** tries) * 0.5 + random.uniform(0.05, 0.35)); tries += 1; continue
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:300]
        print(f"[HTTP {resp.status_code}] {method} {url} params={params} body_size={len(json_body) if isinstance(json_body, list) else 'n/a'} detail={detail}")
        resp.raise_for_status()

def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

# -------------------- Helpers --------------------

TOKEN_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")

def _parse_jsonish(val):
    if isinstance(val, str):
        s = val.strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            if "," in s:
                return [x.strip() for x in s.split(",") if x.strip()]
            return [s] if s else []
    return val

def clean_token_ids(tokens):
    out, seen = [], set()
    for t in tokens or []:
        if t is None:
            continue
        s = str(t).strip()
        if not s:
            continue
        ok_decimal = s.isdigit()
        ok_hex = s.startswith("0x") and len(s) == 66 and all(c in "0123456789abcdefABCDEF" for c in s[2:])
        if ok_decimal or ok_hex:
            if s not in seen:
                seen.add(s); out.append(s)
    return out

def get_clob_token_ids_from_market(market, digits_only=True):
    ctids = _parse_jsonish(market.get("clobTokenIds")) or []
    if isinstance(ctids, str):
        ctids = [ctids]
    seen, out = set(), []
    for t in ctids:
        if isinstance(t, str) and t and (t.isdigit() if digits_only else True):
            if t not in seen:
                seen.add(t); out.append(t)
    return out

def outcomes_with_tokens(market):
    names  = _parse_jsonish(market.get("outcomes")) or []
    if isinstance(names, str):
        names = [names]
    tokens = get_clob_token_ids_from_market(market)
    L = min(len(names), len(tokens))
    return names[:L], tokens[:L]

def is_live_market(m):
    if m.get("active") is False: return False
    if m.get("archived") is True: return False
    if m.get("closed") is True: return False
    return True

# -------------------- GAMMA: live tag + market discovery --------------------

def _cache_load_tags():
    if not USE_TAG_CACHE: return None
    try:
        if not os.path.exists(TAGS_CACHE_PATH): return None
        if (time.time() - os.path.getmtime(TAGS_CACHE_PATH)) > TAGS_CACHE_TTL_SEC: return None
        with open(TAGS_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _cache_save_tags(tags):
    if not USE_TAG_CACHE: return
    try:
        with open(TAGS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(tags, f, ensure_ascii=False)
    except Exception:
        pass

def fetch_live_events_ids(limit=1000) -> List[str]:
    ids, off = [], 0
    while True:
        params = {
            "active":"true","archived":"false","closed":"false",
            "order":"volume","ascending":"false","limit":limit,"offset":off
        }
        r = _request("GET", f"{BASE_GAMMA}/events", params=params, bucket="GAMMA:events", timeout=HTTP_TIMEOUT_SEC)
        batch = r.json() or []
        if not batch: break
        for e in batch:
            eid = e.get("id")
            if eid: ids.append(eid)
        if len(batch) < limit: break
        off += limit
    return ids

def _fetch_one_event_tags(eid: str) -> List[Dict[str, Any]]:
    try:
        r = _request("GET", f"{BASE_GAMMA}/events/{eid}/tags", bucket="GAMMA:events", timeout=HTTP_TIMEOUT_SEC)
        if r.status_code == 404:
            return []
        return r.json() or []
    except Exception:
        return []

def fetch_live_event_tags_fast() -> List[Dict[str, Any]]:
    cached = _cache_load_tags()
    if cached is not None:
        return cached
    event_ids = fetch_live_events_ids()
    merged: Dict[Union[int, str], Dict[str, Any]] = {}
    if event_ids:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_EVENT_TAGS) as pool:
            futures = {pool.submit(_fetch_one_event_tags, eid): eid for eid in event_ids}
            for ft in tqdm(as_completed(futures), total=len(futures), desc="Event tags (live)", unit="evt"):
                try:
                    tags = ft.result()
                except Exception:
                    tags = []
                for t in tags:
                    key: Union[int, str] = t.get("id") if t.get("id") is not None else f"slug:{t.get('slug')}"
                    merged[key] = t
    tags = list(merged.values())
    if TAG_LABEL_ALLOWLIST:
        lower = {x.lower() for x in TAG_LABEL_ALLOWLIST}
        tags = [t for t in tags if (t.get("label") or "").lower() in lower]
    tags = sorted(tags, key=lambda t: ((t.get("label") or "").lower(), (t.get("slug") or "").lower()))
    _cache_save_tags(tags)
    return tags

def get_live_markets_for_tag(tag_id, limit=1000):
    markets, off = [], 0
    while True:
        params = {
            "tag_id": tag_id,
            "active": True,
            "archived": False,   # enforce live
            "closed": False,     # enforce live
            "limit": limit,
            "offset": off
        }
        r = _request("GET", f"{BASE_GAMMA}/markets", params=params, bucket="GAMMA:markets")
        batch = r.json() or []
        if not batch: break
        # local guard (belt & braces)
        batch = [m for m in batch if is_live_market(m)]
        markets.extend(batch)
        if len(batch) < limit: break
        off += limit
    return markets

# -------------------- CLOB: pricing, history, books --------------------

def get_price_history_for_token(token_id, interval=None, start_ts=None, end_ts=None, fidelity=600):
    params = {"market": str(token_id), "fidelity": str(fidelity)}
    if interval:
        params["interval"] = interval
    else:
        params.update({"startTs": int(start_ts), "endTs": int(end_ts)})
    r = _request("GET", f"{BASE_CLOB}/prices-history", params=params, bucket="CLOB:prices_history")
    return r.json().get("history", [])

def get_multiple_market_prices(token_ids, side, batch_size=150):
    out = {}
    tids = clean_token_ids(token_ids)
    for chunk in _chunked(tids, batch_size):
        body = [{"token_id": tid, "side": side} for tid in chunk]
        r = _request("POST", f"{BASE_CLOB}/prices", json_body=body, bucket="CLOB:prices")
        out.update(r.json() or {})
    return out

def get_multiple_midpoints(token_ids, batch_size=150):
    out = {}
    tids = clean_token_ids(token_ids)
    for chunk in _chunked(tids, batch_size):
        body = [{"token_id": tid} for tid in chunk]
        r = _request("POST", f"{BASE_CLOB}/midpoints", json_body=body, bucket="CLOB:midpoints")
        out.update(r.json() or {})
    return out

def get_bid_ask_spreads(token_ids, batch_size=150):
    out = {}
    tids = clean_token_ids(token_ids)
    for chunk in _chunked(tids, batch_size):
        body = [{"token_id": tid} for tid in chunk]
        r = _request("POST", f"{BASE_CLOB}/spreads", json_body=body, bucket="CLOB:spreads")
        out.update(r.json() or {})
    return out

def get_books(token_ids, batch_size=100):
    out = {}
    tids = clean_token_ids(token_ids)
    for chunk in _chunked(tids, batch_size):
        body = [{"token_id": tid} for tid in chunk]
        r = _request("POST", f"{BASE_CLOB}/books", json_body=body, bucket="CLOB:books")
        out.update(r.json() or {})
    return out

# -------------------- DATA API: trades + aggregation --------------------

def get_trades_for_condition_paged(condition_id, page_limit=1000, max_pages=1, start_ts=None):
    trades, offset = [], 0
    if not condition_id:
        return trades
    for _ in range(max_pages):
        r = _request(
            "GET", f"{BASE_DATA}/trades",
            params={"market": condition_id, "limit": page_limit, "offset": offset, "takerOnly": "true"},
            bucket="DATA:trades"
        )
        batch = r.json() or []
        if not batch: break
        trades.extend(batch)
        if start_ts and any(t.get("timestamp") and t["timestamp"] < start_ts for t in batch):
            break
        if len(batch) < page_limit: break
        offset += page_limit
    return trades

def summarize_trades_by_outcome(trades):
    agg = {}
    for t in trades:
        oi = t.get("outcomeIndex"); side = t.get("side")
        if oi is None or side not in ("BUY","SELL"): continue
        size  = float(t.get("size", 0) or 0)
        price = float(t.get("price", 0) or 0)
        val   = size * price
        node = agg.setdefault(oi, {
            "BUY":{"count":0,"tokens":0.0,"value":0.0},
            "SELL":{"count":0,"tokens":0.0,"value":0.0}
        })
        node[side]["count"]  += 1
        node[side]["tokens"] += size
        node[side]["value"]   += val
    for oi, node in agg.items():
        bt, st = node["BUY"]["tokens"], node["SELL"]["tokens"]
        bv, sv = node["BUY"]["value"],  node["SELL"]["value"]
        tot_t, tot_v = bt + st, bv + sv
        node["totals"] = {"tokens": tot_t, "value": tot_v, "trades": node["BUY"]["count"] + node["SELL"]["count"]}
        node["buy_ratio_tokens"] = (bt / tot_t) if tot_t > 0 else None
        node["buy_ratio_value"]  = (bv / tot_v) if tot_v > 0 else None
        node["net_flow_tokens"]  = bt - st
        node["net_flow_value"]   = bv - sv
    return agg

# -------------------- Writers --------------------

def build_outcomes_parsed(market, token_snaps, histories_by_token, trades_summary):
    names, tokens = outcomes_with_tokens(market)
    out = []
    for i, name in enumerate(names):
        tok  = tokens[i] if i < len(tokens) else None
        snap = token_snaps.get(tok, {}) if tok else {}
        hist = histories_by_token.get(tok, []) if tok else []
        last_price = hist[-1]["p"] if hist and isinstance(hist[-1], dict) and "p" in hist[-1] else None
        out.append({
            "index": i,
            "name": name,
            "token_id": tok,
            "best_bid":  snap.get("price_sell_side"),
            "best_ask":  snap.get("price_buy_side"),
            "midpoint":  snap.get("midpoint"),
            "spread":    snap.get("spread"),
            "last_price": last_price,
            "trades":    trades_summary.get(i, {}),
        })
    return out

def write_raw_record(fh_raw, tag, market, token_snaps, histories_by_token, trades_summary, trades_sample):
    _, tokens = outcomes_with_tokens(market)
    histories = {tok: histories_by_token.get(tok, []) for tok in tokens}
    record = {
        "record_type": "market_snapshot",
        "collected_at": datetime.utcnow().isoformat(timespec="seconds")+"Z",
        "tag": {"id": tag.get("id"), "label": tag.get("label"), "slug": tag.get("slug")},
        "market": {
            "id": market.get("id"),
            "slug": market.get("slug"),
            "question": market.get("question"),
            "conditionId": market.get("conditionId"),
            "outcomes": market.get("outcomes"),
            "clobTokenIds": market.get("clobTokenIds"),
            "volumeNum": market.get("volumeNum"),
            "liquidityNum": market.get("liquidityNum"),
            "active": market.get("active"),
            "archived": market.get("archived"),
            "closed": market.get("closed"),
        },
        "token_snapshots": token_snaps,
        "prices_history": histories,
        "trades_summary": trades_summary,
    }
    if TRADE_SAMPLE_MAX > 0:
        record["trades_sample"] = trades_sample[:TRADE_SAMPLE_MAX]
    fh_raw.write(json.dumps(record, ensure_ascii=False) + "\n"); fh_raw.flush()

def write_parsed_record(fh_parsed, tag, market, token_snaps, histories_by_token, trades_summary):
    rec = {
        "collected_at": datetime.utcnow().isoformat(timespec="seconds")+"Z",
        "tag": {"label": tag.get("label"), "slug": tag.get("slug")},
        "market": {
            "id": market.get("id"),
            "slug": market.get("slug"),
            "question": market.get("question"),
            "is_live": is_live_market(market),
        },
        "outcomes": build_outcomes_parsed(market, token_snaps, histories_by_token, trades_summary),
    }
    fh_parsed.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh_parsed.flush()

# -------------------- Per-tag processor (LIVE ONLY) --------------------

SEEN_MARKETS = set()  # conditionId preferred, fallback to market id

def process_tag(tag, fh_raw, fh_parsed=None, *, skip_history=False, skip_trades=False):
    tag_id = tag["id"]
    markets = get_live_markets_for_tag(tag_id)
    if not markets:
        return

    # Keep only live markets early (defensive)
    markets = [m for m in markets if is_live_market(m)]
    if not markets:
        return

    # Time window
    end_ts = int(datetime.utcnow().timestamp())
    start_ts = int((datetime.utcnow() - timedelta(days=HISTORY_DAYS)).timestamp())

    # Collect tokens (unique per tag) + per-market mapping
    token_ids_all: List[str] = []
    per_market_tokens: List[List[str]] = []
    market_keys: List[str] = []

    for m in markets:
        _, toks = outcomes_with_tokens(m)
        per_market_tokens.append(toks)
        token_ids_all.extend(toks)
        market_keys.append(m.get("conditionId") or m.get("id"))

    token_ids_all = sorted(set(token_ids_all))

    # ------------------ Batch snapshots up front ------------------
    prices_buy  = get_multiple_market_prices(token_ids_all, side="BUY")   # best ask
    prices_sell = get_multiple_market_prices(token_ids_all, side="SELL")  # best bid
    midpoints   = get_multiple_midpoints(token_ids_all)
    spreads     = get_bid_ask_spreads(token_ids_all)
    books       = get_books(token_ids_all) if INCLUDE_BOOKS else {}

    def _snap_for_token(tok: str):
        book = books.get(tok) or {}
        best_bid = (book.get("bids") or [{}])[0] if book else None
        best_ask = (book.get("asks") or [{}])[0] if book else None
        return {
            "price_buy_side":  prices_buy.get(tok),   # best ask
            "price_sell_side": prices_sell.get(tok),  # best bid
            "midpoint":        midpoints.get(tok),
            "spread":          spreads.get(tok),
            "best_bid":        best_bid,
            "best_ask":        best_ask,
        }

    # ------------------ Prefetch ALL histories (optional) ------------------
    histories_cache: Dict[str, Any] = {}

    if not skip_history and token_ids_all:
        def _fetch_hist(tok: str):
            try:
                return tok, get_price_history_for_token(
                    tok,
                    interval=PRICES_INTERVAL,
                    start_ts=None if PRICES_INTERVAL else start_ts,
                    end_ts=None if PRICES_INTERVAL else end_ts
                )
            except requests.HTTPError as e:
                return tok, {"error": str(e)}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS_HISTORY) as pool:
            futures = {pool.submit(_fetch_hist, tok): tok for tok in token_ids_all}
            for ft in tqdm(as_completed(futures), total=len(futures), desc=f"Histories [{tag.get('label','?')}]"):
                tok, hist = ft.result()
                histories_cache[tok] = hist

    # ------------------ Prefetch trades (optional, per market) --------------
    trades_cache: Dict[str, Any] = {}
    if not skip_trades:
        def _fetch_trades_and_sum(cid: Optional[str]):
            if not cid:
                return None, {}
            try:
                t = get_trades_for_condition_paged(
                    cid,
                    page_limit=1000,
                    max_pages=TRADES_PAGES,
                    start_ts=start_ts
                )
                return cid, summarize_trades_by_outcome(t)
            except requests.HTTPError as e:
                return cid, {"error": str(e)}

        condition_ids = sorted({m.get("conditionId") for m in markets if m.get("conditionId")})
        if condition_ids:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS_TRADES) as pool:
                futures = {pool.submit(_fetch_trades_and_sum, cid): cid for cid in condition_ids}
                for ft in tqdm(as_completed(futures), total=len(futures), desc=f"Trades [{tag.get('label','?')}]"):
                    cid, tsum = ft.result()
                    if cid:
                        trades_cache[cid] = tsum

    # ------------------ Write per-market ------------------
    for m, token_ids, mkey in zip(markets, per_market_tokens, market_keys):
        if DEDUPE_MARKETS and mkey in SEEN_MARKETS:
            continue
        SEEN_MARKETS.add(mkey)

        token_snaps = {tok: _snap_for_token(tok) for tok in token_ids}
        histories_by_token = {tok: histories_cache.get(tok, []) for tok in token_ids} if not skip_history else {tok: [] for tok in token_ids}
        tsum = trades_cache.get(m.get("conditionId"), {}) if not skip_trades else {}
        trades_sample = []  # reserved for future raw sample keeping

        write_raw_record(fh_raw, tag, m, token_snaps, histories_by_token, tsum, trades_sample)
        if fh_parsed:
            write_parsed_record(fh_parsed, tag, m, token_snaps, histories_by_token, tsum)

# -------------------- CLI / MAIN --------------------

def cli():
    p = argparse.ArgumentParser(description="Polymarket LIVE exporter (fast)")
    p.add_argument("--parsed", action="store_true", help="Also write parsed JSONL alongside raw")
    p.add_argument("--include-books", action="store_true", default=False, help="Include books (top-of-book) snapshots")
    p.add_argument("--trades-pages", type=int, default=TRADES_PAGES, help="Pages of 1000 trades per market (default 1)")
    p.add_argument("--history-days", type=int, default=HISTORY_DAYS, help="History window in days")
    p.add_argument("--prices-interval", type=str, default=PRICES_INTERVAL, help='Use "1h","1d","max" or "" for startTs/endTs')
    p.add_argument("--workers-history", type=int, default=MAX_WORKERS_HISTORY, help="Concurrent workers for history fetch")
    p.add_argument("--workers-trades", type=int, default=MAX_WORKERS_TRADES, help="Concurrent workers for trades fetch")
    p.add_argument("--event-tag-workers", type=int, default=MAX_WORKERS_EVENT_TAGS, help="Concurrent workers for /events/{id}/tags")
    p.add_argument("--allowlist", type=str, default="", help="Comma-separated tag labels to include (case-insensitive)")
    p.add_argument("--tags-cache-ttl", type=int, default=TAGS_CACHE_TTL_SEC, help="Seconds to keep tags cache (default 600)")
    p.add_argument("--no-tag-cache", action="store_true", help="Disable tag cache")
    # big speed levers:
    p.add_argument("--skip-history", action="store_true", help="Do NOT fetch price histories")
    p.add_argument("--skip-trades", action="store_true", help="Do NOT fetch trades/aggregations")
    return p.parse_args()

def get_polymarket_all():
    global INCLUDE_BOOKS, TRADES_PAGES, HISTORY_DAYS, PRICES_INTERVAL
    global MAX_WORKERS_HISTORY, MAX_WORKERS_TRADES, MAX_WORKERS_EVENT_TAGS
    global TAG_LABEL_ALLOWLIST, TAGS_CACHE_TTL_SEC, USE_TAG_CACHE

    args = cli()
    INCLUDE_BOOKS       = bool(args.include_books)
    TRADES_PAGES        = int(args.trades_pages)
    HISTORY_DAYS        = int(args.history_days)
    PRICES_INTERVAL     = args.prices_interval if args.prices_interval != "" else None
    MAX_WORKERS_HISTORY = int(args.workers_history)
    MAX_WORKERS_TRADES  = int(args.workers_trades)
    MAX_WORKERS_EVENT_TAGS = int(args.event_tag_workers)
    TAGS_CACHE_TTL_SEC  = int(args.tags_cache_ttl)
    USE_TAG_CACHE       = not args.no_tag_cache

    if args.allowlist:
        TAG_LABEL_ALLOWLIST.clear()
        TAG_LABEL_ALLOWLIST.update({x.strip() for x in args.allowlist.split(",") if x.strip()})

    # 1) Live tag discovery FIRST (fast + cached)
    tags = fetch_live_event_tags_fast()
    print(f"Total unique LIVE tags: {len(tags)}")
    if TAG_LABEL_ALLOWLIST:
        print("Allowlist active:", TAG_LABEL_ALLOWLIST)

    # 2) Prepare outputs
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    raw_path    = f"polymarket_live_{stamp}.jsonl"
    parsed_path = f"polymarket_live_{stamp}.parsed.jsonl" if args.parsed else None

    # 3) Stream per tag
    with open(raw_path, "a", encoding="utf-8") as fh_raw:
        fh_parsed = None
        try:
            if parsed_path:
                fh_parsed = open(parsed_path, "a", encoding="utf-8")
            for tag in tqdm(tags, desc="Process LIVE tags", unit="tag"):
                try:
                    process_tag(tag, fh_raw, fh_parsed, skip_history=args.skip_history, skip_trades=args.skip_trades)
                except Exception as e:
                    print(f"!! Failed tag {tag.get('label')} ({tag.get('id') or tag.get('slug')}): {e}")
        finally:
            if fh_parsed and not fh_parsed.closed:
                fh_parsed.close()

    print("✅ Wrote:", raw_path)
    if parsed_path:
        print("✅ Wrote:", parsed_path)

if __name__ == "__main__":
    get_polymarket_all()()
