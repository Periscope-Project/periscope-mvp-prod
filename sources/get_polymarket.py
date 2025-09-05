#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Polymarket exporter → ONE raw JSONL + ONE parsed JSONL (append-as-you-go per tag)

What you get per market:
- snapshots: best bid/ask (via /prices BUY/SELL), midpoint, spread (+ optional top-of-book)
- history:   price timeseries per outcome token (default last 90 days)
- trades:    BUY/SELL aggregation per outcome (counts, token/value, ratios)
- parsed:    trimmed, human-usable snapshot per outcome (no huge blobs)

Also:
- tags = union of catalog (/tags) + live-event tags (/events -> /events/{id}/tags)
- de-duplicates markets across tags (write once per market)
"""

import os, re, json, time, random, requests
from datetime import datetime, timedelta
from collections import defaultdict, deque
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# -------------------- CONFIG --------------------

BASE_GAMMA = "https://gamma-api.polymarket.com"
BASE_CLOB  = "https://clob.polymarket.com"
BASE_DATA  = "https://data-api.polymarket.com"

HISTORY_DAYS     = 90       # recent history window
PRICES_INTERVAL  = "1d"     # set to "1h"/"1d"/"max" to use interval mode; None → startTs/endTs
INCLUDE_BOOKS    = False    # True → fetch top-of-book (heavier)
TRADES_PAGES     = 3        # pages of 1000 trades per market
TRADE_SAMPLE_MAX = 0        # set >0 to keep some raw trade rows in raw output

# Only process these tag labels (case-insensitive). Empty set → ALL tags.
TAG_LABEL_ALLOWLIST = set()

# De-duplication: write each market once even if it belongs to many tags
DEDUPE_MARKETS = True

# Rate-limit buckets (requests per 10s)
RL_LIMITS = {
    # CLOB
    "CLOB:prices_history": (100, 10.0),
    "CLOB:prices":         (100, 10.0),  # /price (GET) and /prices (POST)
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

_buckets = defaultdict(deque)

def _acquire_bucket(key: str):
    limit, window = RL_LIMITS.get(key, (40, 10.0))
    q = _buckets[key]
    now = time.monotonic()
    while q and (now - q[0]) > window:
        q.popleft()
    if len(q) >= limit:
        sleep_for = window - (now - q[0]) + 0.01
        if sleep_for > 0:
            time.sleep(sleep_for)
        return _acquire_bucket(key)
    q.append(now)

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
    s.headers.update({"User-Agent": "polymarket-export/1.1 (+contact@example.com)"})
    return s

SESSION = _build_session()

def _request(method: str, url: str, *, params=None, json_body=None, bucket=None, max_tries=8, timeout=30):
    tries = 0
    while True:
        if bucket:
            _acquire_bucket(bucket)

        if method == "GET":
            resp = SESSION.get(url, params=params, timeout=timeout)
        else:
            resp = SESSION.post(url, params=params, json=json_body, timeout=timeout)

        if resp.status_code < 400:
            return resp

        if resp.status_code == 429 and tries < max_tries:
            ra = _parse_retry_after(resp.headers.get("Retry-After", "")) or ((2 ** tries) * 0.6 + random.uniform(0.05, 0.35))
            time.sleep(ra); tries += 1; continue

        if 500 <= resp.status_code < 600 and tries < max_tries:
            time.sleep((2 ** tries) * 0.5 + random.uniform(0.05, 0.35)); tries += 1; continue

        # helpful error print
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:300]
        print(f"[HTTP {resp.status_code}] {method} {url} params={params} body_size={len(json_body) if isinstance(json_body, list) else 'n/a'} detail={detail}")
        resp.raise_for_status()

def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

# -------------------- Helpers: tokens, status, mapping --------------------

TOKEN_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")

def clean_token_ids(tokens):
    """Filter invalid/empty token ids and dedupe."""
    out = []
    for t in tokens or []:
        if isinstance(t, str) and TOKEN_RE.match(t):
            out.append(t)
    return sorted(set(out))

def is_live_market(m):
    """Basic 'live' flag (Gamma markets endpoint already requests active=True)."""
    if m.get("active") is False: return False
    if m.get("archived") is True: return False
    if m.get("closed") is True: return False
    return True



def get_clob_token_ids_from_market(market, digits_only=True):
    """Return a cleaned list of CLOB token ids from a Gamma market object."""
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
    """Aligned outcome names ↔ token ids."""
    names  = _parse_jsonish(market.get("outcomes")) or []
    if isinstance(names, str):
        names = [names]
    tokens = get_clob_token_ids_from_market(market)  # <-- CLOB token ids here
    L = min(len(names), len(tokens))
    return names[:L], tokens[:L]

# -------------------- GAMMA: tags/markets --------------------

def fetch_catalog_tags(limit=1000, include_template=True, order="label", ascending=True):
    tags_by_id, offset = {}, 0
    while True:
        params = {
            "limit": limit, "offset": offset, "order": order,
            "ascending": str(ascending).lower(),
            "include_template": str(include_template).lower(),
        }
        r = _request("GET", f"{BASE_GAMMA}/tags", params=params, bucket="GAMMA:tags")
        batch = r.json() or []
        for t in batch:
            if t.get("id") is not None:
                tags_by_id[t["id"]] = t
        if len(batch) < limit: break
        offset += limit
    tags = list(tags_by_id.values())
    if TAG_LABEL_ALLOWLIST:
        lower = {x.lower() for x in TAG_LABEL_ALLOWLIST}
        tags = [t for t in tags if (t.get("label") or "").lower() in lower]
    return tags

def fetch_live_event_tags(limit=1000):
    events, off = [], 0
    while True:
        params = {"active":"true","archived":"false","closed":"false",
                  "order":"volume","ascending":"false","limit":limit,"offset":off}
        r = _request("GET", f"{BASE_GAMMA}/events", params=params, bucket="GAMMA:events")
        batch = r.json() or []
        events.extend(batch)
        if len(batch) < limit: break
        off += limit

    by_key = {}
    for e in events:
        eid = e.get("id")
        if not eid: continue
        r = _request("GET", f"{BASE_GAMMA}/events/{eid}/tags", bucket="GAMMA:events")
        if r.status_code == 404:
            continue
        for t in (r.json() or []):
            key = t.get("id") if t.get("id") is not None else f"slug:{t.get('slug')}"
            by_key[key] = t
    return list(by_key.values())

def fetch_all_tags_union():
    # catalog = fetch_catalog_tags()
    catalog = []
    live    = fetch_live_event_tags()
    merged = {}
    for t in catalog + live:
        key = t.get("id") if t.get("id") is not None else f"slug:{t.get('slug')}"
        merged[key] = t
    # pretty order
    return sorted(merged.values(), key=lambda t: ((t.get("label") or "").lower(), (t.get("slug") or "").lower()))

def get_markets_for_tag(tag_id, limit=1000):
    markets, off = [], 0
    while True:
        params = {"tag_id": tag_id, "active": True, "limit": limit, "offset": off}
        r = _request("GET", f"{BASE_GAMMA}/markets", params=params, bucket="GAMMA:markets")
        batch = r.json() or []
        markets.extend(batch)
        if len(batch) < limit: break
        off += limit
    return markets

# -------------------- CLOB: pricing, history, books --------------------

import json

def _parse_jsonish(val):
    if isinstance(val, str):
        s = val.strip()
        try:
            # Try to parse as JSON first (handles '["Yes","No"]', '["1","2"]', etc.)
            return json.loads(s)
        except json.JSONDecodeError:
            # Fallback: treat "Yes,No" as CSV-ish
            if "," in s:
                return [x.strip() for x in s.split(",") if x.strip()]
            # If it's just a single token like "Yes", you can choose list-or-string:
            return [s] if s else []
    return val


def get_price_history_for_token(token_id, interval=None, start_ts=None, end_ts=None, fidelity=600): #fidelity 10 hours
    params = {"market": token_id,"fidelity":str(fidelity)}
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

def get_trades_for_condition_paged(condition_id, page_limit=1000, max_pages=10, start_ts=None):
    trades, offset = [], 0
    if not condition_id:
        return trades
    for _ in range(max_pages):
        r = _request("GET", f"{BASE_DATA}/trades",
                     params={"market": condition_id, "limit": page_limit, "offset": offset, "takerOnly": "true"},
                     bucket="DATA:trades")
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
        size = float(t.get("size", 0) or 0)
        price = float(t.get("price", 0) or 0)
        val = size * price
        node = agg.setdefault(oi, {"BUY":{"count":0,"tokens":0.0,"value":0.0},
                                   "SELL":{"count":0,"tokens":0.0,"value":0.0}})
        node[side]["count"]  += 1
        node[side]["tokens"] += size
        node[side]["value"]  += val
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

# -------------------- Writers: raw + parsed --------------------

def build_outcomes_parsed(market, token_snaps, histories, trades_summary):
    names, tokens = outcomes_with_tokens(market)
    out = []
    for i, name in enumerate(names):
        tok = tokens[i] if i < len(tokens) else None
        snap = token_snaps.get(tok, {}) if tok else {}
        hist = histories.get(tok, []) if tok else []
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

def write_raw_record(fh_raw, tag, market, token_snaps, histories, trades_summary, trades_sample):
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

def write_parsed_record(fh_parsed, tag, market, token_snaps, histories, trades_summary):
    #FIXME: parsing somehow split outcome into like... characters
    rec = {
        "collected_at": datetime.utcnow().isoformat(timespec="seconds")+"Z",
        "tag": {"label": tag.get("label"), "slug": tag.get("slug")},
        "market": {
            "id": market.get("id"),
            "slug": market.get("slug"),
            "question": market.get("question"),
            "is_live": is_live_market(market),
        },
        "outcomes": build_outcomes_parsed(market, token_snaps, histories, trades_summary),
    }
    fh_parsed.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh_parsed.flush()

# -------------------- Main streaming per tag --------------------

SEEN_MARKETS = set()  # conditionId preferred, fallback to market id

def process_tag(tag, fh_raw, fh_parsed=None):
    tag_id = tag["id"]
    markets = get_markets_for_tag(tag_id)

    # time window
    end_ts = int(datetime.utcnow().timestamp())
    start_ts = int((datetime.utcnow() - timedelta(days=HISTORY_DAYS)).timestamp())

    # collect unique token ids for batch snapshots
    token_ids_all = []
    per_market_tokens = []
    for m in markets:
        _, toks = outcomes_with_tokens(m)
        per_market_tokens.append(toks)
        token_ids_all.extend(toks)
    token_ids_all = sorted(set(token_ids_all))

    # batch snapshots
    prices_buy  = get_multiple_market_prices(token_ids_all, side="BUY")
    prices_sell = get_multiple_market_prices(token_ids_all, side="SELL")
    midpoints   = get_multiple_midpoints(token_ids_all)
    spreads     = get_bid_ask_spreads(token_ids_all)
    books       = get_books(token_ids_all) if INCLUDE_BOOKS else {}

    for m, token_ids in zip(markets, per_market_tokens):
        key = m.get("conditionId") or m.get("id")
        if DEDUPE_MARKETS and key in SEEN_MARKETS:
            continue
        SEEN_MARKETS.add(key)

        # histories
        histories = {}
        for tok in token_ids:
            try:
                hist = get_price_history_for_token(tok,
                                                   interval=PRICES_INTERVAL,
                                                   start_ts=None if PRICES_INTERVAL else start_ts,
                                                   end_ts=None if PRICES_INTERVAL else end_ts)
                histories[tok] = hist
            except requests.HTTPError as e:
                histories[tok] = {"error": str(e)}

        # trades
        trades = get_trades_for_condition_paged(m.get("conditionId"), page_limit=1000, max_pages=TRADES_PAGES, start_ts=start_ts)
        tsum = summarize_trades_by_outcome(trades)

        # token snapshots bundle
        token_snaps = {}
        for tok in token_ids:
            book = books.get(tok) or {}
            best_bid = (book.get("bids") or [{}])[0] if book else None
            best_ask = (book.get("asks") or [{}])[0] if book else None
            token_snaps[tok] = {
                "price_buy_side":  prices_buy.get(tok),   # best ask
                "price_sell_side": prices_sell.get(tok),  # best bid
                "midpoint":        midpoints.get(tok),
                "spread":          spreads.get(tok),
                "best_bid":        best_bid,
                "best_ask":        best_ask,
            }

        # write both outputs
        write_raw_record(fh_raw, tag, m, token_snaps, histories, tsum, trades)
        
        if fh_parsed:
            write_parsed_record(fh_parsed, tag, m, token_snaps, histories, tsum)

# -------------------- MAIN --------------------

def main():
    tags = fetch_all_tags_union()
    print(f"Total unique tags (catalog ∪ live): {len(tags)}")

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    raw_path    = f"polymarket_all_tags_{stamp}_live.jsonl"
    # parsed_path = f"polymarket_all_tags_{stamp}.parsed.jsonl"

    with open(raw_path, "a", encoding="utf-8") as fh_raw:
        for tag in tqdm(tags, desc="Tags", unit="tag"):
            try:
                process_tag(tag, fh_raw)   # writes as it goes (after every tag)
            except Exception as e:
                print(f"!! Failed tag {tag.get('label')} ({tag.get('id') or tag.get('slug')}): {e}")

    print("✅ Wrote:", raw_path)
    # print("✅ Wrote:", parsed_path)

if __name__ == "__main__":
    main()
