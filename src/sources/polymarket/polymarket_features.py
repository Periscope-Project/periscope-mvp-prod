"""
polymarket_features.py — minimal, prediction-focused features + full metadata (no IDs)

What this gives you:
- Compact numeric/boolean features for modeling.
- PLUS "all the metadata except the id bullshits":
  • Top-level: record_type, collected_at (ISO)
  • From tag: label, slug (no ids)
  • From market: slug, question, choices (normalized from outcomes), active/archived/closed,
    and any other simple market fields EXCEPT anything that looks like an ID
    (e.g., conditionId), and EXCEPT heavy fields (clobTokenIds, token_snapshots, prices_history, trades_summary).
  • Volume/liquidity are already features, so they’re *not* duplicated in metadata.

Defaults:
  main(..., include_meta=True, meta_mode="all_no_ids", flatten=True, only_features=True)
  → output is a list of dicts like:
     {
       "slug": "...",
       "question": "...",
       "choices": [...],
       "active": true,
       "archived": false,
       "closed": true,
       "collected_at": "2025-09-05T16:49:01+0000",
       "tag_label": "...",
       "tag_slug": "...",
       "market_category": "...",
       ...
       # features:
       "collected_ts": 1_757_090_...,
       "mkt_volume": ...,
       "mkt_liquidity": ...,
       "mkt_turnover": ...,
       "out0_mid": ...,
       ...
     }
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np


# =========================
# helpers
# =========================

ISO_FMT = "%Y-%m-%dT%H:%M:%S%z"  # e.g. 2025-09-05T16:49:01+0000


def _jsonish(x: Any) -> Any:
    """Decode JSON-encoded strings; pass through lists/dicts; else None."""
    if x is None:
        return None
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
    return x


def _to_float(x: Any) -> Optional[float]:
    """Coerce numbers / numeric strings / {'BUY':'0.051'} into float."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        if isinstance(x, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            v = float(s)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except Exception:
            return None
    if isinstance(x, Mapping):
        try:
            return _to_float(next(iter(x.values())))
        except Exception:
            return None
    return None


def _safe_get(d: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Nested get: _safe_get(obj, "a","b") -> obj.get("a",{}).get("b")."""
    for k in keys:
        if not isinstance(d, Mapping):
            return default
        d = d.get(k, default)
    return d


def _parse_collected_at(s: Optional[str]) -> Tuple[Optional[str], Optional[float]]:
    """Return (ISO UTC, epoch seconds). We only keep epoch for modeling."""
    if not s:
        return None, None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime(ISO_FMT), dt.timestamp()
    except Exception:
        return None, None


def _ols_slope(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """OLS slope of y on x (centered x)."""
    if x is None or y is None or len(x) < 2 or len(y) < 2:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom == 0.0:
        return None
    return float((x * y).sum() / denom)


def _history_stats(points: List[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    """Compact stats from [{'t': epoch, 'p': price}, ...]. Keep only what helps."""
    if not points:
        return {
            "hist_last": None,
            "hist_slope": None,
            "hist_realized_vol": None,
        }
    ts = [p.get("t") for p in points if p and "t" in p and "p" in p]
    ps = [p.get("p") for p in points if p and "t" in p and "p" in p]
    if not ts or not ps:
        return {"hist_last": None, "hist_slope": None, "hist_realized_vol": None}

    ts = np.asarray(ts, dtype=float)
    ps = np.asarray(ps, dtype=float)

    last = float(ps[-1])
    slope = _ols_slope(ts, ps)

    if len(ps) >= 2 and (ps > 0).all():
        rets = np.diff(ps) / ps[:-1]
        realized_vol = float(np.std(rets)) if len(rets) else 0.0
    else:
        realized_vol = None

    return {"hist_last": last, "hist_slope": slope, "hist_realized_vol": realized_vol}


def _entropy(p: Optional[float]) -> Optional[float]:
    """Binary entropy H(p) in nats; None if p not in (0,1)."""
    if p is None or not (0.0 < p < 1.0):
        return None
    return float(-(p * math.log(p) + (1 - p) * math.log(1 - p)))


@dataclass
class OutcomeQuotes:
    midpoint: Optional[float]
    spread: Optional[float]
    bb: Optional[float]
    ba: Optional[float]


def _parse_outcome_quotes(tok_snap: Optional[Mapping[str, Any]]) -> OutcomeQuotes:
    """Pull mid/spread/best bid/best ask as floats (others omitted)."""
    if not isinstance(tok_snap, Mapping) or not tok_snap:
        return OutcomeQuotes(None, None, None, None)
    mid = _to_float(tok_snap.get("midpoint"))
    spr = _to_float(tok_snap.get("spread"))
    bb = _to_float(tok_snap.get("best_bid"))
    ba = _to_float(tok_snap.get("best_ask"))
    return OutcomeQuotes(mid, spr, bb, ba)


# =========================
# metadata (no IDs)
# =========================

def extract_minimal_meta(
    snap: Mapping[str, Any],
    meta_keys: Tuple[str, ...] = ("slug", "question", "choices"),
) -> Dict[str, Any]:
    """Tiny metadata: slug, question, choices."""
    mkt = snap.get("market") or {}
    outcomes = _jsonish(mkt.get("outcomes")) or []
    meta: Dict[str, Any] = {}
    if "slug" in meta_keys:
        meta["slug"] = mkt.get("slug")
    if "question" in meta_keys:
        meta["question"] = mkt.get("question")
    if "choices" in meta_keys:
        meta["choices"] = [str(o) for o in outcomes] if isinstance(outcomes, list) else None
    return meta


def extract_full_meta_no_ids(snap: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Full metadata with IDs stripped out (case-insensitive 'id' in key),
    and heavy fields removed (clobTokenIds, token_snapshots, prices_history, trades_summary).
    Volume/Liquidity are excluded here (they're features).
    """
    out: Dict[str, Any] = {}

    # top-level
    out["record_type"] = snap.get("record_type")
    iso, _ = _parse_collected_at(snap.get("collected_at"))
    out["collected_at"] = iso

    # tag: everything except *id*
    tag = snap.get("tag") or {}
    for k, v in tag.items():
        if "id" in str(k).lower():
            continue
        out[f"tag_{k}"] = v

    # market: keep simple fields, exclude ids & heavy stuff
    mkt = snap.get("market") or {}
    exclude_keys = {"outcomes", "clobTokenIds", "token_snapshots", "prices_history", "trades_summary",
                    "volumeNum", "liquidityNum"}
    for k, v in mkt.items():
        kl = str(k).lower()
        if ("id" in kl) or (k in exclude_keys):
            continue
        # normalize choices from outcomes separately
        if k == "slug":
            out["slug"] = v
        elif k == "question":
            out["question"] = v
        elif k in ("active", "archived", "closed"):
            out[k] = bool(v)
        else:
            # keep only simple scalars to avoid dumping big nested blobs
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[f"market_{k}"] = v

    # choices
    outcomes = _jsonish(mkt.get("outcomes")) or []
    out["choices"] = [str(o) for o in outcomes] if isinstance(outcomes, list) else None

    return out


# =========================
# core: one snapshot -> features
# =========================

def extract_market_features(snap: Mapping[str, Any]) -> Dict[str, Any]:
    """Minimal, prediction-ready features from one `market_snapshot`."""
    feats: Dict[str, Any] = {}

    # time (numeric)
    _, collected_ts = _parse_collected_at(snap.get("collected_at"))
    feats["collected_ts"] = collected_ts

    # market-level numbers
    mkt = snap.get("market") or {}
    vol = _to_float(mkt.get("volumeNum"))
    liq = _to_float(mkt.get("liquidityNum"))
    feats["mkt_volume"] = vol
    feats["mkt_liquidity"] = liq
    feats["mkt_turnover"] = (vol / liq) if (vol is not None and liq and liq != 0) else None

    # outcomes + token map
    outcomes = _jsonish(mkt.get("outcomes")) or []
    clob_ids = _jsonish(mkt.get("clobTokenIds")) or []
    feats["n_outcomes"] = int(len(outcomes))
    feats["is_binary"] = bool(len(outcomes) == 2)

    tok_snaps = snap.get("token_snapshots") or {}
    prices_hist = snap.get("prices_history") or {}
    trades = snap.get("trades_summary") or {}
    idx_to_tok = {i: tok for i, tok in enumerate(clob_ids)}

    mids_sum = 0.0
    mids_count = 0

    # per-outcome compact block
    for i in range(len(outcomes)):
        tok = idx_to_tok.get(i)
        q = _parse_outcome_quotes(tok_snaps.get(tok)) if tok else OutcomeQuotes(None, None, None, None)
        ph_points = prices_hist.get(tok) if tok else None
        hist = _history_stats(ph_points) if isinstance(ph_points, list) else _history_stats([])

        if q.midpoint is not None:
            mids_sum += float(q.midpoint)
            mids_count += 1

        td = trades.get(str(i)) or {}
        buy_v = _to_float(_safe_get(td, "BUY", "value")) or 0.0
        sell_v = _to_float(_safe_get(td, "SELL", "value")) or 0.0
        tot_v = buy_v + sell_v
        buy_ratio_val = (buy_v / tot_v) if tot_v else None
        net_flow_val = buy_v - sell_v
        imb_val = (net_flow_val / tot_v) if tot_v else None
        trade_count = int(_safe_get(td, "totals", "trades") or (_safe_get(td, "BUY", "count") or 0) + (_safe_get(td, "SELL", "count") or 0))

        feats.update({
            f"out{i}_mid": q.midpoint,
            f"out{i}_spr": q.spread,
            f"out{i}_bb": q.bb,
            f"out{i}_ba": q.ba,
            f"out{i}_hist_last": hist["hist_last"],
            f"out{i}_hist_slope": hist["hist_slope"],
            f"out{i}_hist_realized_vol": hist["hist_realized_vol"],
            f"out{i}_buy_ratio_value": buy_ratio_val,
            f"out{i}_imbalance_value": imb_val,
            f"out{i}_net_flow_value": net_flow_val,
            f"out{i}_trades": trade_count,
        })

    # aggregated order flow (value)
    agg_buy_v = 0.0
    agg_sell_v = 0.0
    total_trades = 0
    for k, v in (trades.items() if isinstance(trades, Mapping) else []):
        try:
            _ = int(k)
        except Exception:
            continue
        buy_v = _to_float(_safe_get(v, "BUY", "value")) or 0.0
        sell_v = _to_float(_safe_get(v, "SELL", "value")) or 0.0
        total_trades += int(_safe_get(v, "totals", "trades") or 0)
        agg_buy_v += buy_v
        agg_sell_v += sell_v

    denom_v_all = agg_buy_v + agg_sell_v
    feats.update({
        "flow_trades_total": int(total_trades),
        "flow_buy_ratio_value": (agg_buy_v / denom_v_all) if denom_v_all else None,
        "flow_imbalance_value": ((agg_buy_v - agg_sell_v) / denom_v_all) if denom_v_all else None,
    })

    # multi/binary diagnostics
    feats["sum_midpoints_gap_to_1"] = (mids_sum - 1.0) if mids_count > 0 else None

    if feats["is_binary"]:
        try:
            yes_idx = next(i for i, o in enumerate(outcomes) if isinstance(o, str) and o.strip().lower() == "yes")
            no_idx = 1 - yes_idx
        except Exception:
            yes_idx, no_idx = 0, 1

        y_mid = feats.get(f"out{yes_idx}_mid")
        n_mid = feats.get(f"out{no_idx}_mid")

        feats["yes_mid"] = y_mid
        feats["no_mid"] = n_mid
        feats["binary_sum_gap"] = (y_mid + n_mid - 1.0) if (y_mid is not None and n_mid is not None) else None
        feats["binary_duality_gap"] = (abs((y_mid or 0.0) - (1.0 - (n_mid or 0.0)))
                                       if (y_mid is not None and n_mid is not None) else None)
        feats["binary_yes_log_odds"] = (math.log(y_mid / (1.0 - y_mid))
                                        if (y_mid is not None and 0.0 < y_mid < 1.0) else None)
        feats["binary_yes_entropy"] = _entropy(y_mid)

    return feats


# =========================
# enrichment
# =========================

def attach_features(
    record: Mapping[str, Any],
    features: Mapping[str, Any],
    *,
    key: Optional[str] = "features",
    flatten: bool = False,
    prefix: str = "feat_",
    only_features: bool = False,
    inplace: bool = False,
    include_meta: bool = True,
    meta_mode: str = "all_no_ids",  # "all_no_ids" | "minimal"
    meta_keys: Tuple[str, ...] = ("slug", "question", "choices"),
) -> Dict[str, Any]:
    """
    Attach features (and optional metadata) back onto `record`.

    - only_features=True: return only {meta?, features} (original fields dropped).
    - flatten=True: put features at the top-level (with optional prefix).
    - meta_mode: "all_no_ids" (default) or "minimal" (slug/question/choices).
    """
    # compute metadata
    if include_meta:
        meta = extract_full_meta_no_ids(record) if meta_mode == "all_no_ids" else extract_minimal_meta(record, meta_keys)
    else:
        meta = {}

    if only_features:
        base: Dict[str, Any] = {}
        if meta:
            base.update(meta)
        if flatten:
            base.update({f"{prefix}{k}": v for k, v in features.items()})
        else:
            base[key or "features"] = dict(features)
        return base

    base = record if inplace else dict(record)
    if meta:
        base.update(meta)
    if flatten:
        for k, v in features.items():
            base[f"{prefix}{k}"] = v
    else:
        base[key or "features"] = dict(features)
    return base  # type: ignore[return-value]


def enrich_many(
    records: Iterable[Mapping[str, Any]],
    *,
    key: Optional[str] = "features",
    flatten: bool = True,
    prefix: str = "",
    only_features: bool = True,
    inplace: bool = False,
    include_meta: bool = True,
    meta_mode: str = "all_no_ids",
    meta_keys: Tuple[str, ...] = ("slug", "question", "choices"),
) -> List[Dict[str, Any]]:
    """Vectorized extraction + attachment."""
    out: List[Dict[str, Any]] = []
    for rec in records:
        feats = extract_market_features(rec)
        out.append(
            attach_features(
                rec, feats,
                key=key, flatten=flatten, prefix=prefix,
                only_features=only_features, inplace=inplace,
                include_meta=include_meta, meta_mode=meta_mode, meta_keys=meta_keys
            )
        )
    return out


# =========================
# tiny IO helpers (json/jsonl)
# =========================

def read_jsonish(path: str) -> List[Mapping[str, Any]]:
    """Read .json/.jsonl/.ndjson — tolerant of containers."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    if "\n" in txt and txt.lstrip().startswith("{"):
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        try:
            return [json.loads(ln) for ln in lines]
        except Exception:
            pass
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for k in ("records", "data", "items", "results", "markets"):
                v = data.get(k)
                if isinstance(v, list):
                    return v
            return [data]
    except Exception:
        return []
    return []


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_json(records: Iterable[Mapping[str, Any]], path: str, *, indent: int = 2) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(records), f, ensure_ascii=False, indent=indent)


# =========================
# main (no CLI): expand -> enrich -> (optional) write
# =========================

def _expand_inputs(
    inputs: Union[str, Mapping[str, Any], Iterable[Mapping[str, Any]], Iterable[str]]
) -> List[Mapping[str, Any]]:
    import os, glob
    if isinstance(inputs, Mapping):
        return [inputs]  # type: ignore[list-item]
    if isinstance(inputs, Iterable) and not isinstance(inputs, (str, bytes)):
        lst = list(inputs)
        if not lst:
            return []
        if isinstance(lst[0], Mapping):
            return lst  # snapshots already
        # assume paths/globs
        paths: List[str] = []
        for pat in lst:  # type: ignore[assignment]
            if not isinstance(pat, str):
                continue
            matches = sorted(glob.glob(pat))
            if matches:
                paths.extend(matches)
            elif os.path.exists(pat):
                paths.append(pat)
        recs: List[Mapping[str, Any]] = []
        for p in paths:
            recs.extend(read_jsonish(p))
        return recs
    if isinstance(inputs, str):
        import glob, os
        cands = sorted(glob.glob(inputs)) or ([inputs] if os.path.exists(inputs) else [])
        recs: List[Mapping[str, Any]] = []
        for p in cands:
            recs.extend(read_jsonish(p))
        return recs
    return []


def main(
    inputs: Union[str, Mapping[str, Any], Iterable[Mapping[str, Any]], Iterable[str]],
    *,
    out_json: Optional[str] = None,
    out_jsonl: Optional[str] = None,
    limit: Optional[int] = None,
    # enrichment behavior
    features_key: Optional[str] = "features",
    flatten: bool = True,
    prefix: str = "",
    only_features: bool = True,
    inplace: bool = False,
    # metadata behavior
    include_meta: bool = True,
    meta_mode: str = "all_no_ids",  # "all_no_ids" | "minimal"
    meta_keys: Tuple[str, ...] = ("slug", "question", "choices"),
    return_records: bool = True,
) -> Optional[List[Dict[str, Any]]]:
    """
    expand -> extract -> attach -> optionally write -> return.

    Defaults are tuned for modeling + rich metadata without IDs:
      • flatten=True, prefix=""   → put features at top-level
      • only_features=True        → drop original fields
      • include_meta=True + meta_mode="all_no_ids"
    """
    records = _expand_inputs(inputs)
    if not records:
        print("[warn] no snapshots to process.")
        return None

    if limit is not None:
        records = records[: int(limit)]

    enriched = enrich_many(
        records,
        key=features_key,
        flatten=flatten,
        prefix=prefix,
        only_features=only_features,
        inplace=inplace,
        include_meta=include_meta,
        meta_mode=meta_mode,
        meta_keys=meta_keys,
    )

    if out_jsonl:
        write_jsonl(enriched, out_jsonl)
        print(f"[ok] wrote {out_jsonl}")
    if out_json:
        write_json(enriched, out_json)
        print(f"[ok] wrote {out_json}")

    return enriched if return_records else None


if __name__ == "__main__":
    # quick dev run: edit inputs or call from your pipeline
    main(
        inputs=[
            "data/daily/polymarket_*.jsonl",
            "data/polymarket_*.ndjson",
            "data/polymarket_*.json",
        ],
        out_jsonl="data/outputs/enriched_poly.jsonl",
        out_json=None,
        limit=None,
        # keep only features + metadata (no original fields)
        flatten=True,
        prefix="",
        only_features=True,
        include_meta=True,
        meta_mode="all_no_ids",  # <- this is the key bit for “all the metadata except ids”
        return_records=True,
    )
