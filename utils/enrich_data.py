# utils/enrich_data.py
from __future__ import annotations
from typing import Iterable, Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np

# ---------- Time normalization ----------

def ensure_created_datetime(df: pd.DataFrame, tz: str = "Europe/London") -> pd.DataFrame:
    """
    Add/normalize:
      - df['created_dt']  : timezone-aware UTC datetime
      - df['created_iso'] : ISO string in target timezone
    Works with any of: 'created_iso' (string), 'created_utc' (epoch s/ms), 'created' (string).
    Safe to call even if none exist (will create NaT).
    """
    if df is None or len(df) == 0:
        return df

    dt = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    if "created_iso" in df.columns:
        parsed = pd.to_datetime(df["created_iso"], errors="coerce", utc=True)
        dt = parsed.fillna(dt)

    if "created_utc" in df.columns:
        s = df["created_utc"]
        if pd.api.types.is_numeric_dtype(s):
            unit = "ms" if s.dropna().abs().gt(1e12).any() else "s"
            parsed = pd.to_datetime(s, unit=unit, errors="coerce", utc=True)
            dt = dt.fillna(parsed)

    if "created" in df.columns:
        parsed = pd.to_datetime(df["created"], errors="coerce", utc=True)
        dt = dt.fillna(parsed)

    df["created_dt"] = dt
    # ISO output in target tz (e.g., Europe/London)
    df["created_iso"] = (
        df["created_dt"]
        .dt.tz_convert(tz)
        .dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    )
    return df


# ---------- Re-enrichment helpers ----------

REDDIT_META_CANDIDATES = [
    "created_dt", "created_iso", "created_utc",
    "source", "title", "text",
    "subreddit", "subreddit_id", "subreddit_subscribers",
    "author", "author_fullname", "author_premium",
    "score", "upvote_ratio", "num_comments",
    "flair", "is_ad", "content_type",
    "media_url", "url",
    # anything else you want to carry over can be added here
]

POLY_META_CANDIDATES = [
    "created_dt", "created_iso",
    "title", "url",
    "market_type", "category", "subcategory",
    "start_date", "end_date",
    "volume", "liquidity",
    "yes_price", "no_price", "probability",
    # add any other fields you keep in your polymarket_df
]

def re_enrich_doc_info(
    doc_info: pd.DataFrame,
    source_df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    id_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Attach source metadata columns from `source_df` onto BERTopic's `doc_info`.

    Modes:
      - ID-based merge: if `id_col` exists in both frames (robust if rows were filtered/reordered)
      - Positional join: if lengths match and no id provided

    `cols` limits which columns to copy from source. If None, auto-select everything
    except large/vector-ish or columns already in doc_info.
    """
    out = doc_info.reset_index(drop=True).copy()
    src = source_df.reset_index(drop=True).copy()

    if cols is None:
        skip = set(out.columns) | {"embedding", "embeddings"}
        cols = [c for c in src.columns if c not in skip]

    # Prefer ID-based merge when possible
    if id_col and (id_col in out.columns) and (id_col in src.columns):
        merged = out.merge(src[[id_col] + [c for c in cols if c in src.columns]], on=id_col, how="left")
        return merged

    # Otherwise positional-join (requires same length)
    if len(out) != len(src):
        raise ValueError(
            f"Cannot positional-join: doc_info len={len(out)} vs source len={len(src)}. "
            "Provide id_col for an ID-based merge, or ensure lengths match."
        )

    for c in cols:
        if c not in src.columns:
            continue
        # don't overwrite if already exists; if needed, write to suffixed column
        if c in out.columns:
            out[f"{c}_src"] = src[c].to_numpy()
        else:
            out[c] = src[c].to_numpy()
    return out


# ---------- Topic utilities ----------

def _topic_keywords(topic_model, topic_id: int, top_n: int = 10) -> List[str]:
    """Return top N keywords for a given topic_id from BERTopic model."""
    words = topic_model.get_topic(topic_id)  # list of (word, weight)
    if not words:
        return []
    return [w for (w, _w) in words[:top_n]]

def _get_topic_id_column(df: pd.DataFrame) -> str:
    """Support either 'topic' or 'Topic' column naming."""
    if "topic" in df.columns:
        return "topic"
    if "Topic" in df.columns:
        return "Topic"
    raise KeyError("Centroids dataframe must contain 'topic' or 'Topic' column.")

def _subset_docs(
    doc_info_enriched: pd.DataFrame,
    topic_id: int
) -> pd.DataFrame:
    """Subset doc_info rows for a given topic id."""
    if "Topic" not in doc_info_enriched.columns:
        raise KeyError("doc_info must contain 'Topic' column from BERTopic.")
    return doc_info_enriched.loc[doc_info_enriched["Topic"] == topic_id]

def _collect_documents_payload(
    topic_rows: pd.DataFrame,
    cleaned_texts: List[str],
    meta_cols: List[str]
) -> List[Dict[str, Any]]:
    """
    Build per-document payloads including cleaned text + selected metadata.
    Assumes doc_info has 'Document' index mapping or we use row index position.
    BERTopic's doc_info includes a 'Document' column with the (preprocessed) text;
    but we prefer the externally supplied cleaned_texts list using the doc_info index.
    """
    out = []
    # Ensure we have index that maps to position in cleaned_texts
    for idx, row in topic_rows.reset_index().iterrows():
        # original row index in doc_info corresponds to position in cleaned_texts
        orig_idx = int(row["index"]) if "index" in row else int(topic_rows.index[idx])
        payload = {
            "doc_index": orig_idx,
            "text": cleaned_texts[orig_idx] if 0 <= orig_idx < len(cleaned_texts) else row.get("Document", ""),
            "probability": row.get("Probability", None),
        }
        for c in meta_cols:
            if c in topic_rows.columns:
                payload[c] = row.get(c, None)
        out.append(payload)
    return out

def _topic_dates_summary(topic_rows: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Build min/max created_iso (if present) for the topic subset."""
    if "created_dt" in topic_rows.columns:
        dt_min = topic_rows["created_dt"].min()
        dt_max = topic_rows["created_dt"].max()
        # Render as ISO if tz-aware, else str
        def _fmt(x):
            if pd.isna(x):
                return None
            try:
                return x.tz_convert("Europe/London").strftime("%Y-%m-%dT%H:%M:%S%z")
            except Exception:
                return str(x)
        return {
            "min_created_iso": _fmt(dt_min),
            "max_created_iso": _fmt(dt_max),
        }
    if "created_iso" in topic_rows.columns:
        # Fallback if only string is present
        vals = topic_rows["created_iso"].dropna().astype(str)
        return {
            "min_created_iso": vals.min() if not vals.empty else None,
            "max_created_iso": vals.max() if not vals.empty else None,
        }
    return {"min_created_iso": None, "max_created_iso": None}


# ---------- Alignment helpers ----------

def _build_alignment_from_sim_matrix(
    sim_matrix: pd.DataFrame,
    threshold: float
) -> pd.DataFrame:
    """
    Expect sim_matrix with:
      - index: reddit topic ids
      - columns: polymarket topic ids
      - values: similarity
    Returns a long DataFrame: ['reddit_topic','polymarket_topic','similarity'] filtered by threshold,
    keeping only the best polymarket match per reddit topic (greedy by max sim).
    """
    long_df = (
        sim_matrix.reset_index()
        .melt(id_vars=sim_matrix.index.name or "index", var_name="polymarket_topic", value_name="similarity")
    )
    # Normalize column names
    if (sim_matrix.index.name or "index") != "reddit_topic":
        long_df = long_df.rename(columns={sim_matrix.index.name or "index": "reddit_topic"})
    # filter by threshold
    long_df = long_df.dropna(subset=["similarity"])
    long_df = long_df[long_df["similarity"] >= threshold]
    # keep best per reddit topic
    long_df = long_df.sort_values(["reddit_topic", "similarity"], ascending=[True, False])
    best = long_df.groupby("reddit_topic", as_index=False).first()
    return best[["reddit_topic", "polymarket_topic", "similarity"]]


# ---------- Trend metrics (velocity / acceleration) ----------

def _topic_timeseries_counts(
    topic_rows: pd.DataFrame,
    freq: str = "D",
    tz: str = "Europe/London"
) -> pd.Series:
    """
    Return a pandas Series of counts indexed by period (e.g., daily) for the topic.
    Uses 'created_dt' when present; otherwise parses 'created_iso' best-effort.
    """
    ts = None
    if "created_dt" in topic_rows.columns:
        ts = topic_rows["created_dt"]
    elif "created_iso" in topic_rows.columns:
        ts = pd.to_datetime(topic_rows["created_iso"], errors="coerce", utc=True)
    else:
        return pd.Series(dtype="int64")

    ts = ts.dropna()
    if ts.empty:
        return pd.Series(dtype="int64")

    # Normalize to desired timezone then to period frequency
    ts = ts.dt.tz_convert(tz)
    # Count per period
    counts = (
        ts.dt.floor(freq)
          .value_counts()
          .sort_index()
          .asfreq(freq, fill_value=0)
    )
    return counts


def _linreg_slope(y: np.ndarray) -> Optional[float]:
    """
    Return slope from simple linear regression y ~ a*x + b with x = 0..len(y)-1.
    Returns None if not enough points.
    """
    if y is None or len(y) < 2:
        return None
    x = np.arange(len(y), dtype=float)
    # np.polyfit returns [slope, intercept]
    slope = np.polyfit(x, y.astype(float), 1)[0]
    return float(slope)


def _compute_velocity_acceleration(
    counts: pd.Series,
    velocity_window_days: int = 14,
    smooth_window: int = 3
) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    """
    counts: time-indexed Series (e.g., daily counts). Missing days should be filled with 0 already.
    velocity: slope of smoothed counts over last N days (counts per day).
    acceleration: slope of the velocity series (first differences of smoothed counts) over last N-1 days.
    Returns (velocity, acceleration, recap_metrics).
    """
    if counts is None or counts.empty:
        return None, None, {"last_7d": 0, "prev_7d": 0, "wow_change": None}

    # Ensure daily frequency and fill gaps
    s = counts.asfreq("D", fill_value=0)

    # Smooth (centered rolling mean)
    if smooth_window > 1:
        s_sm = s.rolling(window=smooth_window, min_periods=1, center=True).mean()
    else:
        s_sm = s.astype(float)

    # Tail windows
    tail = s_sm.iloc[-velocity_window_days:]
    v_series = tail.diff().dropna()  # day-over-day change

    velocity = _linreg_slope(tail.values) if len(tail) >= 2 else None
    acceleration = _linreg_slope(v_series.values) if len(v_series) >= 2 else None

    # Recap metrics (useful in UIs)
    last_7 = int(s.iloc[-7:].sum()) if len(s) >= 1 else 0
    prev_7 = int(s.iloc[-14:-7].sum()) if len(s) >= 8 else 0
    wow = (last_7 - prev_7) / prev_7 if prev_7 > 0 else (None if last_7 == 0 else float("inf"))

    recap = {
        "last_7d": last_7,
        "prev_7d": prev_7,
        "wow_change": wow,
    }
    return velocity, acceleration, recap


# ---------- Main builder ----------

def build_exact_aligned_topics_with_dates_and_meta(
    reddit_centroids: pd.DataFrame,
    polymarket_centroids: pd.DataFrame,
    sim_matrix: pd.DataFrame,
    reddit_topic_model,
    polymarket_topic_model,
    reddit_doc_info: pd.DataFrame,
    poly_doc_info: pd.DataFrame,
    cleaned_reddit_texts: List[str],
    cleaned_poly_texts: List[str],
    threshold: float = 0.70,
    reddit_raw_df: Optional[pd.DataFrame] = None,
    polymarket_raw_df: Optional[pd.DataFrame] = None,
    alignment_df: Optional[pd.DataFrame] = None,
    top_n_words: int = 10,
) -> List[Dict[str, Any]]:
    """
    Build the final 'aligned_full' JSON-like structure with:
      - group_id
      - source: 'Merged', 'Reddit Only', 'Polymarket Only'
      - similarity (if merged)
      - reddit_topic {id, label, keywords, size, date_range, documents:[...]}
      - polymarket_topic {id, label, keywords, size, date_range, documents:[...]}

    Enriches doc_info with *all available* Reddit/Polymarket fields.
    """
    # ---- Enrich doc_info with raw API fields ----
    if reddit_raw_df is not None:
        ensure_created_datetime(reddit_raw_df, tz="Europe/London")
        reddit_doc_info = re_enrich_doc_info(
            reddit_doc_info,
            reddit_raw_df,
            cols=[c for c in REDDIT_META_CANDIDATES if c in reddit_raw_df.columns],
            id_col=("doc_id" if "doc_id" in reddit_doc_info.columns and "doc_id" in reddit_raw_df.columns else None),
        )
    else:
        # still ensure time columns exist to avoid downstream KeyErrors
        if "created_dt" not in reddit_doc_info.columns and "created_iso" not in reddit_doc_info.columns:
            reddit_doc_info["created_dt"] = pd.NaT
            reddit_doc_info["created_iso"] = None

    if polymarket_raw_df is not None:
        ensure_created_datetime(polymarket_raw_df, tz="Europe/London")
        poly_doc_info = re_enrich_doc_info(
            poly_doc_info,
            polymarket_raw_df,
            cols=[c for c in POLY_META_CANDIDATES if c in polymarket_raw_df.columns],
            id_col=("doc_id" if "doc_id" in poly_doc_info.columns and "doc_id" in polymarket_raw_df.columns else None),
        )
    else:
        if "created_dt" not in poly_doc_info.columns and "created_iso" not in poly_doc_info.columns:
            poly_doc_info["created_dt"] = pd.NaT
            poly_doc_info["created_iso"] = None

    # ---- Prepare topic id columns ----
    r_topic_col = _get_topic_id_column(reddit_centroids)
    p_topic_col = _get_topic_id_column(polymarket_centroids)

    reddit_topic_ids = set(reddit_centroids[r_topic_col].astype(int).tolist())
    poly_topic_ids   = set(polymarket_centroids[p_topic_col].astype(int).tolist())

    # ---- Alignment pairs (merged) ----
    if alignment_df is None:
        if not isinstance(sim_matrix, pd.DataFrame):
            raise TypeError("sim_matrix must be a pandas DataFrame or provide alignment_df.")
        alignment_df = _build_alignment_from_sim_matrix(sim_matrix, threshold)

    merged_pairs: List[Tuple[int, int, float]] = []
    for _, row in alignment_df.iterrows():
        merged_pairs.append((
            int(row["reddit_topic"]),
            int(row["polymarket_topic"]),
            float(row["similarity"])
        ))

    matched_reddit = {r for r, _p, _s in merged_pairs}
    matched_poly   = {p for _r, p, _s in merged_pairs}

    reddit_only_ids = sorted(list(reddit_topic_ids - matched_reddit))
    poly_only_ids   = sorted(list(poly_topic_ids - matched_poly))

    # ---- Builder for each topic side ----
    def build_topic_payload(
        side: str,
        topic_id: int,
        topic_model,
        doc_info_enriched: pd.DataFrame,
        cleaned_texts: List[str],
        centroids_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        # Label
        label_col = "label" if "label" in centroids_df.columns else ("Name" if "Name" in centroids_df.columns else None)
        topic_col = _get_topic_id_column(centroids_df)
        label = None
        if label_col:
            lab = centroids_df.loc[centroids_df[topic_col] == topic_id, label_col]
            if len(lab):
                label = lab.iloc[0]

        # Keywords
        keywords = _topic_keywords(topic_model, topic_id, top_n=top_n_words)

        # Documents (subset)
        topic_rows = _subset_docs(doc_info_enriched, topic_id)
        meta_cols = [
            *REDDIT_META_CANDIDATES,
            *[c for c in POLY_META_CANDIDATES if c not in REDDIT_META_CANDIDATES],
        ]
        documents = _collect_documents_payload(topic_rows, cleaned_texts, meta_cols)
        size = len(documents)

        # Date summary
        date_range = _topic_dates_summary(topic_rows)

        # ---- NEW: time-series + velocity / acceleration ----
        counts = _topic_timeseries_counts(topic_rows, freq="D", tz="Europe/London")
        velocity, acceleration, recap = _compute_velocity_acceleration(
            counts,
            velocity_window_days=14,   # tweak as you like
            smooth_window=3            # tweak as you like
        )

        return {
            "id": topic_id,
            "label": label,
            "keywords": keywords,
            "size": size,
            "date_range": date_range,
            "documents": documents,
            "metrics": {
                "timeseries_freq": "D",
                "velocity_window_days": 14, #CHANGE THIS IF YOU CHANGE THE NUMBER OF DAYS
                "smooth_window": 3,
                "velocity": velocity,             # counts/day
                "acceleration": acceleration,     # counts/day^2
                "last_7d": recap["last_7d"],
                "prev_7d": recap["prev_7d"],
                "wow_change": recap["wow_change"],  # None/inf/float
            },
            # Optional: expose raw series if you want to chart it downstream
            # "timeseries_counts": counts.to_dict(),  # datetime -> int
        }


    # ---- Build groups ----
    groups: List[Dict[str, Any]] = []
    gid = 0

    # Merged groups
    for r_id, p_id, sim in merged_pairs:
        groups.append({
            "group_id": gid,
            "source": "Merged",
            "similarity": sim,
            "reddit_topic": build_topic_payload(
                "reddit", r_id, reddit_topic_model, reddit_doc_info, cleaned_reddit_texts, reddit_centroids
            ),
            "polymarket_topic": build_topic_payload(
                "polymarket", p_id, polymarket_topic_model, poly_doc_info, cleaned_poly_texts, polymarket_centroids
            ),
        })
        gid += 1

    # Reddit-only
    for r_id in reddit_only_ids:
        groups.append({
            "group_id": gid,
            "source": "Reddit Only",
            "similarity": None,
            "reddit_topic": build_topic_payload(
                "reddit", r_id, reddit_topic_model, reddit_doc_info, cleaned_reddit_texts, reddit_centroids
            ),
            "polymarket_topic": None,
        })
        gid += 1

    # Polymarket-only
    for p_id in poly_only_ids:
        groups.append({
            "group_id": gid,
            "source": "Polymarket Only",
            "similarity": None,
            "reddit_topic": None,
            "polymarket_topic": build_topic_payload(
                "polymarket", p_id, polymarket_topic_model, poly_doc_info, cleaned_poly_texts, polymarket_centroids
            ),
        })
        gid += 1

    return groups

from datetime import datetime, date  # add near other imports in __main__

def _json_default(o):
    import numpy as _np
    import pandas as _pd
    if isinstance(o, (_pd.Timestamp, datetime, date)):
        # Convert pandas/py datetimes to ISO8601
        # If tz-aware, keep it; else make an ISO string
        try:
            return o.isoformat()
        except Exception:
            return str(o)
    if o is _pd.NaT:
        return None
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, set):
        return list(o)
    return str(o)  # last resort


if __name__ == "__main__":
    """
    Test runner for enrich_data:
      • loads freshest Reddit & Polymarket daily dumps (data/daily)
      • builds corpora (same cleaners as pipeline.topic_modelling)
      • fits BERTopic for Reddit & Polymarket
      • computes centroids + cross-source similarity
      • enriches doc_info with original metadata + time fields
      • writes aligned_full_YYYYmmdd_HHMMSS.json to data/outputs
    """
    import os, sys, json, random
    from pathlib import Path
    from datetime import datetime

    # ---- make repo importable then pull modelling helpers ----
    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(REPO_ROOT))
    from pipeline.topic_modelling import (
        build_bert_corpus_from_reddit,
        build_bert_corpus_from_polymarket_snapshots,
        embed_and_fit,
        compute_topic_centroids,
        extract_semantically_overlapping_topics,
    )

    DAILY_DIR = REPO_ROOT / "data" / "daily"
    OUT_DIR   = REPO_ROOT / "data" / "outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- helpers ----------
    def _latest(pattern: str) -> Optional[Path]:
        files = list(DAILY_DIR.glob(pattern))
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    def _load_jsonish(path: Optional[Path]) -> list[dict]:
        if not path or not path.exists():
            return []
        txt = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt:
            return []
        # Try JSONL first
        if "\n" in txt and txt.lstrip().startswith("{"):
            try:
                return [json.loads(line) for line in txt.splitlines() if line.strip()]
            except Exception:
                pass
        # Fallback: regular JSON
        try:
            data = json.loads(txt)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for k in ("records", "data", "items", "results", "markets"):
                    if isinstance(data.get(k), list):
                        return data[k]
                return [data]
        except Exception:
            return []
        return []

    def _encode_for_centroids(backend_or_st, texts, fallback_st=None):
        """
        Try BERTopic backend API first; fall back to SentenceTransformer.encode.
        """
        try:
            return np.asarray(backend_or_st.embed_documents(texts))
        except Exception:
            pass
        try:
            return backend_or_st.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        except Exception:
            from sentence_transformers import SentenceTransformer
            st = fallback_st or SentenceTransformer("all-MiniLM-L6-v2")
            return st.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # ---------- pick inputs ----------
    reddit_path = _latest("reddit_daily_all_*.ndjson") or _latest("reddit_daily_all_*.json")
    popular_path = _latest("reddit_popular_*.ndjson")  # optional
    poly_path = _latest("polymarket_*.jsonl") or _latest("polymarket_*.json")

    print("Inputs:")
    print("  Reddit all :", reddit_path)
    print("  Reddit pop :", popular_path)
    print("  Polymarket :", poly_path)

    reddit_raw = _load_jsonish(reddit_path) + _load_jsonish(popular_path)
    poly_raw   = _load_jsonish(poly_path)

    if not reddit_raw or not poly_raw:
        raise SystemExit("Missing inputs: need both Reddit and Polymarket daily dumps under data/daily.")

    # ---------- small cap for quick run ----------
    random.seed(42)
    CAP = int(os.environ.get("ENRICH_CAP", "2000"))
    if len(reddit_raw) > CAP: reddit_raw = random.sample(reddit_raw, CAP)
    if len(poly_raw)   > CAP: poly_raw   = random.sample(poly_raw,   CAP)

    # ---------- corpora ----------
    _, reddit_texts = build_bert_corpus_from_reddit(
        reddit_raw,
        include_title=True,
        include_text=True,
        include_top_comments=0,
        model_name="bert-base-uncased",
        max_length=128,
        phrase_min_count=5,
        phrase_threshold=10,
    )
    _, poly_texts = build_bert_corpus_from_polymarket_snapshots(
        poly_raw,
        include_tag=True,
        include_outcomes=True,
        include_slug=False,
        model_name="bert-base-uncased",
        max_length=128,
        phrase_min_count=5,
        phrase_threshold=10,
    )
    reddit_texts = [t for t in reddit_texts if t and t.strip()]
    poly_texts   = [t for t in poly_texts   if t and t.strip()]
    print(f"Clean texts → reddit={len(reddit_texts)}  poly={len(poly_texts)}")

    # ---------- topic models (handle both 3- and 4-value returns) ----------
    print("Fitting BERTopic (Reddit)…")
    _ret_r = embed_and_fit(reddit_texts, ngram_range=(1,3), min_df=2, max_df=0.9)
    if isinstance(_ret_r, tuple) and len(_ret_r) == 4:
        r_topics, _, r_model, r_st_model = _ret_r
    else:
        r_topics, _, r_model = _ret_r
        r_st_model = None

    print("Fitting BERTopic (Polymarket)…")
    _ret_p = embed_and_fit(poly_texts, ngram_range=(1,3), min_df=2, max_df=0.9)
    if isinstance(_ret_p, tuple) and len(_ret_p) == 4:
        p_topics, _, p_model, p_st_model = _ret_p
    else:
        p_topics, _, p_model = _ret_p
        p_st_model = None

    # ---------- document info ----------
    r_doc = r_model.get_document_info(reddit_texts)
    p_doc = p_model.get_document_info(poly_texts)

    # ---------- add embeddings for centroid calc ----------
    r_emb = _encode_for_centroids(
        getattr(r_model, "embedding_model", r_st_model) or (r_st_model),
        reddit_texts,
        fallback_st=r_st_model,
    )
    p_emb = _encode_for_centroids(
        getattr(p_model, "embedding_model", p_st_model) or (p_st_model),
        poly_texts,
        fallback_st=p_st_model,
    )
    r_doc["embedding"] = [e.tolist() for e in r_emb]
    p_doc["embedding"] = [e.tolist() for e in p_emb]

    # ---------- centroids ----------
    r_centroids = compute_topic_centroids(r_doc, emb_col="embedding", topic_col="Topic", label_col="Name")
    p_centroids = compute_topic_centroids(p_doc, emb_col="embedding", topic_col="Topic", label_col="Name")

    # normalize to expected schema for builder
    if "topic_id" in r_centroids.columns: r_centroids = r_centroids.rename(columns={"topic_id": "Topic"})
    if "label"   in r_centroids.columns: r_centroids = r_centroids.rename(columns={"label":   "Name"})
    if "topic_id" in p_centroids.columns: p_centroids = p_centroids.rename(columns={"topic_id": "Topic"})
    if "label"   in p_centroids.columns: p_centroids = p_centroids.rename(columns={"label":   "Name"})

    # ---------- alignment ----------
    THRESHOLD = float(os.environ.get("ALIGN_THRESHOLD", "0.60"))
    alignment_df, sim_matrix = extract_semantically_overlapping_topics(
        r_centroids, p_centroids, threshold=THRESHOLD
    )

    # ---------- raw DataFrames for re-enrichment ----------
    reddit_df = pd.DataFrame(reddit_raw)
    poly_df   = pd.DataFrame(poly_raw)

    # ---------- build aligned structure ----------
    aligned = build_exact_aligned_topics_with_dates_and_meta(
        reddit_centroids=r_centroids,
        polymarket_centroids=p_centroids,
        sim_matrix=sim_matrix,                 # full matrix (DataFrame)
        reddit_topic_model=r_model,
        polymarket_topic_model=p_model,
        reddit_doc_info=r_doc,
        poly_doc_info=p_doc,
        cleaned_reddit_texts=reddit_texts,
        cleaned_poly_texts=poly_texts,
        threshold=THRESHOLD,
        reddit_raw_df=reddit_df,
        polymarket_raw_df=poly_df,
        alignment_df=alignment_df,             # explicit pairs (best per reddit topic)
        top_n_words=10,
    )

    # ---------- write output ----------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"aligned_full_{ts}.json"
    out_path.write_text(
    json.dumps(aligned, ensure_ascii=False, indent=2, default=_json_default),
    encoding="utf-8"
)

    print(f"\n✅ Wrote {out_path}")

    #TODO polymarket not enriched properly, need all the other stuff