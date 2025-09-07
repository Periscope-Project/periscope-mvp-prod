# utils/enrich_data.py
from __future__ import annotations

from typing import Iterable, Optional, Dict, Any, List, Tuple
import json
import math
import numpy as np
import pandas as pd

# Try to import the feature extractor from either location
try:
    from utils.polymarket_features import extract_market_features as _extract_market_features
except Exception:
    # fallback to top-level module name
    import polymarket_features as _pf  # type: ignore
    _extract_market_features = _pf.extract_market_features  # type: ignore


# ======================================================================================
# Time normalization
# ======================================================================================

def ensure_created_datetime(df: pd.DataFrame, tz: str = "Europe/London") -> pd.DataFrame:
    """
    Add/normalize:
      - df['created_dt']  : timezone-aware UTC datetime
      - df['created_iso'] : ISO string in target timezone
    Accepts any of: 'created_iso' (string), 'created_utc' (epoch s/ms), 'created' (string).
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
    df["created_iso"] = (
        df["created_dt"]
        .dt.tz_convert(tz)
        .dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    )
    return df


# ======================================================================================
# Lightweight helpers
# ======================================================================================

_EXCLUDE_DOCINFO = {
    "Document", "Topic", "Name", "Representation", "Probability",
    "embedding", "embeddings", "index"
}

POLY_FEATURE_PREFIX = "feat_"
POLY_EXTRA_META     = ["question", "slug", "outcomes"]  # human-usable text bits


def _mget_market(row: Dict[str, Any], key: str):
    m = row.get("market") if isinstance(row, dict) else None
    return (m or {}).get(key)


def _jsonish(x: Any) -> Any:
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


def _all_meta_cols(df: pd.DataFrame) -> List[str]:
    """Everything except typical BERTopic columns and heavy vectors."""
    return [c for c in df.columns if c not in _EXCLUDE_DOCINFO]


# ======================================================================================
# Build side DataFrames the main will pass back for enrichment (doc_id alignment)
# ======================================================================================

def make_polymarket_meta_with_features(poly_raw: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """
    From raw Polymarket snapshot records:
      - flattens 'question','slug','outcomes'
      - computes model-ready features via extract_market_features()
      - normalizes time, using feat_collected_ts if available
      - adds doc_id (0..n-1) for robust ID-based re-enrichment

    Returns a DataFrame where ALL feature columns are prefixed with 'feat_'.
    """
    poly_raw = list(poly_raw)
    df = pd.DataFrame(poly_raw)

    # text meta that you want to keep in outputs
    df["question"] = df.apply(lambda r: _mget_market(r, "question"), axis=1)
    df["slug"]     = df.apply(lambda r: _mget_market(r, "slug"), axis=1)
    df["outcomes"] = df.apply(lambda r: _jsonish(_mget_market(r, "outcomes")), axis=1)

    # compute features (one per input record), then add with 'feat_' prefix
    feats = [_extract_market_features(rec) for rec in poly_raw]
    feats_df = pd.DataFrame(feats).add_prefix(POLY_FEATURE_PREFIX)
    df = pd.concat([df.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)

    # time normalization — prefer features timestamp if present
    if "feat_collected_ts" in df.columns and "created_utc" not in df.columns:
        df["created_utc"] = df["feat_collected_ts"]
    df = ensure_created_datetime(df, tz="Europe/London")

    # stable id for ID-merge
    df["doc_id"] = np.arange(len(df))
    return df


def make_reddit_meta(reddit_raw: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """
    Simple Reddit side:
      - copy to DataFrame
      - normalize time (if created_* present)
      - add doc_id (0..n-1) for robust ID-based re-enrichment
    """
    reddit_raw = list(reddit_raw)
    df = pd.DataFrame(reddit_raw)
    df = ensure_created_datetime(df, tz="Europe/London")
    df["doc_id"] = np.arange(len(df))
    return df


# ======================================================================================
# Re-enrichment (attach meta back to BERTopic doc_info)
# ======================================================================================

def re_enrich_doc_info(
    doc_info: pd.DataFrame,
    source_df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    id_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Safely merge metadata back onto BERTopic's doc_info.
    - Dedup right-hand columns
    - Exclude id_col from 'cols' to avoid duplicate key in selection
    - Validate m:1 merge when using ID
    - Fall back to positional join only if lengths match
    """
    out = doc_info.reset_index(drop=True).copy()
    src = source_df.reset_index(drop=True).copy()

    # 1) de-duplicate any duplicate column names on the right
    src = src.loc[:, ~src.columns.duplicated(keep="first")]

    # 2) choose columns to copy
    if cols is None:
        skip = set(out.columns) | {"embedding", "embeddings"}
        cols = [c for c in src.columns if c not in skip]

    # 3) never include the id_col inside 'cols'
    if id_col:
        cols = [c for c in cols if c != id_col]

    # 4) prefer ID merge when possible
    if id_col and (id_col in out.columns) and (id_col in src.columns):
        # coerce id types to be comparable (nullable int)
        out[id_col] = pd.to_numeric(out[id_col], errors="coerce").astype("Int64")
        src[id_col] = pd.to_numeric(src[id_col], errors="coerce").astype("Int64")

        # sanity: right-hand ids must be unique for m:1 merge
        if not src[id_col].is_unique:
            dups = src[id_col][src[id_col].duplicated(keep=False)]
            raise ValueError(
                f"source_df has non-unique '{id_col}' values; sample: {dups.head().tolist()}"
            )

        return out.merge(
            src[[id_col] + [c for c in cols if c in src.columns]],
            on=id_col,
            how="left",
            validate="m:1",
        )

    # 5) fallback: positional join (same length required)
    if len(out) != len(src):
        raise ValueError(
            f"Cannot positional-join: doc_info len={len(out)} vs source len={len(src)}. "
            "Provide id_col for an ID-based merge, or ensure lengths match."
        )

    for c in cols:
        if c not in src.columns:
            continue
        if c in out.columns:
            out[f"{c}_src"] = src[c].to_numpy()
        else:
            out[c] = src[c].to_numpy()
    return out



# ======================================================================================
# Topic helpers (keywords, slicing, time series metrics)
# ======================================================================================

def _topic_keywords(topic_model, topic_id: int, top_n: int = 10) -> List[str]:
    words = topic_model.get_topic(topic_id)  # list[(word, weight)]
    if not words:
        return []
    return [w for (w, _w) in words[:top_n]]


def _get_topic_id_column(df: pd.DataFrame) -> str:
    if "topic" in df.columns:
        return "topic"
    if "Topic" in df.columns:
        return "Topic"
    raise KeyError("Centroids dataframe must contain 'topic' or 'Topic' column.")


def _subset_docs(doc_info_enriched: pd.DataFrame, topic_id: int) -> pd.DataFrame:
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
    Adds both 'doc_id' (if present) and 'doc_index' (row order fallback).
    """
    out = []
    base_cols = set(topic_rows.columns)
    use_cols = [c for c in meta_cols if c in base_cols]
    prob_col = "Probability" if "Probability" in base_cols else None

    for i, (idx, row) in enumerate(topic_rows.reset_index().iterrows()):
        orig_idx = int(row["index"]) if "index" in row else int(topic_rows.index[i])
        payload = {
            "doc_index": orig_idx,
            "text": cleaned_texts[orig_idx] if 0 <= orig_idx < len(cleaned_texts) else row.get("Document", ""),
        }
        if prob_col:
            payload["probability"] = row.get(prob_col, None)
        if "doc_id" in base_cols:
            payload["doc_id"] = row.get("doc_id", None)

        for c in use_cols:
            payload[c] = row.get(c, None)

        out.append(payload)
    return out


def _topic_dates_summary(topic_rows: pd.DataFrame) -> Dict[str, Optional[str]]:
    if "created_dt" in topic_rows.columns:
        dt_min = topic_rows["created_dt"].min()
        dt_max = topic_rows["created_dt"].max()

        def _fmt(x):
            if pd.isna(x):
                return None
            try:
                return x.tz_convert("Europe/London").strftime("%Y-%m-%dT%H:%M:%S%z")
            except Exception:
                return str(x)

        return {"min_created_iso": _fmt(dt_min), "max_created_iso": _fmt(dt_max)}

    if "created_iso" in topic_rows.columns:
        vals = topic_rows["created_iso"].dropna().astype(str)
        return {
            "min_created_iso": vals.min() if not vals.empty else None,
            "max_created_iso": vals.max() if not vals.empty else None,
        }
    return {"min_created_iso": None, "max_created_iso": None}


def _topic_timeseries_counts(topic_rows: pd.DataFrame, freq: str = "D", tz: str = "Europe/London") -> pd.Series:
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

    ts = ts.dt.tz_convert(tz)
    counts = (
        ts.dt.floor(freq)
          .value_counts()
          .sort_index()
          .asfreq(freq, fill_value=0)
    )
    return counts


def _linreg_slope(y: np.ndarray) -> Optional[float]:
    if y is None or len(y) < 2:
        return None
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, np.asarray(y, dtype=float), 1)[0]
    return float(slope)


def _compute_velocity_acceleration(
    counts: pd.Series,
    velocity_window_days: int = 14,
    smooth_window: int = 3
) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    if counts is None or counts.empty:
        return None, None, {"last_7d": 0, "prev_7d": 0, "wow_change": None}

    s = counts.asfreq("D", fill_value=0)
    s_sm = s.rolling(window=smooth_window, min_periods=1, center=True).mean() if smooth_window > 1 else s.astype(float)

    tail = s_sm.iloc[-velocity_window_days:]
    v_series = tail.diff().dropna()

    velocity = _linreg_slope(tail.values) if len(tail) >= 2 else None
    acceleration = _linreg_slope(v_series.values) if len(v_series) >= 2 else None

    last_7 = int(s.iloc[-7:].sum()) if len(s) >= 1 else 0
    prev_7 = int(s.iloc[-14:-7].sum()) if len(s) >= 8 else 0
    wow = (last_7 - prev_7) / prev_7 if prev_7 > 0 else (None if last_7 == 0 else float("inf"))

    return velocity, acceleration, {"last_7d": last_7, "prev_7d": prev_7, "wow_change": wow}


# ======================================================================================
# Aligned groups builder (final JSON-ready structure)
# ======================================================================================

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

    Re-enriches doc_info with ALL available fields from the raw DataFrames,
    including every 'feat_*' column from Polymarket.
    """
    # ---- Re-enrich doc_info with RAW meta (ID-based if 'doc_id' exists) ----
    if reddit_raw_df is not None:
        reddit_doc_info = re_enrich_doc_info(
            reddit_doc_info,
            reddit_raw_df,
            cols=_all_meta_cols(reddit_raw_df),
            id_col=("doc_id" if "doc_id" in reddit_doc_info.columns and "doc_id" in reddit_raw_df.columns else None),
        )
    else:
        if "created_dt" not in reddit_doc_info.columns and "created_iso" not in reddit_doc_info.columns:
            reddit_doc_info["created_dt"] = pd.NaT
            reddit_doc_info["created_iso"] = None

    if polymarket_raw_df is not None:
        poly_cols = _all_meta_cols(polymarket_raw_df)
        poly_doc_info = re_enrich_doc_info(
            poly_doc_info,
            polymarket_raw_df,
            cols=poly_cols,
            id_col=("doc_id" if "doc_id" in poly_doc_info.columns and "doc_id" in polymarket_raw_df.columns else None),
        )
    else:
        if "created_dt" not in poly_doc_info.columns and "created_iso" not in poly_doc_info.columns:
            poly_doc_info["created_dt"] = pd.NaT
            poly_doc_info["created_iso"] = None

    # ---- Topic id columns ----
    r_topic_col = _get_topic_id_column(reddit_centroids)
    p_topic_col = _get_topic_id_column(polymarket_centroids)

    reddit_topic_ids = set(reddit_centroids[r_topic_col].astype(int).tolist())
    poly_topic_ids   = set(polymarket_centroids[p_topic_col].astype(int).tolist())

    # ---- Alignment pairs (merged) ----
    if alignment_df is None:
        if not isinstance(sim_matrix, pd.DataFrame):
            raise TypeError("sim_matrix must be a pandas DataFrame or provide alignment_df.")
        long_df = (
            sim_matrix.reset_index()
            .melt(id_vars=sim_matrix.index.name or "index", var_name="polymarket_topic", value_name="similarity")
        )
        if (sim_matrix.index.name or "index") != "reddit_topic":
            long_df = long_df.rename(columns={sim_matrix.index.name or "index": "reddit_topic"})
        long_df = long_df.dropna(subset=["similarity"])
        long_df = long_df[long_df["similarity"] >= threshold]
        long_df = long_df.sort_values(["reddit_topic", "similarity"], ascending=[True, False])
        alignment_df = long_df.groupby("reddit_topic", as_index=False).first()[["reddit_topic", "polymarket_topic", "similarity"]]

    merged_pairs: List[Tuple[int, int, float]] = []
    for _, row in alignment_df.iterrows():
        merged_pairs.append((int(row["reddit_topic"]), int(row["polymarket_topic"]), float(row["similarity"])))

    matched_reddit = {r for r, _p, _s in merged_pairs}
    matched_poly   = {p for _r, p, _s in merged_pairs}

    reddit_only_ids = sorted(list(reddit_topic_ids - matched_reddit))
    poly_only_ids   = sorted(list(poly_topic_ids - matched_poly))

    # ---- Builder for each side ----
    def _build_topic_payload(
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

        # Documents for this topic
        topic_rows = _subset_docs(doc_info_enriched, topic_id)
        meta_cols = _all_meta_cols(topic_rows)

        documents = _collect_documents_payload(topic_rows, cleaned_texts, meta_cols)
        size = len(documents)

        # Date summary
        date_range = _topic_dates_summary(topic_rows)

        # Simple topic dynamics
        counts = _topic_timeseries_counts(topic_rows, freq="D", tz="Europe/London")
        velocity, acceleration, recap = _compute_velocity_acceleration(counts, velocity_window_days=14, smooth_window=3)

        return {
            "id": topic_id,
            "label": label,
            "keywords": keywords,
            "size": size,
            "date_range": date_range,
            "documents": documents,
            "metrics": {
                "timeseries_freq": "D",
                "velocity_window_days": 14,
                "smooth_window": 3,
                "velocity": velocity,
                "acceleration": acceleration,
                "last_7d": recap["last_7d"],
                "prev_7d": recap["prev_7d"],
                "wow_change": recap["wow_change"],
            },
        }

    # ---- Build groups ----
    groups: List[Dict[str, Any]] = []
    gid = 0

    # Merged
    for r_id, p_id, sim in merged_pairs:
        groups.append({
            "group_id": gid,
            "source": "Merged",
            "similarity": sim,
            "reddit_topic": _build_topic_payload("reddit", r_id, reddit_topic_model, reddit_doc_info, cleaned_reddit_texts, reddit_centroids),
            "polymarket_topic": _build_topic_payload("polymarket", p_id, polymarket_topic_model, poly_doc_info, cleaned_poly_texts, polymarket_centroids),
        })
        gid += 1

    # Reddit-only
    for r_id in reddit_only_ids:
        groups.append({
            "group_id": gid,
            "source": "Reddit Only",
            "similarity": None,
            "reddit_topic": _build_topic_payload("reddit", r_id, reddit_topic_model, reddit_doc_info, cleaned_reddit_texts, reddit_centroids),
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
            "polymarket_topic": _build_topic_payload("polymarket", p_id, polymarket_topic_model, poly_doc_info, cleaned_poly_texts, polymarket_centroids),
        })
        gid += 1

    return groups


if __name__ == "__main__":
    """
    Offline smoke test for enrich_data:
      • loads freshest Reddit & Polymarket dumps from data/daily (no network calls)
      • builds corpora with pipeline/topic_modelling.py
      • fits BERTopic on each side
      • computes centroids + alignment
      • re-enriches doc_info with ALL meta + feat_* (ID-based merge via doc_id)
      • writes aligned_full_YYYYmmdd_HHMMSS.json to data/outputs
    """
    import os, sys, json, random
    from pathlib import Path
    from datetime import datetime
    import numpy as np
    import pandas as pd

    # --- repo paths ---
    REPO_ROOT = Path(__file__).resolve().parents[1]
    DAILY_DIR = REPO_ROOT / "data" / "daily"
    OUT_DIR   = REPO_ROOT / "data" / "outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # make 'pipeline' importable
    sys.path.append(str(REPO_ROOT))

    # use your existing topic_modelling.py
    from pipeline.topic_modelling import (
        build_bert_corpus_from_reddit,
        build_bert_corpus_from_polymarket_snapshots,
        embed_and_fit,
        compute_topic_centroids,
        extract_semantically_overlapping_topics,
    )

    # ---------- helpers ----------
    def _latest(pattern: str) -> Path | None:
        files = list(DAILY_DIR.glob(pattern))
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    def _load_jsonish(path: Path | None) -> list[dict]:
        if not path or not path.exists():
            return []
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return []
        # Try JSONL first
        if "\n{" in text or (text.startswith("{") and "\n" in text):
            try:
                return [json.loads(line) for line in text.splitlines() if line.strip()]
            except Exception:
                pass
        # Fallback: JSON array/object
        try:
            data = json.loads(text)
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

    # ---------- pick inputs (NO fetching) ----------
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
        raise SystemExit("Missing inputs: need both Reddit and Polymarket dumps under data/daily.")

    # ---------- corpora (TEXT ONLY to BERTopic) ----------
    # Build initial text arrays (1:1 with raw)
    _, reddit_texts_all = build_bert_corpus_from_reddit(
        reddit_raw,
        include_title=True,
        include_text=True,
        include_top_comments=0,
        model_name="bert-base-uncased",
        max_length=128,
        phrase_min_count=5,
        phrase_threshold=10,
    )
    _, poly_texts_all = build_bert_corpus_from_polymarket_snapshots(
        poly_raw,
        include_tag=True,
        include_outcomes=True,
        include_slug=False,
        model_name="bert-base-uncased",
        max_length=128,
        phrase_min_count=5,
        phrase_threshold=10,
    )

    # Keep mapping of non-empty texts → stable doc_id that points back to *raw*
    keep_r = [i for i, t in enumerate(reddit_texts_all) if t and str(t).strip()]
    keep_p = [i for i, t in enumerate(poly_texts_all) if t and str(t).strip()]

    reddit_texts = [reddit_texts_all[i] for i in keep_r]
    poly_texts   = [poly_texts_all[i]   for i in keep_p]

    print(f"Clean texts → reddit={len(reddit_texts)}  poly={len(poly_texts)}")

    # ---------- fit topic models ----------
    print("Fitting BERTopic (Reddit)…")
    r_topics, _, r_model, r_st_model = embed_and_fit(reddit_texts, ngram_range=(1,3), min_df=2, max_df=0.9)
    print("Fitting BERTopic (Polymarket)…")
    p_topics, _, p_model, p_st_model = embed_and_fit(poly_texts, ngram_range=(1,3), min_df=2, max_df=0.9)

    # ---------- doc_info + embeddings for centroids ----------
    r_doc = r_model.get_document_info(reddit_texts)
    p_doc = p_model.get_document_info(poly_texts)

    # carry original raw indices as doc_id (critical for ID-based re-enrichment)
    r_doc["doc_id"] = keep_r
    p_doc["doc_id"] = keep_p

    # embeddings (use model backend if available, else ST.encode)
    def _encode(backend_or_st, texts, fallback_st=None):
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

    r_emb = _encode(getattr(r_model, "embedding_model", r_st_model) or r_st_model, reddit_texts, fallback_st=r_st_model)
    p_emb = _encode(getattr(p_model, "embedding_model", p_st_model) or p_st_model, poly_texts, fallback_st=p_st_model)

    r_doc["embedding"] = [e.tolist() for e in r_emb]
    p_doc["embedding"] = [e.tolist() for e in p_emb]

    # ---------- centroids ----------
    r_centroids = compute_topic_centroids(r_doc, emb_col="embedding", topic_col="Topic", label_col="Name")
    p_centroids = compute_topic_centroids(p_doc, emb_col="embedding", topic_col="Topic", label_col="Name")
    # normalize to expected schema
    if "topic_id" in r_centroids.columns: r_centroids = r_centroids.rename(columns={"topic_id": "Topic"})
    if "label"   in r_centroids.columns: r_centroids = r_centroids.rename(columns={"label":   "Name"})
    if "topic_id" in p_centroids.columns: p_centroids = p_centroids.rename(columns={"topic_id": "Topic"})
    if "label"   in p_centroids.columns: p_centroids = p_centroids.rename(columns={"label":   "Name"})

    # ---------- alignment ----------
    THRESHOLD = float(os.environ.get("ALIGN_THRESHOLD", "0.60"))
    alignment_df, sim_matrix = extract_semantically_overlapping_topics(r_centroids, p_centroids, threshold=THRESHOLD)

    # ---------- meta with features (adds doc_id 0..n-1) ----------
    reddit_df = make_reddit_meta(reddit_raw)
    poly_df   = make_polymarket_meta_with_features(poly_raw)

    # ---------- build aligned structure ----------
    aligned = build_exact_aligned_topics_with_dates_and_meta(
        reddit_centroids=r_centroids,
        polymarket_centroids=p_centroids,
        sim_matrix=sim_matrix,
        reddit_topic_model=r_model,
        polymarket_topic_model=p_model,
        reddit_doc_info=r_doc,
        poly_doc_info=p_doc,
        cleaned_reddit_texts=reddit_texts,
        cleaned_poly_texts=poly_texts,
        threshold=THRESHOLD,
        reddit_raw_df=reddit_df,
        polymarket_raw_df=poly_df,
        alignment_df=alignment_df,
        top_n_words=10,
    )

    # ---------- write output ----------
    def _json_default(o):
        import numpy as _np
        import pandas as _pd
        from datetime import datetime, date
        if isinstance(o, (_pd.Timestamp, datetime, date)):
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
        return str(o)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"aligned_full_{ts}.json"
    out_path.write_text(json.dumps(aligned, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    print(f"\n✅ Wrote {out_path}")
