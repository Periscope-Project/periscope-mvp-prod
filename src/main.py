"""
Periscope Pipeline (Reddit + Polymarket → Topics → Alignment → (optional) LLM → MySQL)

Run modes:
• Single run (default):     python -m src.main
• Daily schedule in-code:   RUN_SCHEDULE=1 (requires APScheduler)
• API will also call run_pipeline() via POST /run-pipeline (locked with FileLock)

Key env flags (with defaults):
  PRINT_DEVICE=1
  SKIP_POLY_FETCH=1
  SKIP_GET_REDDIT=1
  RUN_LITELLM=0
  PERISCOPE_OUT_DIR=public/files/
  TOPIC_OVERLAP_THRESHOLD=0.50
  REDDIT_LOOKBACK_DAYS=14
  MINILM_MODEL=all-MiniLM-L6-v2
  BERT_MODEL=bert-base-uncased
  BERT_MAX_LEN=128
  REDDIT_TOP_COMMENTS=2
  POLYMARKET_FALLBACK_PATH=""
  PIPELINE_LOCK_FILE=/tmp/periscope_pipeline.lock

Scheduler env (only if RUN_SCHEDULE=1):
  SCHEDULE_TZ=Europe/London
  SCHEDULE_HOUR=2
  SCHEDULE_MINUTE=0
"""

from __future__ import annotations

import os
import json
from glob import glob
from datetime import date, datetime
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Local modules
import pipeline.topic_modelling as topic_modelling
from sources.polymarket import get_polymarket
from sources.reddit import get_reddit
from utils import data_helpers, enrich_data
from pipeline.lite_llm import summarize_topics
from database_utils.data_to_sql import load_trends

# ───────────────────────────────────────────────────────────────────────────────
# ENV / CONFIG
# ───────────────────────────────────────────────────────────────────────────────
def env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", "no", ""}

PRINT_DEVICE         = env_bool("PRINT_DEVICE", "1")
SKIP_POLY_FETCH      = env_bool("SKIP_POLY_FETCH", "0")
SKIP_GET_REDDIT      = env_bool("SKIP_GET_REDDIT", "0")
RUN_LITELLM          = env_bool("RUN_LITELLM", "0")

SAVE_DIR             = os.getenv("PERISCOPE_OUT_DIR", "public/files/")
THRESHOLD            = float(os.getenv("TOPIC_OVERLAP_THRESHOLD", "0.50"))
REDDIT_DAYS          = int(os.getenv("REDDIT_LOOKBACK_DAYS", "14"))
MINILM_MODEL         = os.getenv("MINILM_MODEL", "all-MiniLM-L6-v2")
BERT_MODEL           = os.getenv("BERT_MODEL", "bert-base-uncased")
MAX_LEN              = int(os.getenv("BERT_MAX_LEN", "128"))
INCLUDE_TOP_COMMENTS = int(os.getenv("REDDIT_TOP_COMMENTS", "2"))
LOCAL_POLY_FALLBACK  = os.getenv("POLYMARKET_FALLBACK_PATH", "").strip()

RUN_SCHEDULE         = env_bool("RUN_SCHEDULE", "0")
SCHEDULE_TZ          = os.getenv("SCHEDULE_TZ", "Europe/London")
SCHEDULE_HOUR        = int(os.getenv("SCHEDULE_HOUR", "2"))
SCHEDULE_MINUTE      = int(os.getenv("SCHEDULE_MINUTE", "0"))

LOCK_PATH            = os.getenv("PIPELINE_LOCK_FILE", "public/files/tmp/periscope_pipeline.lock")

# ───────────────────────────────────────────────────────────────────────────────
# UTILS
# ───────────────────────────────────────────────────────────────────────────────
def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def out_path(*parts: str) -> str:
    path = os.path.join(SAVE_DIR, *parts)
    ensure_dir(path if path.endswith("/") else os.path.dirname(path))
    return path

def json_default(o: Any):
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    if o is pd.NaT:
        return None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, set):
        return list(o)
    return str(o)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
if PRINT_DEVICE:
    print("CUDA available:", torch.cuda.is_available())

# ───────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ───────────────────────────────────────────────────────────────────────────────
def _load_reddit(days: int) -> Tuple[pd.DataFrame, list]:
    """
    Loads, cleans, and returns (df, cleaned_texts) for Reddit.
    """
    print("\n[Reddit] Loading…")
    df = data_helpers.load_reddit_range(
        root="public/files/source_data/reddit",
        _glob="reddit_daily_all*.ndjson",
        days=days,
    )
    if df is None or len(df) == 0:
        print("[Reddit] No files matched. Returning empty.")
        return pd.DataFrame(), []

    df = data_helpers.clean_null_df(df, "text")
    df = enrich_data.ensure_created_datetime(df, tz="Europe/London")
    records = df.to_dict(orient="records")

    print("[Reddit] Tokenizing…")
    tokenized, cleaned = topic_modelling.build_bert_corpus_from_reddit(
        records,
        include_top_comments=INCLUDE_TOP_COMMENTS,
        model_name=BERT_MODEL,
        max_length=MAX_LEN,
    )
    print(f"[Reddit] Clean texts: {len(cleaned)}")
    return df, cleaned

def _load_polymarket() -> Tuple[pd.DataFrame, list]:
    """
    Loads live or fallback Polymarket data, builds features, returns (df, cleaned_texts).
    """
    print("\n[Polymarket] Loading…")
    polymarket_json = None

    # Try live fetch if enabled
    if not SKIP_POLY_FETCH and get_polymarket is not None:
        try:
            polymarket_json = get_polymarket.get_polymarket_all()
            print(f"[Polymarket] Live events fetched: {len(polymarket_json)}")
            ts = datetime.now()
            live_path = out_path(f"source_data/polymarket/polymarket_live_{ts:%Y-%m-%d_%H%M%S}.jsonl")
            with open(live_path, "w", encoding="utf-8") as f:
                for rec in (polymarket_json or []):
                    f.write(json.dumps(rec, ensure_ascii=False, default=json_default) + "\n")
            print(f"[Save] {live_path}")
        except Exception as e:
            print(f"[Polymarket] Live fetch failed → {e}")
            polymarket_json = None

    # Fallback: explicit env file first, then newest local jsonl
    if not polymarket_json and LOCAL_POLY_FALLBACK and os.path.exists(LOCAL_POLY_FALLBACK):
        polymarket_json = data_helpers.load_json_or_jsonl(LOCAL_POLY_FALLBACK)
        print(f"[Polymarket] Loaded fallback file: {LOCAL_POLY_FALLBACK}")

    if not polymarket_json:
        candidates = sorted(glob("public/files/source_data/polymarket/*.jsonl"))
        if not candidates:
            print("[Polymarket] No data found (no live, no fallback, no local). Returning empty.")
            return pd.DataFrame(), []
        latest = candidates[-1]
        polymarket_json = data_helpers.load_json_or_jsonl(latest)
        print(f"[Polymarket] Loaded latest local file: {latest}")

    # Build features
    meta_df = enrich_data.make_polymarket_meta_with_features(polymarket_json)
    meta_df = data_helpers.clean_null_df(meta_df, "question")

    # Debug save
    today = datetime.now().strftime("%d_%m_%y")
    meta_path = out_path(f"polymarket_meta_{today}.ndjson")
    meta_df.to_json(meta_path, orient="records", lines=True, force_ascii=False)
    print(f"[Save] {meta_path}")

    # Tokenize
    print("[Polymarket] Tokenizing…")
    records = meta_df.to_dict(orient="records")
    tokenized, cleaned = topic_modelling.build_bert_corpus_from_polymarket_snapshots(
        records,
        include_title=True,
        include_description=True,
        include_all_questions=True,
        model_name=BERT_MODEL,
        max_length=MAX_LEN,
    )
    print(f"[Polymarket] Clean texts: {len(cleaned)}")
    return meta_df, cleaned

def run_pipeline() -> None:
    """
    Full end-to-end run. Produces artifacts in PERISCOPE_OUT_DIR and (optionally) writes to MySQL.
    """
    today = datetime.now().strftime("%d_%m_%y")
    print(f"\n=== Periscope pipeline run @ {datetime.utcnow().isoformat()}Z ===")

    # 1) (Optional) fetch Reddit
    if not SKIP_GET_REDDIT:
        print("\n[Reddit] Fetching…")
        reddit_path = get_reddit.fetch_reddit_data(
            out_dir="public/files/source_data/reddit",
            days=REDDIT_DAYS,
            include_top_comments=INCLUDE_TOP_COMMENTS,
            verbose=True,
        )
        print(f"[Reddit] Saved to: {reddit_path}")

    # Load sources
    reddit_df, cleaned_reddit = _load_reddit(REDDIT_DAYS)
    poly_df, cleaned_poly = _load_polymarket()

    has_reddit = len(cleaned_reddit) > 0
    has_poly   = len(cleaned_poly) > 0

    if not has_reddit and not has_poly:
        raise RuntimeError("No Reddit or Polymarket data available. Provide source files or disable skips.")

    # 2) Fit BERTopic for available sources
    if has_reddit:
        print("\n[Topics] Fitting BERTopic (Reddit)…")
        _, _, reddit_topic_model, _ = topic_modelling.embed_and_fit(
            cleaned_reddit,
            stop_words="english",
            ngram_range=(1, 3),
            min_df=0.01,
            max_df=0.9,
            device=device,
        )
        reddit_doc_info = reddit_topic_model.get_document_info(cleaned_reddit)
        reddit_doc_info_path = out_path(f"reddit_doc_info_{today}.ndjson")
        reddit_doc_info.to_json(reddit_doc_info_path, orient="records", lines=True, force_ascii=False)
        print(f"[Save] {reddit_doc_info_path}")
    else:
        reddit_topic_model, reddit_doc_info = None, pd.DataFrame()

    if has_poly:
        print("\n[Topics] Fitting BERTopic (Polymarket)…")
        _, _, poly_topic_model, _ = topic_modelling.embed_and_fit(
            cleaned_poly,
            stop_words="english",
            ngram_range=(1, 3),
            min_df=0.01,
            max_df=0.9,
            device=device,
        )
        poly_doc_info = poly_topic_model.get_document_info(cleaned_poly)
        poly_doc_info_path = out_path(f"poly_doc_info_{today}.ndjson")
        poly_doc_info.to_json(poly_doc_info_path, orient="records", lines=True, force_ascii=False)
        print(f"[Save] {poly_doc_info_path}")
    else:
        poly_topic_model, poly_doc_info = None, pd.DataFrame()

    # 3) MiniLM embeddings (only for existing sources)
    if has_reddit or has_poly:
        print("\n[Embeddings] SentenceTransformer (MiniLM)…")
        st_model = SentenceTransformer(MINILM_MODEL, device=device)

        if has_reddit:
            enc_reddit = st_model.encode(cleaned_reddit, convert_to_numpy=True, show_progress_bar=True)
            reddit_doc_info["embedding"] = [e.tolist() for e in enc_reddit]

        if has_poly:
            enc_poly = st_model.encode(cleaned_poly, convert_to_numpy=True, show_progress_bar=True)
            poly_doc_info["embedding"] = [e.tolist() for e in enc_poly]

    # 4) Topic centroids & optional alignment
    print("\n[Centroids] Computing…")
    reddit_centroids = None
    poly_centroids = None

    if has_reddit:
        reddit_centroids = topic_modelling.compute_topic_centroids(
            reddit_doc_info, emb_col="embedding", topic_col="Topic", label_col="Name"
        )
        reddit_centroids = reddit_centroids.rename(columns={"topic_id": "Topic", "label": "Name"}, errors="ignore")

    if has_poly:
        poly_centroids = topic_modelling.compute_topic_centroids(
            poly_doc_info, emb_col="embedding", topic_col="Topic", label_col="Name"
        )
        poly_centroids = poly_centroids.rename(columns={"topic_id": "Topic", "label": "Name"}, errors="ignore")

    # 5) Build outputs
    if has_reddit and has_poly:
        print("\n[Align] Extracting overlapping topics…")
        alignment_df, sim_matrix = topic_modelling.extract_semantically_overlapping_topics(
            reddit_centroids, poly_centroids, threshold=THRESHOLD
        )
        alignment_out_path = out_path(f"topic_overlap_alignment_{today}.csv")
        alignment_df.to_csv(alignment_out_path, index=False)
        print(f"[Save] {alignment_out_path} (threshold >= {THRESHOLD})")

        print("\n[Build] Full aligned JSON…")
        aligned_full = enrich_data.build_exact_aligned_topics_with_dates_and_meta(
            reddit_centroids=reddit_centroids,
            polymarket_centroids=poly_centroids,
            sim_matrix=sim_matrix,
            reddit_topic_model=reddit_topic_model,
            polymarket_topic_model=poly_topic_model,
            reddit_doc_info=reddit_doc_info,
            poly_doc_info=poly_doc_info,
            cleaned_reddit_texts=cleaned_reddit,
            cleaned_poly_texts=cleaned_poly,
            threshold=THRESHOLD,
            reddit_raw_df=reddit_df,
            polymarket_raw_df=poly_df,
            alignment_df=alignment_df,
            top_n_words=10,
        )
    elif has_reddit:
        print("\n[Build] Reddit-only JSON…")
        aligned_full = enrich_data.build_single_source_topics(
            topic_model=reddit_topic_model,
            doc_info=reddit_doc_info,
            raw_df=reddit_df,
            source="reddit",
            top_n_words=10,
        )
    else:
        print("\n[Build] Polymarket-only JSON…")
        aligned_full = enrich_data.build_single_source_topics(
            topic_model=poly_topic_model,
            doc_info=poly_doc_info,
            raw_df=poly_df,
            source="polymarket",
            top_n_words=10,
        )

    aligned_out_path = out_path(f"aligned_topics_full_{today}.json")
    with open(aligned_out_path, "w", encoding="utf-8") as f:
        json.dump(aligned_full, f, indent=2, ensure_ascii=False, default=json_default)
    print(f"[Save] {aligned_out_path}")

    # 6) Optional LLM summarization → SQL
    if RUN_LITELLM:
        try:
            litellm_result = summarize_topics(
                aligned_full,
                save_dir=SAVE_DIR,
                model_name=os.getenv("LITELLM_MODEL", "vertex_ai/gemini-2.5-pro"),
                max_chars=int(os.getenv("LITELLM_MAX_CHARS", "300000")),
                sleep_sec=float(os.getenv("LITELLM_SLEEP_SEC", "0")),
                api_key=os.getenv("LITELLM_API_KEY"),
                base_url=os.getenv("LITELLM_LOCATION"),
            )
            print(f"[LiteLLM] Raw responses: {litellm_result['raw_txt']}")
            print(f"[LiteLLM] Combined JSON: {litellm_result['combined_json']}")

            parsed = data_helpers._parse_trends(litellm_result)
            litellm_result_path = out_path(f"trend_briefs_litellm_{today}.json")
            with open(litellm_result_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False, default=json_default)
            print(f"[Save] {litellm_result_path}")

            if os.getenv("PUSH_TO_SQL", "1").lower() not in {"0", "false"}:
                rows = load_trends(litellm_result_path)
                print(f"[SQL] Trend briefs loaded into MySQL: {rows} rows")

        except Exception as e:
            import traceback
            print("[LiteLLM] Error while summarizing topics:", repr(e))
            print(traceback.format_exc())

    print("\n✅ Pipeline complete.")

# ───────────────────────────────────────────────────────────────────────────────
# OPTIONAL: Locked wrapper (used by API and/or scheduler)
# ───────────────────────────────────────────────────────────────────────────────
def run_pipeline_with_lock() -> None:
    from filelock import FileLock, Timeout  # local import to avoid hard dep if unused
    try:
        with FileLock(LOCK_PATH, timeout=1):
            run_pipeline()
    except Timeout:
        print("⏭ skipped: another run is in progress")

# ───────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if RUN_SCHEDULE:
        try:
            from apscheduler.schedulers.blocking import BlockingScheduler  # type: ignore
            from apscheduler.triggers.cron import CronTrigger  # type: ignore
        except ImportError:
            raise SystemExit(
                "RUN_SCHEDULE=1 requires APScheduler. Install:\n  pip install apscheduler pytz\n"
            )

        print(
            f"⏰ Scheduling daily run at {SCHEDULE_HOUR:02d}:{SCHEDULE_MINUTE:02d} in {SCHEDULE_TZ} "
            f"(set RUN_SCHEDULE=0 to run once)."
        )
        scheduler = BlockingScheduler(timezone=SCHEDULE_TZ)
        scheduler.add_job(run_pipeline_with_lock, trigger=CronTrigger(hour=SCHEDULE_HOUR, minute=SCHEDULE_MINUTE),
                          name="periscope_daily_run", max_instances=1, coalesce=True, misfire_grace_time=3600)
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            print("Scheduler stopped.")
    else:
        run_pipeline_with_lock()
