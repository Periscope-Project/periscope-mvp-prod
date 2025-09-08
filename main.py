# main.py
from __future__ import annotations

import os
import json
from datetime import datetime, date
from glob import glob
from typing import Any

import numpy as np
import pandas as pd
import torch

import pipeline.topic_modelling as topic_modelling
from sources.polymarket import get_polymarket
from sources.reddit import get_reddit
from utils import data_helpers, enrich_data
from pipeline.lite_llm import summarize_topics


from sentence_transformers import SentenceTransformer

# -----------------------------
# Config
# -----------------------------
PRINT_DEVICE = True
SKIP_POLY_FETCH = True
RUN_LITELLM = True  # flip to False to skip this step
SAVE_DIR = os.environ.get("PERISCOPE_OUT_DIR", "public/files/")
THRESHOLD = float(os.environ.get("TOPIC_OVERLAP_THRESHOLD", "0.50"))
REDDIT_DAYS = int(os.environ.get("REDDIT_LOOKBACK_DAYS", "14"))
MINILM_MODEL = os.environ.get("MINILM_MODEL", "all-MiniLM-L6-v2")
BERT_MODEL = os.environ.get("BERT_MODEL", "bert-base-uncased")
MAX_LEN = int(os.environ.get("BERT_MAX_LEN", "128"))
INCLUDE_TOP_COMMENTS = int(os.environ.get("REDDIT_TOP_COMMENTS", "2"))
LOCAL_POLY_FALLBACK = os.environ.get("POLYMARKET_FALLBACK_PATH", "").strip()

today = datetime.now().strftime("%d_%m_%y")

# -----------------------------
# Helpers
# -----------------------------
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

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
if PRINT_DEVICE:
    print("CUDA available:", torch.cuda.is_available())

# -----------------------------
# 1) Reddit ingest + normalize
# -----------------------------
print("\n[Reddit] Loading…")
reddit_df = data_helpers.load_reddit_range(root="public/files/source_data/reddit",_glob="reddit_daily_all*.ndjson",days=REDDIT_DAYS)
reddit_df = data_helpers.clean_null_df(reddit_df, "text")
reddit_df = enrich_data.ensure_created_datetime(reddit_df, tz="Europe/London")
reddit_records = reddit_df.to_dict(orient="records")

print("[Reddit] Tokenizing…")
tokenized_reddit, cleaned_reddit_texts = topic_modelling.build_bert_corpus_from_reddit(
    reddit_records,
    include_top_comments=INCLUDE_TOP_COMMENTS,
    model_name=BERT_MODEL,
    max_length=MAX_LEN,
)
print(f"[Reddit] Clean texts: {len(cleaned_reddit_texts)}")

# -----------------------------
# 2) Polymarket ingest + normalize (simple)
# -----------------------------
print("\n[Polymarket] Loading…")
polymarket_json = None

# 1) Try live (and save a raw copy if it works)
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

# 2) Fallback: newest local .jsonl
if not polymarket_json:
    candidates = sorted(glob("public/files/source_data/polymarket/*.jsonl"))
    if not candidates:
        raise RuntimeError("No Polymarket data available (no live + no local .jsonl).")
    latest = candidates[-1]
    polymarket_json = data_helpers.load_json_or_jsonl(latest)
    print(f"[Polymarket] Loaded latest local file: {latest}")

# 3) Build meta/features and keep only rows with a non-empty 'question'
poly_meta_df = enrich_data.make_polymarket_meta_with_features(polymarket_json)
poly_meta_df = data_helpers.clean_null_df(poly_meta_df, "question")

# Optional debug save
poly_meta_path = out_path(f"polymarket_meta_{today}.ndjson")
poly_meta_df.to_json(poly_meta_path, orient="records", lines=True, force_ascii=False)
print(f"[Save] {poly_meta_path}")

# 4) Tokenize (pass records, not the DataFrame)
poly_records = poly_meta_df.to_dict(orient="records")
print("[Polymarket] Tokenizing…")
tokenized_poly, cleaned_poly_texts = topic_modelling.build_bert_corpus_from_polymarket_snapshots(
    poly_records,
    include_title=True,
    include_description=True,
    include_all_questions=True,
    model_name=BERT_MODEL,
    max_length=MAX_LEN,
)
print(f"[Polymarket] Clean texts: {len(cleaned_poly_texts)}")


# -----------------------------
# 3) Fit BERTopic models
# -----------------------------
print("\n[Topics] Fitting BERTopic (Reddit)…")
reddit_topics, reddit_probs, reddit_topic_model, reddit_embedding_model = topic_modelling.embed_and_fit(
    cleaned_reddit_texts,
    stop_words="english",
    ngram_range=(1, 3),
    min_df=0.01,
    max_df=0.9,
)


print("[Topics] Fitting BERTopic (Polymarket)…")
poly_topics, poly_probs, poly_topic_model, poly_embedding_model = topic_modelling.embed_and_fit(
    cleaned_poly_texts,
    stop_words="english",
    ngram_range=(1, 3),
    min_df=0.01,
    max_df=0.9,
)

print(f"[Topics] Reddit topics: {len(reddit_topics)} | Polymarket topics: {len(poly_topics)}")

# -----------------------------
# 4) Document info + MiniLM embeddings
# -----------------------------
print("\n[Embeddings] Document info + MiniLM…")
reddit_doc_info = reddit_topic_model.get_document_info(cleaned_reddit_texts)
poly_doc_info   = poly_topic_model.get_document_info(cleaned_poly_texts)

# Save doc_info early (without embeddings)
reddit_doc_info_path = out_path(f"reddit_doc_info_{today}.ndjson")
poly_doc_info_path   = out_path(f"poly_doc_info_{today}.ndjson")
reddit_doc_info.to_json(reddit_doc_info_path, orient="records", lines=True, force_ascii=False)
poly_doc_info.to_json(poly_doc_info_path, orient="records", lines=True, force_ascii=False)
print(f"[Save] {reddit_doc_info_path}")
print(f"[Save] {poly_doc_info_path}")

# Encode with MiniLM
st_model = SentenceTransformer(MINILM_MODEL, device=device)
enc_reddit = st_model.encode(cleaned_reddit_texts, convert_to_numpy=True, show_progress_bar=True)
enc_poly   = st_model.encode(cleaned_poly_texts,   convert_to_numpy=True, show_progress_bar=True)

reddit_doc_info["embedding"] = [e.tolist() for e in enc_reddit]
poly_doc_info["embedding"]   = [e.tolist() for e in enc_poly]

# -----------------------------
# 5) Topic centroids
# -----------------------------
print("\n[Centroids] Computing…")
reddit_centroids = topic_modelling.compute_topic_centroids(
    reddit_doc_info, emb_col="embedding", topic_col="Topic", label_col="Name"
)
poly_centroids   = topic_modelling.compute_topic_centroids(
    poly_doc_info,   emb_col="embedding", topic_col="Topic", label_col="Name"
)

# Normalize column names if function returned topic_id/label
if "topic_id" in reddit_centroids.columns:
    reddit_centroids = reddit_centroids.rename(columns={"topic_id": "Topic"})
if "label" in reddit_centroids.columns:
    reddit_centroids = reddit_centroids.rename(columns={"label": "Name"})
if "topic_id" in poly_centroids.columns:
    poly_centroids = poly_centroids.rename(columns={"topic_id": "Topic"})
if "label" in poly_centroids.columns:
    poly_centroids = poly_centroids.rename(columns={"label": "Name"})

# -----------------------------
# 6) Overlap/alignment
# -----------------------------
print("\n[Align] Extracting overlapping topics…")
alignment_df, sim_matrix = topic_modelling.extract_semantically_overlapping_topics(
    reddit_centroids, poly_centroids, threshold=THRESHOLD
)

alignment_out_path = out_path(f"topic_overlap_alignment_{today}.csv")
alignment_df.to_csv(alignment_out_path, index=False)
print(f"[Save] {alignment_out_path} (threshold >= {THRESHOLD})")

# -----------------------------
# 7) Build aligned JSON (rich)
# -----------------------------
print("\n[Build] Full aligned JSON…")
aligned_full = enrich_data.build_exact_aligned_topics_with_dates_and_meta(
    reddit_centroids=reddit_centroids,
    polymarket_centroids=poly_centroids,
    sim_matrix=sim_matrix,
    reddit_topic_model=reddit_topic_model,
    polymarket_topic_model=poly_topic_model,
    reddit_doc_info=reddit_doc_info,
    poly_doc_info=poly_doc_info,
    cleaned_reddit_texts=cleaned_reddit_texts,
    cleaned_poly_texts=cleaned_poly_texts,
    threshold=THRESHOLD,
    reddit_raw_df=reddit_df,
    polymarket_raw_df=poly_meta_df,
    alignment_df=alignment_df,
    top_n_words=10,
)

aligned_out_path = out_path(f"aligned_topics_full_{today}.json")
with open(aligned_out_path, "w", encoding="utf-8") as f:
    json.dump(aligned_full, f, indent=2, ensure_ascii=False, default=json_default)
print(f"[Save] {aligned_out_path}")

if RUN_LITELLM:
    try:
        litellm_result = summarize_topics(
            aligned_full,                 # pass the in-memory TopicSignal list
            save_dir=SAVE_DIR,            # defaults to "public/files"
            # Optional overrides (otherwise uses env/module defaults):
            model_name=os.getenv("LITELLM_MODEL", "vertex_ai/gemini-2.5-pro"),
            max_chars=int(os.getenv("LITELLM_MAX_CHARS", "300000")),
            sleep_sec=float(os.getenv("LITELLM_SLEEP_SEC", "0")),
            # You can pass API creds explicitly; otherwise they’re read from env:
            api_key=os.getenv("LITELLM_API_KEY"),
            base_url=os.getenv("LITELLM_LOCATION"),
        )

        print(f"[LiteLLM] Raw responses: {litellm_result['raw_txt']}")
        print(f"[LiteLLM] Combined JSON: {litellm_result['combined_json']}")
    except Exception as e:
        # Don't crash the whole pipeline if post-processing fails
        import traceback
        print("[LiteLLM] Error while summarizing topics:", repr(e))
        print(traceback.format_exc())

print("\n✅ Pipeline complete.")
