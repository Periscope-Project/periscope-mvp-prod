# main.py
from __future__ import annotations

import os
import json
from datetime import datetime

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


#FIXME fix imports


import utils.data_helpers as data_helpers
import utils.get_polymarket as get_polymarket
import utils.topic_modelling as topic_modelling
import utils.enrich_data as enrich_data
from datetime import date
import numpy as np


print("CUDA available:", torch.cuda.is_available())

today = datetime.now().strftime("%d_%m_%y")

# -------- helpers --------
def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

# -------- Reddit --------
reddit_df = data_helpers.load_reddit_range(days=14)
reddit_df = data_helpers.clean_null_df(reddit_df, "text")
# Normalize time on RAW reddit (adds created_dt + created_iso)
reddit_df = enrich_data.ensure_created_datetime(reddit_df, tz="Europe/London")

# Records for tokenizer
reddit_records = reddit_df.to_dict(orient="records")

# Tokenize Reddit
tokenized_reddit, cleaned_reddit_texts = topic_modelling.build_bert_corpus_from_reddit(
    reddit_records,
    include_top_comments=2,
    model_name="bert-base-uncased",
    max_length=128,
)

print(f"Clean Reddit texts {len(cleaned_reddit_texts)}")

# -------- Polymarket --------
# try:
#     polymarket_json = get_polymarket.get_trending_gamma_events(limit=1000)  # API rate limits ~500
# except Exception as e:
#     print(f"Error fetching Polymarket data: {e}")
#     fallback_path = f"data/daily_polymarket/trending_polymarket_events_{today}.jsonl"
#     with open(fallback_path, "r", encoding="utf-8") as f:
#         polymarket_json = json.loads(f.read())

# new polymarket events with tags hardcoded FIXME
with open("data/daily_polymarket/all_unique_events_05_08_24.json", "r", encoding="utf-8") as f:
    polymarket_json = json.load(f)

polymarket_df = pd.DataFrame(polymarket_json)
polymarket_df = data_helpers.clean_null_df(polymarket_df, "title")
polymarket_df = enrich_data.ensure_created_datetime(polymarket_df, tz="Europe/London")

# Tokenize Polymarket titles as text
polymarket_records = polymarket_df.to_dict(orient="records")

tokenized_poly, cleaned_poly_texts = topic_modelling.build_bert_corpus_from_polymarket(
    polymarket_records,
    include_title=True,
    include_description=True,
    include_all_questions=True,
    model_name="bert-base-uncased",
    max_length=128,
)

print(f"Clean poly texts {len(cleaned_poly_texts)}")
# now len(cleaned_poly_texts) == len(polymarket_records) == 1278

# -------- Fit BERTopic --------
reddit_topics, reddit_probs, reddit_topic_model = topic_modelling.embed_and_fit(
    cleaned_reddit_texts,
    stop_words="english",
    ngram_range=(1, 3),
    min_df=0.01,
    max_df=0.9,
)
poly_topics, poly_probs, poly_topic_model = topic_modelling.embed_and_fit(
    cleaned_poly_texts,
    stop_words="english",
    ngram_range=(1, 3),
    min_df=0.01,
    max_df=0.9,
)

print(f"Number of reddit topics: {len(reddit_topics)}")
print(f"Number of polymarket topics: {len(poly_topics)}")

# -------- Document info + embeddings --------
reddit_doc_info = reddit_topic_model.get_document_info(cleaned_reddit_texts)
poly_doc_info   = poly_topic_model.get_document_info(cleaned_poly_texts)

# Save raw doc_info (optional)
ensure_dir("data/outputs/reddit_doc_info.ndjson")
reddit_doc_info.to_json("data/outputs/reddit_doc_info.ndjson",
                        orient="records", lines=True, force_ascii=False)
ensure_dir("data/outputs/poly_doc_info.ndjson")
poly_doc_info.to_json("data/outputs/poly_doc_info.ndjson",
                      orient="records", lines=True, force_ascii=False)

# Encode with MiniLM
device = "cuda" if torch.cuda.is_available() else "cpu"
st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
enc_reddit = st_model.encode(cleaned_reddit_texts, convert_to_numpy=True, show_progress_bar=True)
enc_poly   = st_model.encode(cleaned_poly_texts,   convert_to_numpy=True, show_progress_bar=True)

reddit_doc_info["embedding"] = [e.tolist() for e in enc_reddit]
poly_doc_info["embedding"]   = [e.tolist() for e in enc_poly]

# -------- Centroids --------
reddit_centroids = topic_modelling.compute_topic_centroids(
    reddit_doc_info, emb_col="embedding", topic_col="Topic", label_col="Name"
)
poly_centroids   = topic_modelling.compute_topic_centroids(
    poly_doc_info,   emb_col="embedding", topic_col="Topic", label_col="Name"
)
#TODO keep names the same Topic, Name, embedding etc

# IMPORTANT: your topic_modelling.compute_topic_centroids() returns columns
#   topic_id, label, embedding
# Rename them to what enrich_data expects: Topic, Name, embedding
if "topic_id" in reddit_centroids.columns:
    reddit_centroids = reddit_centroids.rename(columns={"topic_id": "Topic"})
if "label" in reddit_centroids.columns:
    reddit_centroids = reddit_centroids.rename(columns={"label": "Name"})

if "topic_id" in poly_centroids.columns:
    poly_centroids = poly_centroids.rename(columns={"topic_id": "Topic"})
if "label" in poly_centroids.columns:
    poly_centroids = poly_centroids.rename(columns={"label": "Name"})


# -------- Topic overlap (use alignment_df explicitly) --------
THRESHOLD = 0.50
alignment_df, sim_matrix = topic_modelling.extract_semantically_overlapping_topics(
    reddit_centroids, poly_centroids, threshold=THRESHOLD
)

# Save alignment pairs
ensure_dir("data/outputs/topic_overlap_alignment.csv")
alignment_out_path = "data/outputs/topic_overlap_alignment.csv"
alignment_df.to_csv(alignment_out_path, index=False)
print(f"Saved alignment (threshold >= {THRESHOLD}) to {alignment_out_path}")

# -------- Build aligned structure --------
aligned_full = enrich_data.build_exact_aligned_topics_with_dates_and_meta(
    reddit_centroids=reddit_centroids,
    polymarket_centroids=poly_centroids,
    sim_matrix=sim_matrix,                 # still pass; will be ignored since we pass alignment_df
    reddit_topic_model=reddit_topic_model,
    polymarket_topic_model=poly_topic_model,
    reddit_doc_info=reddit_doc_info,
    poly_doc_info=poly_doc_info,
    cleaned_reddit_texts=cleaned_reddit_texts,
    cleaned_poly_texts=cleaned_poly_texts,
    threshold=THRESHOLD,
    reddit_raw_df=reddit_df,               # re-enrich with ALL Reddit fields
    polymarket_raw_df=polymarket_df,       # re-enrich with ALL Polymarket fields
    alignment_df=alignment_df,             # <-- KEY: use the explicit pairs; avoids sim_matrix DataFrame requirement
    top_n_words=10,
)

def json_default(o):
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
    # last resort
    return str(o)


# -------- Write output JSON --------
ensure_dir(f"data/outputs/aligned_topics_full_{today}.json")
out_path = f"data/outputs/aligned_topics_full_{today}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(aligned_full, f, indent=2, ensure_ascii=False, default=json_default)

print(f"Wrote aligned full JSON to {out_path}")

