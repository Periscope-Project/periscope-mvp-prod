import pandas as pd
import numpy as np
import re
import html

from pathlib import Path
import os, sys
from pathlib import Path
import nltk

# Prefer env → vendor → venv → defaults
VENDOR_NLTK = Path(__file__).resolve().parents[2] / "vendor" / "nltk_data"
CANDIDATE_DIRS = [
    os.getenv("NLTK_DATA"),
    str(VENDOR_NLTK),
    str(Path(sys.prefix) / "nltk_data"),  # venv/local site
]
# Prepend in order (keep any existing)
for p in [d for d in CANDIDATE_DIRS if d]:
    if p not in nltk.data.path:
        nltk.data.path.insert(0, p)

REQUIRED = [
    ("corpora/stopwords", "stopwords"),
    ("tokenizers/punkt", "punkt"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]

def _nltk_has(resource_key: str) -> bool:
    try:
        nltk.data.find(resource_key)
        return True
    except LookupError:
        return False

_missing = [name for key, name in REQUIRED if not _nltk_has(key)]

# Zero-download production: error clearly if vendor missing.
# If you *do* want automatic download fallback on fresh EC2, set ALLOW_NLTK_DOWNLOAD=1 in env.
if _missing:
    if os.getenv("ALLOW_NLTK_DOWNLOAD") == "1":
        for _, name in REQUIRED:
            nltk.download(name, download_dir=str(VENDOR_NLTK), quiet=True)
        _missing = [name for key, name in REQUIRED if not _nltk_has(key)]
    if _missing:
        raise RuntimeError(
            f"Missing NLTK data: {_missing}. "
            f"Ensure repo contains vendor/nltk_data or set NLTK_DATA to that path."
        )


from typing import Any, Dict, List, Tuple
import json

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from transformers import AutoTokenizer
from gensim.models.phrases import Phrases, Phraser
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import seaborn as sns

import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))





#TODO no limit for topic modelling

# --- flexible column detection helpers ---

def _find_topic_id_col(df: pd.DataFrame) -> str:
    for c in ("topic_id", "Topic", "topic", "cluster", "id"):
        if c in df.columns:
            return c
    raise KeyError("No topic-id column found (tried topic_id/Topic/topic/cluster/id).")

def _find_label_col(df: pd.DataFrame) -> str | None:
    for c in ("label", "Name", "name"):
        if c in df.columns:
            return c
    return None


# --------- ensure NLTK data ---------


def _ensure_nltk():
    for key in ["corpora/stopwords","tokenizers/punkt","corpora/wordnet","corpora/omw-1.4"]:
        nltk.data.find(key)


_ensure_nltk()

# --------- regex & cleaning ---------
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
QUOTE_LINE_RE = re.compile(r"^\s*>\s.*$", re.MULTILINE)
SUB_USER_RE = re.compile(r"(?:^|\s)(?:r|u)/[A-Za-z0-9_]+")
MULTISPACE_RE = re.compile(r"\s+")
HTML_TAG_RE = re.compile(r"<[^>]+>")

BOILERPLATE_PATTERNS = [
    r"^I am a bot.*$",
    r"^This (?:message|action) was performed automatically.*$",
    r"^Please contact the moderators.*$",
    r"^Welcome to r/.*$",
    r"^I am a bot, and this action was performed automatically\. Please contact the moderators of this subreddit\.$",
]
BOILERPLATE_RES = [re.compile(p, re.IGNORECASE | re.MULTILINE)
                   for p in BOILERPLATE_PATTERNS]

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = nltk.stem.WordNetLemmatizer()
TOKENIZER = ToktokTokenizer()


def _drop_boilerplate(text: str) -> str:
    for rx in BOILERPLATE_RES:
        text = rx.sub("", text)
    return text


def clean_text_for_bert(
    rec: Dict[str, Any],
    include_title: bool = True,
    include_text: bool = True,
    include_top_comments: int = 0,
    lower: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    parts: List[str] = []
    if include_title and rec.get("title"):
        parts.append(str(rec["title"]))
    if include_text and rec.get("text"):
        parts.append(str(rec["text"]))
    if include_top_comments and rec.get("top_comments"):
        parts.extend(rec.get("top_comments")[:include_top_comments])
    if not parts:
        return ""
    text = "\n".join(parts)
    text = _drop_boilerplate(text)
    text = html.unescape(text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = URL_RE.sub("", text)
    text = SUB_USER_RE.sub(" ", text)
    text = QUOTE_LINE_RE.sub("", text)
    text = HTML_TAG_RE.sub(" ", text)
    if lower:
        text = text.lower()
    if collapse_whitespace:
        text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def tokenize_for_bert(
    texts: List[str],
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    padding: str = "max_length",
    truncation: bool = True,
    return_tensors: str = "pt",
) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    encoded = tokenizer(
        texts,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=return_tensors,
        return_attention_mask=True,
    )
    return encoded

#======== Build corpus from Reddit posts ========
def build_bert_corpus_from_reddit(
    records: List[Dict[str, Any]],
    include_title: bool = True,
    include_text: bool = True,
    include_top_comments: int = 0,
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    phrase_min_count: int = 5,
    phrase_threshold: int = 10,
) -> Tuple[Dict[str, Any], List[str]]:
    # Clean
    cleaned = [
        clean_text_for_bert(
            rec,
            include_title=include_title,
            include_text=include_text,
            include_top_comments=include_top_comments,
        ) for rec in records
    ]
    # Phrase detection
    token_lists = [text.split() for text in cleaned]
    bigram = Phrases(token_lists, min_count=phrase_min_count,
                     threshold=phrase_threshold)
    phraser = Phraser(bigram)
    docs_with_phrases = [" ".join(phraser[d]) for d in token_lists]
    # Tokenize
    tokenized = tokenize_for_bert(
        docs_with_phrases, model_name=model_name, max_length=max_length)
    return tokenized, docs_with_phrases


#======== Build corpus from Polymarket snapshots ========

def _safe_outcomes(o) -> List[str]:
    """
    Outcomes can arrive as a JSON-encoded string like '["Yes","No"]'
    or as a Python list. Return a flat list of strings.
    """
    if o is None:
        return []
    if isinstance(o, list):
        return [str(x) for x in o if x is not None]
    if isinstance(o, str):
        try:
            parsed = json.loads(o)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if x is not None]
        except Exception:
            # fallback: split on commas / spaces if someone passed a raw string
            return [s.strip() for s in o.split(",") if s.strip()]
    return []

def _clean_blob(text: str) -> str:
    return clean_text_for_bert(
        {"title": "", "text": text, "top_comments": []},
        include_title=False,
        include_text=True,
        include_top_comments=0,
        lower=True,
        collapse_whitespace=True,
    )

def _build_snapshot_text(
    rec: Dict[str, Any],
    include_tag: bool,
    include_outcomes: bool,
    include_slug: bool,
) -> str:
    """
    Build one text blob from a single market_snapshot-style record.
    """
    parts = []
    # 1) core question
    q = (
        rec.get("market", {}) or {}
    ).get("question", "")
    if q:
        parts.append(str(q))

    # 2) optional small enrichments that help topic quality without leaking numeric fields
    if include_tag:
        tag_label = ((rec.get("tag") or {}).get("label")) or ""
        if tag_label:
            parts.append(str(tag_label))

    if include_outcomes:
        outcomes = _safe_outcomes((rec.get("market") or {}).get("outcomes"))
        if outcomes:
            # keep compact; joins to avoid noise
            parts.append(" / ".join(outcomes))

    if include_slug:
        slug = (rec.get("market") or {}).get("slug", "")
        if slug:
            # slugs often include salient tokens (team names, tickers)
            parts.append(slug.replace("-", " "))

    raw = " ".join(parts).strip()
    return _clean_blob(raw)

def _build_event_text(
    rec: Dict[str, Any],
    include_title: bool,
    include_description: bool,
    include_all_questions: bool,
) -> str:
    """
    Backward-compat: your old event schema with a `markets` list.
    """
    parts: List[str] = []
    if include_title and rec.get("title"):
        parts.append(str(rec["title"]))
    if include_description and rec.get("description"):
        parts.append(str(rec["description"]))
    if include_all_questions and rec.get("markets"):
        qs = [m.get("question", "") for m in rec["markets"]]
        parts.append(" ".join(qs))
    raw = "\n".join(parts)
    return _clean_blob(raw)



def _normalize_records(records: Any) -> List[Dict[str, Any]]:
    """Accept DataFrame, list[dict], dict; always return list[dict]."""
    if isinstance(records, list):
        return records
    if pd is not None and isinstance(records, pd.DataFrame):
        return records.to_dict(orient="records")
    if isinstance(records, dict):
        return [records]
    raise TypeError(f"Unsupported records type: {type(records)}")

def _maybe_json(val: Any) -> Any:
    """Parse JSON-in-strings like '["Yes","No"]'."""
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return val
    return val

def _coerce_dict(x: Any) -> Dict[str, Any]:
    """If CSV left dicts as strings, make them real dicts."""
    if isinstance(x, dict):
        return x
    if isinstance(x, str) and x.strip():
        s = x.strip()
        for candidate in (s, s.replace("'", '"')):
            try:
                obj = json.loads(candidate)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                continue
    return {}


def _build_event_text(rec: Dict[str, Any], include_title: bool, include_description: bool, include_all_questions: bool) -> str:
    # Legacy event-style fallback (kept for compatibility)
    parts = []
    if include_title and rec.get("title"):
        parts.append(str(rec["title"]).strip())
    if include_description and rec.get("description"):
        parts.append(str(rec["description"]).strip())
    if include_all_questions and isinstance(rec.get("markets"), list):
        qs = []
        for m in rec["markets"]:
            q = (m.get("question") or m.get("title") or "").strip()
            if q:
                qs.append(q)
        if qs:
            parts.append(" | ".join(qs[:6]))
    return " ".join(p for p in parts if p)

def build_bert_corpus_from_polymarket_snapshots(
    records: Any,  # <-- now accepts DataFrame directly
    *,
    # snapshot options:
    include_tag: bool = True,
    include_outcomes: bool = True,
    include_slug: bool = False,
    # legacy options (kept to avoid breaking callers):
    include_title: bool = True,
    include_description: bool = True,
    include_all_questions: bool = True,
    # modeling:
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    phrase_min_count: int = 5,
    phrase_threshold: int = 10,
) -> Tuple[Dict[str, Any], List[str]]:
    rows = _normalize_records(records)

    cleaned_texts: List[str] = []
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        market = _coerce_dict(rec.get("market"))
        has_snapshot = bool(rec.get("question") or market.get("question"))
        has_event = bool(rec.get("title") or rec.get("description") or isinstance(rec.get("markets"), list))

        if has_snapshot:
            txt = _build_snapshot_text(rec, include_tag, include_outcomes, include_slug)
        elif has_event:
            txt = _build_event_text(rec, include_title, include_description, include_all_questions)
        else:
            txt = str(rec.get("slug") or rec.get("market_id") or market.get("slug") or "").strip()

        cleaned_texts.append(txt or "")

    # light phrase detection → then BERT tokenize (same as your other builders)
    token_lists = [t.split() for t in cleaned_texts]
    bigram = Phrases(token_lists, min_count=phrase_min_count, threshold=phrase_threshold)
    phraser = Phraser(bigram)
    docs_with_phrases = [" ".join(phraser[tok]) for tok in token_lists]

    tokenized = tokenize_for_bert(
        docs_with_phrases,
        model_name=model_name,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return tokenized, docs_with_phrases
# --- END DROP-IN REPLACEMENT ---


def _auto_df(texts, min_df=10, max_df=0.5):
    # drop empties first
    texts = [t for t in texts if t and t.strip()]
    n = max(len(texts), 1)
    # make min_df scale with corpus
    min_df_eff = min(min_df, max(2, int(0.005 * n)))  # at least 2 docs or 0.5%
    max_df_eff = max_df
    # ensure consistency: min_df <= max_df * n
    max_allowed_min_df = max(1, int(max_df_eff * n))
    if min_df_eff > max_allowed_min_df:
        # relax max_df first, then min_df if still too tight
        max_df_eff = min(0.95, max(0.7, min_df_eff / n + 0.05))
        max_allowed_min_df = max(1, int(max_df_eff * n))
        if min_df_eff > max_allowed_min_df:
            min_df_eff = max_allowed_min_df
    return texts, min_df_eff, max_df_eff

def embed_and_fit(
    texts,
    stop_words="english",
    ngram_range=(1, 3),
    min_df=10,
    max_df=0.5,
    umap_params=None,
    hdbscan_params=None,
    device=None,
):
    texts, min_df_eff, max_df_eff = _auto_df(texts, min_df=min_df, max_df=max_df)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Use GPU for embeddings if available
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    vectorizer = CountVectorizer(
        stop_words=stop_words,
        ngram_range=ngram_range,
        min_df=min_df_eff,
        max_df=max_df_eff,
    )

    umap_model = UMAP(**(umap_params or {"n_neighbors": 15, "min_dist": 0.0}))
    hdbscan_model = HDBSCAN(**(hdbscan_params or {"min_cluster_size": 20}))

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        language="english",
        verbose=True,
    )

    # Fit (BERTopic will call embedding_model internally on the chosen device)
    try:
        topics, probs = topic_model.fit_transform(texts)
    except ValueError as e:
        if "max_df corresponds to < documents than min_df" in str(e):
            topic_model.vectorizer_model.set_params(min_df=1, max_df=0.95)
            topics, probs = topic_model.fit_transform(texts)
        else:
            raise

    # Optional: speed/fidelity tweak on Ampere+ GPUs
    if device == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    return topics, probs, topic_model, embedding_model



def compute_topic_centroids(
    doc_df: pd.DataFrame,
    emb_col: str = "embedding",
    topic_col: str = "Topic",
    label_col: str = "Name"
) -> pd.DataFrame:
    centroids = []
    for topic, group in doc_df.groupby(topic_col):
        if topic == -1:
            continue  # skip noise and stopwords
        embeddings = np.vstack(group[emb_col].values)
        centroid = embeddings.mean(axis=0)
        label = group[label_col].iloc[
            0] if label_col in group.columns else f"Topic {topic}"
        centroids.append(
            {"topic_id": topic, "label": label, "embedding": centroid})
    return pd.DataFrame(centroids)


def extract_semantically_overlapping_topics(
    reddit_emb: pd.DataFrame,
    polymarket_emb: pd.DataFrame,
    threshold: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute cosine similarities between topic centroid embeddings and
    return:
      - alignment_df: columns ['reddit_topic','polymarket_topic','similarity'] filtered by threshold,
        keeping all pairs above threshold (you can post-filter to top-1 if desired).
      - sim_matrix: 2D DataFrame (rows=reddit topic ids, cols=polymarket topic ids) of cosine sims.
    Column names for ids/labels are detected automatically.
    """
    # Detect columns
    r_id = _find_topic_id_col(reddit_emb)
    p_id = _find_topic_id_col(polymarket_emb)

    # Stack embeddings
    emb1 = np.vstack(reddit_emb["embedding"].values)
    emb2 = np.vstack(polymarket_emb["embedding"].values)

    # Cosine similarities
    S = cosine_similarity(emb1, emb2)

    # Build similarity matrix as DataFrame with proper ids
    r_ids = reddit_emb[r_id].astype(int).to_list()
    p_ids = polymarket_emb[p_id].astype(int).to_list()
    sim_matrix = pd.DataFrame(S, index=r_ids, columns=p_ids)

    # Threshold to long format
    r_idx, p_idx = np.where(S >= threshold)
    alignment_df = pd.DataFrame({
        "reddit_topic":   [r_ids[i] for i in r_idx],
        "polymarket_topic": [p_ids[j] for j in p_idx],
        "similarity":     S[r_idx, p_idx],
    }).sort_values("similarity", ascending=False, kind="mergesort").reset_index(drop=True)

    return alignment_df, sim_matrix


def get_topic_similarity_heatmap(
    reddit_topics_df: pd.DataFrame,
    polymarket_topics_df: pd.DataFrame,
    sim_matrix: np.ndarray
) -> None:
    plt.figure(figsize=(12, 25))
    sns.heatmap(sim_matrix,
                xticklabels=polymarket_topics_df["label"],
                yticklabels=reddit_topics_df["label"],
                cmap="coolwarm", annot=False)
    plt.title("Topic Overlap Heatmap")
    plt.xlabel("Polymarket Topics")
    plt.ylabel("Reddit Topics")
    plt.tight_layout()
    #save to output
    plt.savefig("/topic_overlap_heatmap.png")


if __name__ == "__main__":
    """
    Quick test runner:
      • auto-picks the freshest Reddit NDJSON and Polymarket JSON from data/daily
      • builds corpora
      • fits separate BERTopic models
      • computes topic centroids and Reddit↔Polymarket alignments
      • saves CSVs + a heatmap under data/outputs
    """
    import os, json, glob, random
    from pathlib import Path
    from datetime import datetime
    import pandas as pd
    import numpy as np
    

    
    # ensure repo root is on sys.path
    REPO_ROOT = Path(__file__).resolve().parents[2]  # adjust if needed
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
        
    os.chdir(REPO_ROOT)
    print(os.getcwd())

    import src.sources.polymarket.polymarket_features as pf
 
    # ---------- paths ----------
    DAILY_DIR = REPO_ROOT / "public/files/source_data"
    OUT_DIR   = REPO_ROOT / "public/files" / "nlp_outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- helpers ----------
    def _latest(pattern: str) -> Path | None:
        files = list(DAILY_DIR.glob(pattern))
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    def _load_jsonish(path: Path) -> list[dict]:
        if not path:
            return []
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return []
        # Try JSONL first
        if "\n{" in text or text.startswith("{") and "\n" in text:
            try:
                return [json.loads(line) for line in text.splitlines() if line.strip()]
            except Exception:
                pass
        # Fallback: regular JSON array/object
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            # Some dumps wrap under a key; flatten best-effort
            if isinstance(data, dict):
                for k in ("records", "data", "markets", "items", "results"):
                    if isinstance(data.get(k), list):
                        return data[k]
                return [data]
        except Exception:
            return []
        return []

    # ---------- pick freshest inputs ----------
    reddit_path = _latest("reddit/reddit_daily_all_*.ndjson") or _latest("reddit/reddit_daily_all_*.json")
    popular_path = _latest("reddit/reddit_popular_*.ndjson")  # optional
    poly_path = _latest("polymarket/polymarket_*.jsonl")

    print("Inputs detected:")
    print("  Reddit all : ", reddit_path)
    print("  Reddit pop : ", popular_path)
    print("  Polymarket : ", poly_path)

    reddit_raw = _load_jsonish(reddit_path) + _load_jsonish(popular_path)
    poly_raw   = _load_jsonish(poly_path)

    if not reddit_raw:
        print("No reddit records found; aborting.")
        raise SystemExit(1)
    if not poly_raw:
        print("No polymarket records found; aborting.")
        raise SystemExit(1)

    # ---------- small, fast sample for a smoke test ----------
    # Bump these caps once you're happy (e.g., 5000/5000)
    random.seed(42)
    REDDIT_CAP = 350000
    POLY_CAP   = 350000
    if len(reddit_raw) > REDDIT_CAP:
        reddit_raw = random.sample(reddit_raw, REDDIT_CAP)
    if len(poly_raw) > POLY_CAP:
        poly_raw = random.sample(poly_raw, POLY_CAP)

    # ---------- build corpora ----------
    print("Building Reddit corpus…")
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

    print("Building Polymarket corpus…")
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
    
    # ---------- Keep only non-empty texts + align raw records ----------
    poly_texts_all = poly_texts
    mask = [bool(t and t.strip()) for t in poly_texts_all]
    poly_texts = [t for t, keep in zip(poly_texts_all, mask) if keep]
    poly_raw_kept = [rec for rec, keep in zip(poly_raw, mask) if keep]
        
    reddit_texts = [t for t in reddit_texts if t and t.strip()]
    poly_texts   = [t for t in poly_texts   if t and t.strip()]
    
    print(f"Clean reddit texts {len(reddit_texts)}")
    print(f"Clean poly texts {len(poly_texts)}")


    # ---------- fit topic models ----------
    print("Fitting Reddit BERTopic…")
    r_topics, _, r_model, r_st_model = embed_and_fit(
        reddit_texts,
        stop_words="english",
        ngram_range=(1, 3),
        min_df=10,
        max_df=0.5,
    )
    print("Fitting Polymarket BERTopic…")
    p_topics, _, p_model, p_st_model = embed_and_fit(
        poly_texts,
        stop_words="english",
        ngram_range=(1, 3),
        min_df=5,        # often fewer markets than reddit posts
        max_df=0.6,
    )

    # ---------- document info + embeddings ----------
    r_docs = r_model.get_document_info(reddit_texts)
    p_docs = p_model.get_document_info(poly_texts)
    p_docs["doc_id"] = np.arange(len(p_docs))
    
    # Compute features for the *same* kept snapshots:
    feat_df = pd.DataFrame([pf.extract_market_features(rec) for rec in poly_raw_kept]).add_prefix("feat_")
    feat_df["doc_id"] = np.arange(len(feat_df))

    # (Optional) attach meta you want to carry:
    meta_df = pd.DataFrame({
        "doc_id": np.arange(len(poly_raw_kept)),
        "market_id": [ (rec.get("market") or {}).get("id") for rec in poly_raw_kept ],
        "question":  [ (rec.get("market") or {}).get("question") for rec in poly_raw_kept ],
        "slug":      [ (rec.get("market") or {}).get("slug") for rec in poly_raw_kept ],
        "outcomes":  [ (rec.get("market") or {}).get("outcomes") for rec in poly_raw_kept ],
        "active":    [ bool((rec.get("market") or {}).get("active")) for rec in poly_raw_kept ],
        "archived":  [ bool((rec.get("market") or {}).get("archived")) for rec in poly_raw_kept ],
        "closed":    [ bool((rec.get("market") or {}).get("closed")) for rec in poly_raw_kept ],
    })

    # Merge into BERTopic doc table (post-hoc, safe):
    p_docs = p_docs.merge(meta_df.merge(feat_df, on="doc_id", how="left"), on="doc_id", how="left")

    # SentenceTransformer embeddings for centroids
    r_emb = r_st_model.encode(reddit_texts, show_progress_bar=False, convert_to_numpy=True)
    p_emb = p_st_model.encode(poly_texts,   show_progress_bar=False, convert_to_numpy=True)

    r_docs["embedding"] = list(r_emb)
    p_docs["embedding"] = list(p_emb)

    # ---------- centroids per topic ----------
    r_centroids = compute_topic_centroids(r_docs, emb_col="embedding", topic_col="Topic", label_col="Name")
    p_centroids = compute_topic_centroids(p_docs, emb_col="embedding", topic_col="Topic", label_col="Name")

    # ---------- align topics ----------
    align_df, sim_matrix = extract_semantically_overlapping_topics(r_centroids, p_centroids, threshold=0.60)

    # ---------- save outputs ----------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (OUT_DIR / f"reddit_topics_{ts}.csv").write_text(r_centroids.to_csv(index=False), encoding="utf-8")
    (OUT_DIR / f"polymarket_topics_{ts}.csv").write_text(p_centroids.to_csv(index=False), encoding="utf-8")
    (OUT_DIR / f"reddit_poly_align_{ts}.csv").write_text(align_df.to_csv(index=False), encoding="utf-8")

    # heatmap (labels come from r_centroids/p_centroids)
    try:
        get_topic_similarity_heatmap(r_centroids.rename(columns={"label": "Name"}),
                                     p_centroids.rename(columns={"label": "Name"}),
                                     sim_matrix.values)
        # function saves to data/outputs/topic_overlap_heatmap.png
    except Exception as e:
        print("Heatmap generation failed:", e)

    print("\nDone.")
    print(f"  Saved: {OUT_DIR / f'reddit_topics_{ts}.csv'}")
    print(f"  Saved: {OUT_DIR / f'polymarket_topics_{ts}.csv'}")
    print(f"  Saved: {OUT_DIR / f'reddit_poly_align_{ts}.csv'}")
    print(f"  Heatmap: {OUT_DIR / 'topic_overlap_heatmap.png'}")
