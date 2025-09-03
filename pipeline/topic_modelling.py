import pandas as pd
import numpy as np
import re
import html
import nltk
from typing import Any, Dict, List, Tuple

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
import matplotlib.pyplot as plt
import seaborn as sns

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
    try:
        nltk.data.find("corpora/stopwords")
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("omw-1.4", quiet=True)


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


def build_bert_corpus_from_polymarket(
    records: List[Dict[str, Any]],
    include_title: bool = True,
    include_description: bool = True,
    include_all_questions: bool = True,
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    phrase_min_count: int = 5,
    phrase_threshold: int = 10,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Build a BERT-style corpus from Polymarket *events*, not individual markets.
    Each record should be one event dict, which may contain a list of 'markets'.
    Returns:
      - tokenized inputs (for .fit_transform)
      - cleaned_texts: one string per event
    """
    # 1) CLEAN + AGGREGATE PER EVENT
    cleaned_texts: List[str] = []
    for rec in records:
        parts: List[str] = []
        if include_title and rec.get("title"):
            parts.append(str(rec["title"]))
        if include_description and rec.get("description"):
            parts.append(str(rec["description"]))
        if include_all_questions and rec.get("markets"):
            # flatten *all* market.questions into one blob
            qs = [m.get("question", "") for m in rec["markets"]]
            parts.append(" ".join(qs))
        raw = "\n".join(parts)
        # reuse your existing cleaning pipeline
        cleaned = clean_text_for_bert(
            {"title": "", "text": raw, "top_comments": []},
            include_title=False,
            include_text=True,
            include_top_comments=0,
            lower=True,
            collapse_whitespace=True,
        )
        cleaned_texts.append(cleaned)

    # 2) PHRASE DETECTION (bigrams/trigrams)
    token_lists = [txt.split() for txt in cleaned_texts]
    bigram = Phrases(token_lists, min_count=phrase_min_count, threshold=phrase_threshold)
    phraser = Phraser(bigram)
    docs_with_phrases = [" ".join(phraser[tok]) for tok in token_lists]

    # 3) TOKENIZE
    tokenized = tokenize_for_bert(
        docs_with_phrases,
        model_name=model_name,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return tokenized, docs_with_phrases




def embed_and_fit(
    texts: List[str],
    stop_words: str = "english",
    ngram_range: Tuple[int, int] = (1, 3),
    min_df: int = 10,
    max_df: float = 0.5,
    umap_params: Dict[str, Any] = None,
    hdbscan_params: Dict[str, Any] = None,
) -> Tuple[np.ndarray, np.ndarray, BERTopic]:
    # Vectorizer to drop stop-words and extract n-grams
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df
    )
    # Embedding & topic model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = UMAP(**(umap_params or {"n_neighbors": 15, "min_dist": 0.0}))
    hdbscan_model = HDBSCAN(**(hdbscan_params or {"min_cluster_size": 20}))
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        language="english",
        verbose=True
    )
    topics, probs = topic_model.fit_transform(texts)
    return topics, probs, topic_model


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
    plt.savefig("data/outputs/topic_overlap_heatmap.png")
