# litellm.py
import os
import json
import textwrap
import time
import traceback
from datetime import datetime
from multiprocessing import Pool, cpu_count
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv
from tqdm import tqdm




# ── ENV / DEFAULTS ───────────────────────────────────────
load_dotenv()

# Module-level defaults (overridable via env or function args)
MODEL_NAME = os.getenv("LITELLM_MODEL", "vertex_ai/gemini-2.5-pro")
MAX_CHARS  = int(os.getenv("LITELLM_MAX_CHARS", "300000"))
SLEEP_SEC  = float(os.getenv("LITELLM_SLEEP_SEC", "0"))
SAVE_DIR_DEFAULT = os.getenv("PERISCOPE_OUT_DIR", "public/files")

def _parse_chunk_json_safe(in_path: str, out_path: str) -> None:
    """
    Import parse_chunk_json lazily so lite_llm loads even if package layout
    varies. Tries utils.data_helpers first, then utils.parse, then package-relative.
    """
    try:
        from utils.data_helpers import parse_chunk_json  # local import
    except Exception:
        try:
            from utils.parse import parse_chunk_json
        except Exception:
            try:
                from ..utils.data_helpers import parse_chunk_json  # type: ignore
            except Exception as e:
                raise ImportError(
                    "Could not import parse_chunk_json. Ensure it exists in "
                    "utils/data_helpers.py or utils/parse.py and run from repo root."
                ) from e
    parse_chunk_json(in_path, out_path)


def _resolve_api(api_key: str | None = None, base_url: str | None = None):
    """Resolve API URL and headers from args/env each time (import-safe)."""
    key = api_key or os.getenv("LITELLM_API_KEY")
    base = (base_url or os.getenv("LITELLM_LOCATION") or "").strip().rstrip("/")
    if not key or not base:
        raise RuntimeError("LiteLLM API config missing: set LITELLM_API_KEY and LITELLM_LOCATION")
    url = f"{base}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    return url, headers

# ── FULL PROMPT HEADER ────────────────────────────────────
super_prompt = textwrap.dedent("""\
You are TrendReporter-Pro, an AI cultural journalist and strategy analyst.

INPUT
• A JSON array of TopicSignal objects (id, label, keywords, reddit_topic, polymarket_topic, metrics, sources).

OUTPUT (STRICT)
Return exactly **one JSON object** with **one key**, so each TopicSignal object output should have 1 key:

{
  "trends": [ TrendBrief, … ]
}

No other top-level keys allowed.

–––––– TrendBrief schema ––––––
Each object inside "trends" must include these keys, and only these keys, **in this order**:

{
  "trend_id":           <TopicSignal.id>,
  "flag_artifact":       <true | false>,  // true ONLY if this topic is artificial noise (e.g., bots, spam, moderator/boilerplate, or otherwise not representing genuine human activity or discourse). DO NOT flag evergreen, organic, or baseline trends as artifacts.
  "maturity_stage":     <"Peak" | "Late-Stage" | "Emerging" | "Early Indicator">,
  "headline":           "<5–8-word hook>",
  "tldr":               "<2–3 sentence summary>",
  "trend_tags":         ["<same as maturity_stage>"],
  "industry_tags":      ["Entertainment","Fashion","Health","Finance","Tech","Gaming","Retail","Politics","Education","AI/ML","Culture","Others","Humour"],
  "cross_platform":     <true | false>,
  "platforms_present":  ["Reddit","Polymarket"],  
  "historical_analogues": [
    "<Brief comparative analysis: draw parallels to a past event or cultural moment, citing timing, virality, fade-out…>"
  ],
  "quotes_citations": Pull out quotes and citations from documents in the signal topic. Say the source as well. Return as a list, for example [
    "\"…\" (`reddit_topic.size:397`)",
    "\"…\" (`reddit_topic.date_range.min_created_iso`)"
  ],
  "narrative_analysis": "<3–5 sentence synthesis using keywords, volume, sentiment, platform spread, market odds, analogues. Include velocity (reddit_topic.size ÷ days) and trajectory.>",
  "prediction_grid": {
    "outlook":          <\"📈 Moderate Growth\" | \"🚀 Breakout Potential\" | \"↔️ Stable Trend\" | \"📉 Declining Signal">,
    "why":              "<1–2 sentence rationale>",
    "break_point_alerts": "<Events that could trigger a shift>"
  },
  "confidence_score": <0–100>, // This score captures the confidence of your predictions based on volume, velocity, sentiment, prediction market odds, historical narratives, platform spread, and search interest growth (if present). See rules for more information.
  "confidence_score_explanation": "<1–3 sentence justification — cite volume/velocity, sentiment, platform spread, novelty, analogues>",
  "watch": { "flag":"Yes"|"No", "rationale":"<1 sentence if to keep tracking>" }
}

RULES:
• Process ALL TopicSignal objects in the input array.
• Compute velocity = reddit_topic.size ÷ days_in_date_range; cite it in narrative_analysis.
• Confidence score should be based on the calculated velocity, document sentiments, historical narratives, platform spread 
• Use concise, journalistic tone.
• Return valid JSON only — no markdown, no extra keys, no commentary.
• Confidence score must be an integer between 0–100.
• Increase confidence if the trend is present across multiple independent platforms.
• If `flag_artifact` is true, cap confidence score at 40.
• If total volume is low (<20 posts/comments), confidence should not exceed 30 unless market odds or analogues strongly support the trend.
• Penalize or moderate confidence if market odds and social momentum (velocity/sentiment) disagree.
• Cite all key factors (volume, velocity, sentiment, spread, novelty, market odds, analogues) in `confidence_score_explanation`.
""")

# ── UTILITIES ────────────────────────────────────────────
def compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

def make_prompt(batch: List[Dict[str, Any]]) -> str:
    payload = compact(batch)
    return f"{super_prompt}\n\nHere is the JSON array of TopicSignal objects:\n{payload}"

def make_batches(items: List[Dict[str, Any]], max_chars: int):
    batch, length = [], 0
    for obj in items:
        s = compact(obj)
        if batch and length + len(s) + 1 > max_chars:
            yield batch
            batch, length = [], 0
        batch.append(obj)
        length += len(s) + 1
    if batch:
        yield batch

# ── PUBLIC: summarize callable (import-safe) ─────────────
def summarize_topics(
    topics: List[Dict[str, Any]] | Dict[str, Any],
    *,
    save_dir: str = SAVE_DIR_DEFAULT,
    model_name: str | None = None,
    max_chars: int | None = None,
    sleep_sec: float | None = None,
    filename_prefix: str = "trend_briefs_litellm",
    api_key: str | None = None,
    base_url: str | None = None,
) -> Dict[str, str]:
    """
    Summarize TopicSignal list and write outputs under save_dir.
    Returns paths dict. Safe to call from main.py.
    """
    url, headers = _resolve_api(api_key=api_key, base_url=base_url)

    mdl   = model_name or MODEL_NAME
    cap   = max_chars if max_chars is not None else MAX_CHARS
    pause = sleep_sec if sleep_sec is not None else SLEEP_SEC

    items = topics if isinstance(topics, list) else [topics]

    today_ymd = datetime.now().strftime("%Y_%m_%d")
    ts = datetime.now().strftime("%H%M%S")
    out_txt  = Path(save_dir) / "litellm" / f"{filename_prefix}_{today_ymd}_{ts}.txt"
    err_log  = Path(save_dir) / "litellm" / f"trend_briefs_errors_{today_ymd}_{ts}.log"
    out_json = Path(save_dir) / "api_app"  / f"all_trends_{today_ymd}.json"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    err_log.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    batches = list(make_batches(items, cap))
    print(f"[litellm] Batches: {len(batches)}  (char cap {cap})")

    outputs, errors = [], []
    for i, batch in enumerate(batches, start=1):
        prompt = make_prompt(batch)
        body = {"model": mdl, "messages": [{"role": "user", "content": prompt}]}
        try:
            resp = requests.post(url, headers=headers, json=body)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            outputs.append(content)
        except Exception as e:
            errors.append(f"BATCH {i} ERROR:\n{repr(e)}\n{traceback.format_exc()}")
        time.sleep(pause)

    with open(out_txt, "w", encoding="utf-8") as f:
        for c in outputs:
            f.write((c or "") + "\n\n")

    if errors:
        with open(err_log, "w", encoding="utf-8") as f:
            f.write("\n\n".join(errors))
        print(f"[litellm] ⚠️ {len(errors)} batches failed → {err_log}")
    else:
        print("[litellm] 🎉 All batches succeeded.")

    _parse_chunk_json_safe(str(out_txt), str(out_json))
    print(f"[litellm] ✅ Combined JSON → {out_json}")

    return {"raw_txt": str(out_txt), "errors": str(err_log), "combined_json": str(out_json)}

# ── CLI helpers ──────────────────────────────────────────
def _find_latest_aligned(save_dir: str = SAVE_DIR_DEFAULT) -> str:
    """
    Find newest aligned_topics_full_*.json under common locations.
    Prefers save_dir root; falls back to data/outputs/.
    """
    candidates = []
    candidates += glob(str(Path(save_dir) / "aligned_topics_full_*.json"))
    candidates += glob("data/outputs/aligned_topics_full_*.json")
    if not candidates:
        raise FileNotFoundError("No aligned_topics_full_*.json found.")
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]

def _resolve_cli_api():
    """Resolve URL/HEADERS into globals for CLI multiprocessing path."""
    url, headers = _resolve_api()
    globals()["URL"] = url
    globals()["HEADERS"] = headers

def call_api(args):
    """
    Worker for CLI multiprocessing path.
    args = (index, batch)
    Returns (index, content_str or None, error_str or None)
    """
    idx, batch = args
    prompt = make_prompt(batch)
    body = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}]}
    try:
        resp = requests.post(URL, headers=HEADERS, json=body)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        time.sleep(SLEEP_SEC)
        return idx, content, None
    except Exception as e:
        return idx, None, f"{repr(e)}\n{traceback.format_exc()}"

# ── MAIN (standalone CLI) ────────────────────────────────
if __name__ == "__main__":
    # CLI config — can be overridden by env
    today = datetime.now().strftime("%Y_%m_%d")
    MODEL_NAME = os.getenv("LITELLM_MODEL", MODEL_NAME)
    MAX_CHARS  = int(os.getenv("LITELLM_MAX_CHARS", str(MAX_CHARS)))
    SLEEP_SEC  = float(os.getenv("LITELLM_SLEEP_SEC", str(SLEEP_SEC)))
    SAVE_DIR   = os.getenv("PERISCOPE_OUT_DIR", SAVE_DIR_DEFAULT)

    _resolve_cli_api()
    JSON_PATH = _find_latest_aligned(SAVE_DIR)
    print(f"[litellm] Using latest: {JSON_PATH}")

    with open(JSON_PATH, encoding="utf-8") as f:
        topics_all = json.load(f)

    all_batches = list(make_batches(topics_all, MAX_CHARS))
    num_batches = len(all_batches)
    print(f"Total batches to send: {num_batches}")

    tasks = list(enumerate(all_batches, start=1))
    results = []
    with Pool(min(cpu_count(), num_batches)) as pool:
        for res in tqdm(pool.imap_unordered(call_api, tasks), total=num_batches, desc="Batches"):
            results.append(res)

    results.sort(key=lambda x: x[0])

    out_txt = Path(SAVE_DIR) / "litellm" / f"trend_briefs_litellm_{today}_{datetime.now():%H%M%S}.txt"
    err_log = Path(SAVE_DIR) / "litellm" / f"trend_briefs_errors_{today}_{datetime.now():%H%M%S}.log"
    out_json = Path(SAVE_DIR) / "api_app"  / f"all_trends_{today}.json"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    err_log.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    outputs, errors = [], []
    with open(out_txt, "w", encoding="utf-8") as out_f:
        for idx, content, err in results:
            if err:
                errors.append(f"BATCH {idx} ERROR:\n{err}")
            else:
                out_f.write(content + "\n\n")
                outputs.append(content)

    if errors:
        with open(err_log, "w", encoding="utf-8") as err_f:
            err_f.write("\n\n".join(errors))
        print(f"⚠️  {len(errors)} batches failed. See {err_log}")
    else:
        print("🎉 All batches succeeded.")

    parse_chunk_json(str(out_txt), str(out_json))
    print(f"✅ Parsed combined JSON to {out_json}")
