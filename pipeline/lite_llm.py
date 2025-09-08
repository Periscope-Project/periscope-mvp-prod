# pipeline/lite_llm.py
from __future__ import annotations

import os
import sys
import json
import textwrap
import time
import traceback
from datetime import datetime, date
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

import requests
from requests.exceptions import HTTPError
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Ensure repo root on sys.path so `from utils...` works when running this file.
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # …/periscope-mvp-prod
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_helpers import parse_chunk_json  # now importable from anywhere

# -----------------------------------------------------------------------------
# ENV / DEFAULTS
# -----------------------------------------------------------------------------
load_dotenv()

MODEL_NAME       = os.getenv("LITELLM_MODEL", "vertex_ai/gemini-2.5-pro")
MAX_CHARS        = int(os.getenv("LITELLM_MAX_CHARS", "300000"))
SLEEP_SEC        = float(os.getenv("LITELLM_SLEEP_SEC", "0"))
SAVE_DIR_DEFAULT = os.getenv("PERISCOPE_OUT_DIR", "public/files")
SPLIT_ON_400     = os.getenv("LITELLM_SPLIT_ON_400", "1") not in ("0", "false", "False")

# Hard safety cap for request size; many proxies balk at very large prompts.
# You can tune this via env if needed.
CHAR_CAP_HINT    = int(os.getenv("LITELLM_CHAR_CAP_HINT", "50000"))

# -----------------------------------------------------------------------------
# API helpers
# -----------------------------------------------------------------------------
def _resolve_api(api_key: str | None = None, base_url: str | None = None) -> tuple[str, Dict[str, str]]:
    """
    Resolve API URL and headers. Uses /v1 path which most LiteLLM proxies expose.
    """
    key = api_key or os.getenv("LITELLM_API_KEY")
    base = (base_url or os.getenv("LITELLM_LOCATION") or "").strip().rstrip("/")
    if not key or not base:
        raise RuntimeError("LiteLLM API config missing: set LITELLM_API_KEY and LITELLM_LOCATION")

    # Normalize to /v1
    if not base.endswith("/v1"):
        base = f"{base}/v1"

    url = f"{base}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    return url, headers

# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# JSON helpers
# -----------------------------------------------------------------------------
def _json_default(o: Any):
    # datetime / date
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    if isinstance(o, np.datetime64):
        return pd.Timestamp(o).isoformat()
    if o is pd.NaT:
        return None

    # numpy / pandas scalars
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(o, (np.bool_,)):
        return bool(o)

    # arrays / sets
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return list(o)

    # generic pandas NA
    try:
        if pd.isna(o):
            return None
    except Exception:
        pass

    return str(o)

def compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, default=_json_default)

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

# -----------------------------------------------------------------------------
# Network helpers with good logs + auto split on 400
# -----------------------------------------------------------------------------
def _post_with_logging(url, headers, model, prompt, *, max_tokens=2000, temperature=0.2, timeout=120):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    try:
        resp.raise_for_status()
    except HTTPError as e:
        # augment with server body for debugging
        server_text = resp.text or ""
        raise HTTPError(
            f"HTTP {resp.status_code} from {url}\n"
            f"Model: {model}\n"
            f"Prompt size (chars): {len(prompt)}\n"
            f"Server said: {server_text[:2000]}",
            response=resp
        ) from e
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

def _send_or_split(batch, *, url, headers, model, cap_chars, save_dir, batch_idx, enable_split=True):
    """
    Try to send the batch. If 400 occurs and multiple items present, split into halves and retry.
    Returns a list of content strings (for each successful sub-batch).
    """
    outputs: List[str] = []
    prompt = make_prompt(batch)
    try:
        content = _post_with_logging(url, headers, model, prompt)
        outputs.append(content)
        return outputs
    except HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if not enable_split or status != 400 or len(batch) <= 1:
            # Save the failing payload for inspection
            fail_dir = Path(save_dir) / "litellm" / "failed_batches"
            fail_dir.mkdir(parents=True, exist_ok=True)
            with open(fail_dir / f"batch_{batch_idx:04d}.json", "w", encoding="utf-8") as f:
                f.write(compact(batch))
            with open(fail_dir / f"batch_{batch_idx:04d}.error.txt", "w", encoding="utf-8") as f:
                f.write(str(e))
            # Re-raise so caller records this as an error
            raise

        # Split & recurse
        mid = len(batch) // 2
        left  = batch[:mid]
        right = batch[mid:]
        outputs += _send_or_split(
            left,  url=url, headers=headers, model=model,
            cap_chars=cap_chars, save_dir=save_dir, batch_idx=batch_idx*2-1, enable_split=enable_split
        )
        outputs += _send_or_split(
            right, url=url, headers=headers, model=model,
            cap_chars=cap_chars, save_dir=save_dir, batch_idx=batch_idx*2, enable_split=enable_split
        )
        return outputs

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

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
    show_progress: bool = True,  # 👈 NEW
) -> Dict[str, str]:
    """
    Summarize TopicSignal list and write outputs under save_dir.
    Returns {"raw_txt": ..., "errors": ..., "combined_json": ...}
    """
    url, headers = _resolve_api(api_key=api_key, base_url=base_url)

    mdl   = model_name or MODEL_NAME
    cap   = max_chars if max_chars is not None else MAX_CHARS
    pause = sleep_sec if sleep_sec is not None else SLEEP_SEC

    # keep batches modest to avoid provider limits
    cap = min(cap, CHAR_CAP_HINT)

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

    outputs: List[str] = []
    errors: List[str]  = []

    it = enumerate(batches, start=1)
    if show_progress:
        it = tqdm(it, total=len(batches), desc="LiteLLM batches", unit="batch")

    for i, batch in it:
        try:
            parts = _send_or_split(
                batch,
                url=url,
                headers=headers,
                model=mdl,
                cap_chars=cap,
                save_dir=save_dir,
                batch_idx=i,
                enable_split=SPLIT_ON_400,
            )
            outputs.extend(parts)
        except Exception as e:
            errors.append(f"BATCH {i} ERROR:\n{e}\n{traceback.format_exc()}")
        finally:
            if show_progress and hasattr(it, "set_postfix_str"):
                it.set_postfix_str(f"ok={len(outputs)} err={len(errors)}")
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

    # Try to combine; if parsing fails, still return paths.
    try:
        parse_chunk_json(str(out_txt), str(out_json))
        print(f"[litellm] ✅ Combined JSON → {out_json}")
    except Exception as e:
        with open(err_log, "a", encoding="utf-8") as f:
            f.write(f"\n\nCOMBINE ERROR:\n{repr(e)}\n{traceback.format_exc()}")
        print(f"[litellm] ⚠️ Combine failed. See {err_log}")

    return {"raw_txt": str(out_txt), "errors": str(err_log), "combined_json": str(out_json)}
# -----------------------------------------------------------------------------
# CLI helpers (optional)
# -----------------------------------------------------------------------------
def _find_latest_aligned(save_dir: str = SAVE_DIR_DEFAULT) -> str:
    """
    Find newest aligned_topics_full_*.json under common locations.
    Prefers save_dir root; falls back to data/outputs/.
    """
    candidates: List[str] = []
    candidates += glob(str(Path(save_dir) / "aligned_topics_full_*.json"))
    candidates += glob("data/outputs/aligned_topics_full_*.json")
    if not candidates:
        raise FileNotFoundError("No aligned_topics_full_*.json found.")
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]

# -----------------------------------------------------------------------------
# Standalone CLI (sequential; uses same logic & logging as summarize_topics)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        url, headers = _resolve_api()
        SAVE_DIR = os.getenv("PERISCOPE_OUT_DIR", SAVE_DIR_DEFAULT)
        JSON_PATH = _find_latest_aligned(SAVE_DIR)
        print(f"[litellm] Using latest aligned file: {JSON_PATH}")

        with open(JSON_PATH, encoding="utf-8") as f:
            topics_all = json.load(f)

        result = summarize_topics(
            topics_all,
            save_dir=SAVE_DIR,
            model_name=os.getenv("LITELLM_MODEL", MODEL_NAME),
            max_chars=int(os.getenv("LITELLM_MAX_CHARS", str(MAX_CHARS))),
            sleep_sec=float(os.getenv("LITELLM_SLEEP_SEC", str(SLEEP_SEC))),
        )

        print(f"[litellm] Raw responses: {result['raw_txt']}")
        print(f"[litellm] Combined JSON: {result['combined_json']}")

    except Exception as e:
        print("[litellm] Fatal error:", repr(e))
        print(traceback.format_exc())
