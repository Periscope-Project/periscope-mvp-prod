import os, json, argparse
import mysql.connector as mysql
from dotenv import load_dotenv
import re

load_dotenv()
DB = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB", "periscope"),
}
LLM_MODEL = os.getenv("LITELLM_MODEL", "vertex_ai/gemini-2.5-pro")


def outlook_to_enum(s: str) -> str | None:
    """
    Strip emojis and return clean outlook string.
    Returns None if missing.
    
    """
    if s == "↔️ Stable Trend" or s == "Stable Trend":
        return "Stable Trend"
    if s == "📈 Moderate Growth" or s == "Moderate Growth":
        return "Stable Trend"
    if s == "📉 Declining Signal" or s == "Declining Signal":
        return "Declining Signal"
    if s == "🚀 Breakout Potential" or s == "Breakout Potential":
        return "Breakout Potential"
    
    return None
        
    


def watch_to_enum(flag: str) -> str:
    if not flag: return "none"
    return "watch" if str(flag).lower().startswith("y") else "none"

def as_json(val):
    return json.dumps(val, ensure_ascii=False) if val is not None else None

def bool_to_tinyint(b):
    if b is True: return 1
    if b is False: return 0
    return None

def platform_spread(platforms_present):
    d = {}
    if isinstance(platforms_present, list):
        for p in platforms_present:
            d[str(p)] = True
    return d or None

UPSERT_TREND = """
INSERT INTO trend_signal_output
(trend_id, group_id, headline, tldr,
 trend_tags_json, industry_tags_json, cross_platform, platform_spread_json,
 historical_analogues_json, quotes_citations_json, narrative_analysis,
 confidence_score, confidence_score_explanation, watch_flag, watch_rationale,
 created_llm_model)
VALUES
(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON DUPLICATE KEY UPDATE
  group_id=VALUES(group_id),
  headline=VALUES(headline),
  tldr=VALUES(tldr),
  trend_tags_json=VALUES(trend_tags_json),
  industry_tags_json=VALUES(industry_tags_json),
  cross_platform=VALUES(cross_platform),
  platform_spread_json=VALUES(platform_spread_json),
  historical_analogues_json=VALUES(historical_analogues_json),
  quotes_citations_json=VALUES(quotes_citations_json),
  narrative_analysis=VALUES(narrative_analysis),
  confidence_score=VALUES(confidence_score),
  confidence_score_explanation=VALUES(confidence_score_explanation),
  watch_flag=VALUES(watch_flag),
  watch_rationale=VALUES(watch_rationale),
  created_llm_model=VALUES(created_llm_model)
"""

UPSERT_GRID = """
INSERT INTO prediction_grid (group_id, outlook, why, break_point_alerts)
VALUES (%s,%s,%s,%s)
ON DUPLICATE KEY UPDATE
  outlook=VALUES(outlook),
  why=VALUES(why),
  break_point_alerts=VALUES(break_point_alerts)
"""

def load_trends(json_path: str) -> int:
    with open(json_path, "r", encoding="utf-8") as f:
        root = json.load(f)

    trends = root.get("trends", [])
    cn = mysql.connect(**DB); cn.autocommit = False
    cur = cn.cursor()

    rows = 0
    for tb in trends:
        group_id = tb.get("group_id")
        trend_id = tb.get("trend_id")
        if group_id is None or trend_id is None:
            continue

        watch = tb.get("watch") or {}
        pred  = tb.get("prediction_grid") or {}

        cur.execute(
            UPSERT_TREND,
            (
                trend_id, group_id,
                tb.get("headline"),
                tb.get("tldr"),
                as_json(tb.get("trend_tags")),
                as_json(tb.get("industry_tags")),
                bool_to_tinyint(tb.get("cross_platform")),
                as_json(platform_spread(tb.get("platforms_present"))),
                as_json(tb.get("historical_analogues")),
                as_json(tb.get("quotes_citations")),
                tb.get("narrative_analysis"),
                tb.get("confidence_score"),
                tb.get("confidence_score_explanation"),
                watch_to_enum(watch.get("flag")),
                watch.get("rationale"),
                LLM_MODEL,
            ),
        )

        cur.execute(
            UPSERT_GRID,
            (
                group_id,
                outlook_to_enum(pred.get("outlook")),
                pred.get("why"),
                pred.get("break_point_alerts"),
            ),
        )
        rows += 1

    cn.commit()
    cur.close(); cn.close()
    return rows

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
        default=os.path.join("public","files","api_app","all_trends_2025_09_08_171009.json"))
    args = parser.parse_args()
    count = load_trends(args.path)
    print(f"Trend briefs loaded: {count} rows (trend_signal_output + prediction_grid by group_id)")
