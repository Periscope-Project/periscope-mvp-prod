# upload_trends.py
from __future__ import annotations
import json, decimal
from typing import Any, Dict, Iterable, Optional
import mysql.connector as mysql

DEC = decimal.Decimal

def get_conn():
    return mysql.connect(
        host="127.0.0.1",
        port=3306,
        user="periscope",
        password="periscope_pw",
        database="periscope",
        autocommit=False,
    )

def _num(x, typ=float):
    try:
        if x is None or x == "":
            return None
        return typ(x)
    except Exception:
        return None

def upsert_trend_group(cur, group_id: Optional[int], label: Optional[str]) -> int:
    if group_id:
        cur.execute(
            "INSERT INTO trend_group (group_id, label) VALUES (%s,%s) "
            "ON DUPLICATE KEY UPDATE label=COALESCE(VALUES(label), label)",
            (group_id, label),
        )
        return group_id
    cur.execute("INSERT INTO trend_group (label) VALUES (%s)", (label,))
    return cur.lastrowid

def upsert_trend_signal(
    cur,
    group_id: int,
    headline: str,
    tldr: Optional[str],
    trend_tags_json: Optional[dict],
    industry_tags_json: Optional[dict],
    cross_platform: Optional[bool],
    platform_spread_json: Optional[dict],
    historical_analogues_json: Optional[dict],
    quotes_citations_json: Optional[dict],
    narrative_analysis: Optional[str],
    confidence_score: Optional[decimal.Decimal],
    confidence_score_explanation: Optional[str],
    watch_flag: Optional[str],  # enum in your table — pass as string ('0'/'1' or named)
    watch_rationale: Optional[str],
    created_llm_model: Optional[str],
):
    cur.execute(
        """
        INSERT INTO trend_signal
          (group_id, headline, tldr, trend_tags_json, industry_tags_json,
           cross_platform, platform_spread_json, historical_analogues_json,
           quotes_citations_json, narrative_analysis, confidence_score,
           confidence_score_explanation, watch_flag, watch_rationale,
           created_date, created_llm_model, created_at)
        VALUES
          (%s,%s,%s,%s,%s,
           %s,%s,%s,
           %s,%s,%s,
           %s,%s,%s,
           CURDATE(), %s, NOW())
        """,
        (
            group_id, headline, tldr,
            json.dumps(trend_tags_json, ensure_ascii=False) if trend_tags_json is not None else None,
            json.dumps(industry_tags_json, ensure_ascii=False) if industry_tags_json is not None else None,
            1 if cross_platform else 0 if cross_platform is not None else None,
            json.dumps(platform_spread_json, ensure_ascii=False) if platform_spread_json is not None else None,
            json.dumps(historical_analogues_json, ensure_ascii=False) if historical_analogues_json is not None else None,
            json.dumps(quotes_citations_json, ensure_ascii=False) if quotes_citations_json is not None else None,
            narrative_analysis,
            _num(confidence_score, DEC),
            confidence_score_explanation,
            watch_flag,  # if it’s enum like ('Y','N') pass those, else coerce to '1'/'0'
            watch_rationale,
            created_llm_model,
        ),
    )

def load_trends_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        # Accept either a plain list or {"trends": [...]}
        data = json.load(f)
        if isinstance(data, dict) and "trends" in data:
            return data
        if isinstance(data, list):
            return {"trends": data}
        return {"trends": data.get("data", [])}

def upload_trends(trends_doc: Dict[str, Any]) -> int:
    inserted = 0
    conn = get_conn()
    cur = conn.cursor()
    try:
        for tr in trends_doc.get("trends", []):
            # Common fields you likely have in your LLM output
            group = tr.get("group") or {}
            group_id = group.get("group_id")
            group_label = group.get("label") or tr.get("group_label")

            gid = upsert_trend_group(cur, group_id, group_label)

            upsert_trend_signal(
                cur=cur,
                group_id=gid,
                headline=tr.get("headline") or tr.get("title") or "Untitled",
                tldr=tr.get("tldr") or tr.get("summary"),
                trend_tags_json=tr.get("trend_tags"),
                industry_tags_json=tr.get("industry_tags"),
                cross_platform=tr.get("cross_platform"),
                platform_spread_json=tr.get("platform_spread"),
                historical_analogues_json=tr.get("historical_analogues"),
                quotes_citations_json=tr.get("quotes_citations"),
                narrative_analysis=tr.get("narrative"),
                confidence_score=tr.get("confidence_score"),
                confidence_score_explanation=tr.get("confidence_score_explanation"),
                watch_flag=str(tr.get("watch_flag")) if tr.get("watch_flag") is not None else None,
                watch_rationale=tr.get("watch_rationale"),
                created_llm_model=tr.get("model"),
            )
            inserted += 1

        conn.commit()
        return inserted
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    trends = load_trends_json(path)
    n = upload_trends(trends)
    print(f"Inserted trend_signal rows: {n}")
