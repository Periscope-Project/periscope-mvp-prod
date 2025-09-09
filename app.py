# app.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import mysql.connector as mysql
from mysql.connector.pooling import MySQLConnectionPool

# ── ENV ────────────────────────────────────────────────────────────────────────
load_dotenv()  # reads .env in the working directory if present

DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB", "periscope"),
    "charset": "utf8mb4",
}

POOL_NAME  = os.getenv("DB_POOL_NAME", "periscope_pool")
POOL_SIZE  = int(os.getenv("DB_POOL_SIZE", "5"))

pool: Optional[MySQLConnectionPool] = None

def get_conn():
    global pool
    if pool is None:
        pool = MySQLConnectionPool(pool_name=POOL_NAME, pool_size=POOL_SIZE, **DB_CONFIG)
    return pool.get_connection()

# ── FASTAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Periscope Trends API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MODELS (response schemas) ──────────────────────────────────────────────────
class Trend(BaseModel):
    trend_id: int
    group_id: int
    headline: Optional[str] = None
    tldr: Optional[str] = None
    trend_tags: Optional[List[str]] = None
    industry_tags: Optional[List[str]] = None
    cross_platform: Optional[bool] = None
    platform_spread: Optional[Dict[str, Any]] = None
    historical_analogues: Optional[List[str]] = None
    quotes_citations: Optional[List[str]] = None
    narrative_analysis: Optional[str] = None
    confidence_score: Optional[int] = None
    confidence_score_explanation: Optional[str] = None
    watch_flag: Optional[str] = None
    watch_rationale: Optional[str] = None
    outlook: Optional[str] = None          # from prediction_grid
    why: Optional[str] = None              # from prediction_grid
    break_point_alerts: Optional[str] = None  # from prediction_grid
    created_llm_model: Optional[str] = None

# ── HELPERS ────────────────────────────────────────────────────────────────────
JSON_COLS = [
    ("trend_tags_json", "trend_tags"),
    ("industry_tags_json", "industry_tags"),
    ("platform_spread_json", "platform_spread"),
    ("historical_analogues_json", "historical_analogues"),
    ("quotes_citations_json", "quotes_citations"),
]

SELECT_BASE = """
SELECT
  t.trend_id, t.group_id, t.headline, t.tldr,
  t.trend_tags_json, t.industry_tags_json, t.cross_platform, t.platform_spread_json,
  t.historical_analogues_json, t.quotes_citations_json, t.narrative_analysis,
  t.confidence_score, t.confidence_score_explanation, t.watch_flag, t.watch_rationale,
  t.created_llm_model,
  g.outlook, g.why, g.break_point_alerts
FROM trend_signal_output t
LEFT JOIN prediction_grid g ON g.group_id = t.group_id
"""

def _parse_json_cols(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for col, target in JSON_COLS:
        raw = row.get(col)
        out[target] = (json.loads(raw) if isinstance(raw, str) and raw else None)
        out.pop(col, None)
    return out

# ── ROUTES ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/feed", response_model=List[Trend])
def feed(
    q: Optional[str] = Query(None, description="Search in headline/tldr"),
    outlook: Optional[str] = Query(None, description="Filter by outlook (e.g., 'Moderate Growth')"),
    watch: Optional[str] = Query(None, description="Filter watch flag: watch/none"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    order: str = Query("trend_id", description="Sort by: trend_id|group_id|confidence_score"),
    desc: bool = Query(True, description="Sort descending"),
):
    try:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)

        where, params = [], []

        if q:
            where.append("(t.headline LIKE %s OR t.tldr LIKE %s)")
            like = f"%{q}%"
            params.extend([like, like])

        if outlook:
            where.append("g.outlook = %s")
            params.append(outlook)

        if watch:
            where.append("t.watch_flag = %s")
            params.append(watch)

        sql = [SELECT_BASE]
        if where:
            sql.append("WHERE " + " AND ".join(where))

        # very small, safe allowlist for ORDER BY
        order_map = {
            "trend_id": "t.trend_id",
            "group_id": "t.group_id",
            "confidence_score": "t.confidence_score",
        }
        order_col = order_map.get(order, "t.trend_id")
        sql.append(f"ORDER BY {order_col} {'DESC' if desc else 'ASC'}")
        sql.append("LIMIT %s OFFSET %s")
        params.extend([limit, offset])

        cur.execute(" ".join(sql), params)
        rows = [ _parse_json_cols(r) for r in cur.fetchall() ]
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            cur.close(); conn.close()
        except Exception:
            pass

@app.get("/trends/{trend_id}", response_model=Trend)
def get_trend(trend_id: int):
    try:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute(SELECT_BASE + " WHERE t.trend_id = %s LIMIT 1", (trend_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
        return _parse_json_cols(row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            cur.close(); conn.close()
        except Exception:
            pass

@app.get("/groups/{group_id}", response_model=List[Trend])
def get_group(group_id: int):
    try:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute(SELECT_BASE + " WHERE t.group_id = %s ORDER BY t.trend_id DESC", (group_id,))
        rows = [ _parse_json_cols(r) for r in cur.fetchall() ]
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            cur.close(); conn.close()
        except Exception:
            pass

# For local run: `uvicorn app:app --host 0.0.0.0 --port 8000 --reload`
if __name__ == "__main__":
    import uvicorn

    # Run on both IPv4 and IPv6 (host "::" binds to all addresses)
    uvicorn.run(
        "app:app",
        host="::",         # IPv6 unspecified address → also covers IPv4 on most OS
        port=8000,
        reload=True,       # hot reload in dev, disable in prod
        log_level="info"
    )
