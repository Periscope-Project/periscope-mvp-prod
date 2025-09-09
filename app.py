# app.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import mysql.connector as mysql
from mysql.connector.pooling import MySQLConnectionPool

# pipeline trigger (exported from src/main.py)
# ensure /opt/periscope/src/__init__.py exists so this import works
from src.main import run_pipeline

# ── ENV ────────────────────────────────────────────────────────────────────────
load_dotenv()  # reads .env in working directory (/opt/periscope/.env)

DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB", "periscope"),
    "charset": "utf8mb4",
}
POOL_NAME = os.getenv("DB_POOL_NAME", "periscope_pool")
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
CORS_ORIGINS = [o for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if o]

# ── DB POOL ────────────────────────────────────────────────────────────────────
pool: Optional[MySQLConnectionPool] = None

def get_conn():
    """Get a pooled MySQL connection."""
    global pool
    if pool is None:
        pool = MySQLConnectionPool(pool_name=POOL_NAME, pool_size=POOL_SIZE, **DB_CONFIG)
    return pool.get_connection()

# ── FASTAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Periscope Trends API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MODELS ─────────────────────────────────────────────────────────────────────
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
    outlook: Optional[str] = None
    why: Optional[str] = None
    break_point_alerts: Optional[str] = None
    created_llm_model: Optional[str] = None

class RunRequest(BaseModel):
    push_to_sql: Optional[bool] = None  # reserved for future per-request overrides

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
    conn = cur = None
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
        rows = [_parse_json_cols(r) for r in cur.fetchall()]
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if cur: cur.close()
            if conn: conn.close()
        except Exception:
            pass

@app.get("/trends/{trend_id}", response_model=Trend)
def get_trend(trend_id: int):
    conn = cur = None
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
            if cur: cur.close()
            if conn: conn.close()
        except Exception:
            pass

@app.get("/groups/{group_id}", response_model=List[Trend])
def get_group(group_id: int):
    conn = cur = None
    try:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute(SELECT_BASE + " WHERE t.group_id = %s ORDER BY t.trend_id DESC", (group_id,))
        rows = [_parse_json_cols(r) for r in cur.fetchall()]
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if cur: cur.close()
            if conn: conn.close()
        except Exception:
            pass

# ── OPTIONAL: manual trigger to run the pipeline once (background) ─────────────
@app.post("/run-pipeline")
def run_pipeline_endpoint(bg: BackgroundTasks, req: RunRequest | None = None):
    """
    Queues a pipeline run in the background so the HTTP request returns quickly.
    Real scheduling should be done by cron (python -m src.main).
    """
    # If you later want per-request overrides, set env here before calling:
    # if req and req.push_to_sql is not None:
    #     os.environ["PUSH_TO_SQL"] = "1" if req.push_to_sql else "0"
    bg.add_task(run_pipeline)
    return {"status": "queued"}

# ── OPTIONAL: in-API scheduler (disabled by default) ───────────────────────────
# Enable with RUN_SCHEDULE_IN_API=1 (use ONLY with a single worker or add a lock)
if os.getenv("RUN_SCHEDULE_IN_API", "0").lower() in {"1", "true", "yes"}:
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger

        @app.on_event("startup")
        async def _start_scheduler():
            tz = os.getenv("SCHEDULE_TZ", "UTC")
            hour = int(os.getenv("SCHEDULE_HOUR", "2"))
            minute = int(os.getenv("SCHEDULE_MINUTE", "0"))
            sched = AsyncIOScheduler(timezone=tz)
            sched.add_job(run_pipeline, CronTrigger(hour=hour, minute=minute),
                          id="daily_pipeline", replace_existing=True)
            sched.start()
            print(f"⏰ In-API schedule enabled @ {hour:02d}:{minute:02d} {tz}")
    except Exception as _e:
        # Scheduler is optional; don't crash the app if it's not installed
        pass

# ── DEV ONLY: local run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="::", port=8000, reload=True, log_level="info")
