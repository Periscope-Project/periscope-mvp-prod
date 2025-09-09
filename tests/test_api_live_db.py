# tests/test_api_live_db.py
import os, sys, json, random, string
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import mysql.connector as mysql
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LIVE = os.getenv("LIVE_DB_TEST", "1") not in {"0","false","no",""}
pytestmark = pytest.mark.skipif(not LIVE, reason="Set LIVE_DB_TEST=1 to run live DB test")

def _rand_suffix(n=6):
    return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(n))

def load_app_module():
    app_path = ROOT / "app.py"
    spec = spec_from_file_location("app", str(app_path))
    mod = module_from_spec(spec); sys.modules["app"] = mod
    assert spec and spec.loader; spec.loader.exec_module(mod)  # type: ignore
    return mod

def test_api_live_db(tmp_path):
    # ----- connect to server using your normal creds -----
    db_host = os.getenv("MYSQL_HOST", "127.0.0.1")
    db_port = int(os.getenv("MYSQL_PORT", "3306"))
    db_user = os.getenv("MYSQL_USER", "root")
    db_pass = os.getenv("MYSQL_PASSWORD", "")
    base_db = os.getenv("MYSQL_DB", "periscope")

    admin = mysql.connect(host=db_host, port=db_port, user=db_user, password=db_pass)
    admin.autocommit = True
    cur = admin.cursor()

    schema = f"{base_db}_test_{_rand_suffix()}"
    cur.execute(f"CREATE DATABASE `{schema}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")

    # minimal tables used by the app’s SELECTs
    cur.execute(f"""
        CREATE TABLE `{schema}`.trend_signal_output (
          trend_id INT PRIMARY KEY,
          group_id INT,
          headline TEXT,
          tldr TEXT,
          trend_tags_json TEXT,
          industry_tags_json TEXT,
          cross_platform TINYINT(1),
          platform_spread_json TEXT,
          historical_analogues_json TEXT,
          quotes_citations_json TEXT,
          narrative_analysis TEXT,
          confidence_score INT,
          confidence_score_explanation TEXT,
          watch_flag VARCHAR(16),
          watch_rationale TEXT,
          created_llm_model VARCHAR(255)
        ) ENGINE=InnoDB;
    """)
    cur.execute(f"""
        CREATE TABLE `{schema}`.prediction_grid (
          group_id INT PRIMARY KEY,
          outlook VARCHAR(255),
          why TEXT,
          break_point_alerts TEXT
        ) ENGINE=InnoDB;
    """)

    # seed data
    cur.execute(f"""
        INSERT INTO `{schema}`.prediction_grid (group_id, outlook, why, break_point_alerts)
        VALUES (7, 'Moderate Growth', 'Cross-signal alignment', 'If BTC < 50k, reassess');
    """)
    cur.execute(f"""
        INSERT INTO `{schema}`.trend_signal_output
        (trend_id, group_id, headline, tldr, trend_tags_json, industry_tags_json, cross_platform,
         platform_spread_json, historical_analogues_json, quotes_citations_json, narrative_analysis,
         confidence_score, confidence_score_explanation, watch_flag, watch_rationale, created_llm_model)
        VALUES
        (42, 7, 'Test Headline', 'Short summary', '["AI","Finance"]', '["AI/ML"]', 1,
         '{{"reddit":5,"polymarket":2}}', '["Dotcom 1999"]', '["source:example"]', 'Looks bullish',
         78, 'Good overlap + momentum', 'watch', 'Rising mentions', 'gemini-2.5-pro'),
        (101, 7, 'Row 1', 'Row 1 TLDR', '["AI"]', '["AI/ML"]', 1,
         '{{"reddit":3}}', '[]', '[]', NULL, 55, NULL, 'watch', NULL, NULL),
        (102, 7, 'Row 2', 'Row 2 TLDR', '["AI"]', '["AI/ML"]', 1,
         '{{"reddit":3}}', '[]', '[]', NULL, 88, NULL, 'none', NULL, NULL);
    """)

    cur.close(); admin.close()

    # ----- point the API at the temp schema, then load app.py -----
    os.environ["MYSQL_DB"] = schema
    os.environ["RUN_SCHEDULE_IN_API"] = "0"

    app_module = load_app_module()
    from fastapi.testclient import TestClient
    client = TestClient(app_module.app)

    # sanity: endpoints should work and read our seeded rows
    assert client.get("/health").json() == {"ok": True}
    assert isinstance(client.get("/meta").json().get("version"), str)

    feed = client.get("/feed?limit=5").json()
    assert len(feed) >= 2
    assert any(r["trend_id"] == 42 for r in feed)
    assert isinstance(feed[0]["trend_tags"], list)

    assert client.get("/trends/42").json()["outlook"] == "Moderate Growth"

    grp = client.get("/groups/7").json()
    ids = {r["trend_id"] for r in grp}
    assert {42, 101, 102}.issubset(ids)

    # ----- cleanup -----
    admin = mysql.connect(host=db_host, port=db_port, user=db_user, password=db_pass)
    cur = admin.cursor()
    cur.execute(f"DROP DATABASE `{schema}`")
    cur.close(); admin.close()
