# tests/test_api_smoke.py
import os, sys, json, types, importlib
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stub src.main.run_pipeline BEFORE loading app.py (avoids heavy deps)
if "src" not in sys.modules:
    sys.modules["src"] = types.ModuleType("src")
if "src.main" not in sys.modules:
    fake_main = types.ModuleType("src.main")
    fake_main.run_pipeline = lambda: None
    sys.modules["src.main"] = fake_main
    sys.modules["src"].main = fake_main

def load_app_module():
    candidates = [ROOT/"app.py", ROOT/"src"/"app.py", ROOT/"api"/"app.py"]
    if not any(p.exists() for p in candidates):
        candidates += list(ROOT.glob("*/app.py"))
    for path in candidates:
        if path.exists():
            spec = spec_from_file_location("app", str(path))
            mod = module_from_spec(spec)
            sys.modules["app"] = mod
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore
            return mod
    raise ModuleNotFoundError("Could not find app.py")

def make_fake_conn():
    class FakeCursor:
        def __init__(self): self._rows=[]
        def execute(self, sql, params=None):
            sql = " ".join(sql.split()); params = params or ()
            if "SELECT MAX(t.trend_id)" in sql and "FROM trend_signal_output t" in sql:
                self._rows = [{"max_trend_id":3,"total_rows":3,"max_group_id":10}]; return
            if "SELECT COALESCE(MAX(group_id), 0)" in sql and "FROM prediction_grid" in sql:
                self._rows = [{"grid_max_group_id":10}]; return
            if "FROM trend_signal_output t LEFT JOIN prediction_grid g" in sql:
                if "WHERE t.trend_id = %s" in sql:
                    trend_id = params[0]
                    if trend_id == 42:
                        self._rows = [{
                            "trend_id":42,"group_id":7,"headline":"Test Headline","tldr":"Short summary",
                            "trend_tags_json":json.dumps(["AI","Finance"]),
                            "industry_tags_json":json.dumps(["AI/ML"]),
                            "cross_platform":True,
                            "platform_spread_json":json.dumps({"reddit":5,"polymarket":2}),
                            "historical_analogues_json":json.dumps(["Dotcom 1999"]),
                            "quotes_citations_json":json.dumps(["source:example"]),
                            "narrative_analysis":"Looks bullish","confidence_score":78,
                            "confidence_score_explanation":"Good overlap + momentum",
                            "watch_flag":"watch","watch_rationale":"Rising mentions",
                            "created_llm_model":"gemini-2.5-pro",
                            "outlook":"Moderate Growth","why":"Cross-signal alignment",
                            "break_point_alerts":"If BTC < 50k, reassess",
                        }]
                    else:
                        self._rows = []
                    return
                base = {
                    "trend_id":101,"group_id":7,"headline":"Row 1","tldr":"Row 1 TLDR",
                    "trend_tags_json":json.dumps(["AI"]),
                    "industry_tags_json":json.dumps(["AI/ML"]),
                    "cross_platform":True,
                    "platform_spread_json":json.dumps({"reddit":3}),
                    "historical_analogues_json":json.dumps([]),
                    "quotes_citations_json":json.dumps([]),
                    "narrative_analysis":None,"confidence_score":55,
                    "confidence_score_explanation":None,"watch_flag":"watch",
                    "watch_rationale":None,"created_llm_model":None,
                    "outlook":"Moderate Growth","why":None,"break_point_alerts":None,
                }
                row2 = {**base,"trend_id":102,"headline":"Row 2","tldr":"Row 2 TLDR","confidence_score":88,"watch_flag":"none"}
                if "WHERE t.group_id = %s" in sql:
                    group_id = params[0]; self._rows = [row2, base] if group_id == 7 else []; return
                self._rows = [row2, base]; return
            self._rows = []
        def fetchone(self): return self._rows[0] if self._rows else None
        def fetchall(self): return list(self._rows)
        def close(self): pass
    class FakeConn:
        def cursor(self, dictionary=True): return FakeCursor()
        def ping(self, reconnect=True, attempts=1, delay=0): return True
        def close(self): pass
    return FakeConn()

def test_api_smoke(monkeypatch):
    os.environ["RUN_SCHEDULE_IN_API"] = "0"
    app_module = load_app_module()
    monkeypatch.setattr(app_module, "get_conn", lambda: make_fake_conn())
    monkeypatch.setattr(app_module, "_safe_run_pipeline", lambda: None)

    from fastapi.testclient import TestClient
    client = TestClient(app_module.app)

    assert client.get("/health").json() == {"ok": True}
    assert isinstance(client.get("/meta").json().get("version"), str)

    items = client.get("/feed?limit=2").json()
    assert len(items) == 2 and isinstance(items[0]["trend_tags"], list)
    assert isinstance(items[0]["platform_spread"], dict)
    assert "trend_tags_json" not in items[0]

    assert client.get("/trends/42").json()["outlook"] == "Moderate Growth"

    grp = client.get("/groups/7").json()
    assert len(grp) == 2 and all(r["group_id"] == 7 for r in grp)

    assert client.post("/run-pipeline", json={}).json() == {"status":"queued"}

if __name__ == "__main__":
    import pytest; raise SystemExit(pytest.main([__file__]))
