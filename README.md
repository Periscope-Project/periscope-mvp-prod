# Periscope MVP – README

## 🚀 Overview
Periscope is a pipeline + API for surfacing emerging trends by combining:
- **Reddit posts** (via `get_reddit.py`)  
- **Polymarket prediction markets** (via `get_polymarket.py`)  
- **Topic modelling** (via `topic_modelling.py`, BERTopic + embeddings)  
- **Enrichment + features** (via `enrich_data.py`, `polymarket_features.py`)  
- **FastAPI web service** (`app.py`)  

Outputs are JSON/NDJSON trend briefs and optional MySQL storage.

---

## 🛠️ Requirements
- Python **3.12**
- Virtualenv or Conda recommended
- MySQL 8.x (if `PUSH_TO_SQL=true`)
- Reddit + Polymarket credentials if fetching data
- LiteLLM / Gemini API key if summarizing

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration
Settings are loaded from `.env` or `config.ini`.  
Example:

```ini
[CONFIG]
RUN_SCHEDULE=false
RUN_SCHEDULE_IN_API=false
SCHEDULE_TZ=Europe/London
SCHEDULE_HOUR=2
SCHEDULE_MINUTE=0

SKIP_GET_REDDIT=1
SKIP_POLY_FETCH=1

[Reddit]
REDDIT_ID=your_id
REDDIT_SECRET=your_secret

[LiteLLM]
LITELLM_API_KEY=your_key
LITELLM_LOCATION=vertex_ai/gemini-2.5-pro

[SQL]
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=yourpass
MYSQL_DB=periscope
PUSH_TO_SQL=false
```

💡 **Tips**
- If `SKIP_GET_REDDIT=1` or `SKIP_POLY_FETCH=1`, the pipeline will skip those fetches.  
- Reddit supports multiple accounts (`REDDIT_ID_AVIKA`, `REDDIT_SECRET_AVIKA`, etc.).  
- MySQL is only used if `PUSH_TO_SQL=true`.

---

## 🧪 Running Tests
Run all tests with pytest:
```bash
pytest -q
```

---

## ▶️ Running the API
Start the FastAPI app:
```bash
python app.py
```

By default it binds to `localhost:8000`.  
Docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🌀 Running the Pipeline Manually
Instead of waiting for the scheduler, you can run the pipeline directly:
```bash
python src/main.py
```

This will:
1. Fetch Reddit + Polymarket (unless skipped).
2. Run topic modelling + enrichment.
3. Summarize with LiteLLM.
4. Write outputs to `public/files/` (JSON, NDJSON, CSV).
5. Optionally push to SQL.

---

## 📦 Output
- `public/files/` → trend JSON, NDJSON, CSV.  
- Example:  
  - `all_trends_YYYY_MM_DD.json`  
  - `reddit_daily_all_YYYY-MM-DD.ndjson`  
  - `polymarket_live_YYYY-MM-DD.json`

---

## 💾 Database
If `PUSH_TO_SQL=true`, trends are inserted into MySQL.  
Schema is defined in `database_utils/data_to_sql.py`.

---

## 🔑 Credentials
- Store secrets in `.env` or pass via environment variables.  
- Never hardcode keys in code.

#TODO drop dupes both reddit and polymarket
#TODO litellm bit
#TODO main pipeline and schedulling
#TODO Sql for align
#TODO check file output paths
#FIXME trendid in sql
#TODO