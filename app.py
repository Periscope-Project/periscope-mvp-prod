from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json, pathlib
from typing import List
import os

os.chdir("C:/Users/Avika/OneDrive - Hogarth Worldwide/Documents/Work/Periscope/early-mvp")

app = FastAPI()

# ---- (optional) allow any origin so the browser won’t block requests ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # be stricter in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_FILE = pathlib.Path(__file__).with_name("all_trends_07_08_25.json")

def load_posts() -> List[dict]:
    """
    Returns a list of {"id": int, "headline": str, "tldr": str}
    extracted from the JSON.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found")

    raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    trends = raw.get("trends", [])
    posts = [t for t in trends if t.get("headline") and t.get("tldr")]

    return posts

@app.get("/feed")
def get_feed():
    try:
        return {"posts": load_posts()}
    except Exception as e:
        # FastAPI will automatically turn this into JSON + 500 status
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="::", port=8000, log_level="info")
    
    #uvicorn fast_api:app --reload --host :: --port 8000


    #http://[::1]:8000/feed