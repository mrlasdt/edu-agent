from fastapi import FastAPI
from services.gateway.src.gateway.orchestrator import Orchestrator

app = FastAPI(title="Gateway Service")


@app.get("/healthz")
async def health():
    return {"status": "ok"}
