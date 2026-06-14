from fastapi import FastAPI

app = FastAPI(title="Agent Service")


@app.get("/healthz")
async def health():
    return {"status": "ok"}
