from fastapi import FastAPI

app = FastAPI(title="Backend Service", version="0.1.0")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Simple health-check endpoint."""
    return {"status": "ok"} 