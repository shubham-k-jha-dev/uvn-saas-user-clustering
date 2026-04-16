import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.model_manager import get_model, set_model
from src.models.model_persistence import load_model
from src.pipelines.clustering_pipeline import run_clustering_pipeline
from src.pipelines.training_pipeline import run_training_pipeline

# Configuration
load_dotenv()

MODEL_PATH: str = os.getenv("MODEL_PATH", "models/clustering_model.pkl")
MODEL_VERSION: str = "v1.0.0"
TRAINING_INTERVAL_SECONDS: int = int(os.getenv("TRAINING_INTERVAL_SECONDS", "600"))

# Logging — configured ONCE here. All other modules use getLogger(__name__).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Background training loop
def _background_training_loop() -> None:
    """
    Runs in a daemon thread.
    Sleeps for TRAINING_INTERVAL_SECONDS between runs so the model stays
    fresh without operator intervention.
    Exceptions are caught and logged so a single failure never kills the thread.
    """
    while True:
        try:
            logger.info("Background training: starting run")
            run_training_pipeline()
            logger.info("Background training: run complete")
        except Exception:
            logger.exception("Background training failed — will retry next interval")
        time.sleep(TRAINING_INTERVAL_SECONDS)


# Application lifespan (replaces deprecated @app.on_event("startup"))
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Called once at startup before any request is served.
    """
    model = load_model(MODEL_PATH)
    if model is not None:
        set_model(model)
        logger.info("Startup: Pipeline loaded from disk into memory")
    else:
        logger.warning(
            "Startup: No persisted model found at %s — "
            "call POST /train-model before serving /cluster-users",
            MODEL_PATH,
        )

    thread = threading.Thread(target=_background_training_loop, daemon=True)
    thread.start()
    logger.info("Startup: background training thread started (interval=%ds)", TRAINING_INTERVAL_SECONDS)

    yield  # application runs here

    logger.info("Shutdown: application stopping")


# FastAPI application
app = FastAPI(
    title="User Clustering Service",
    version="v1",
    description="AI-powered user segmentation and customer intelligence API",
    lifespan=lifespan,
)


# Request / Response schemas (Pydantic)
class FeatureSummary(BaseModel):
    sessionCount: float = Field(..., ge=0, description="Total sessions in observation window")
    avgSessionTime: float = Field(..., ge=0, description="Average session duration in minutes")
    pagesPerSession: float = Field(..., ge=0, description="Average page views per session")
    clicksPerSession: float = Field(..., ge=0, description="Average clicks per session")
    bounceRate: float = Field(..., ge=0, le=1, description="Fraction of single-page sessions [0, 1]")
    recencyDays: float = Field(..., ge=0, description="Days since last session")


class UserInput(BaseModel):
    userId: str = Field(..., description="Unique user identifier")
    tenantId: str = Field(..., description="Tenant / app identifier")
    featureSummary: FeatureSummary


class BatchRequest(BaseModel):
    users: List[UserInput] = Field(..., min_length=1, description="At least one user required")


# FastAPI Dependency — model guard
def get_loaded_model():
    """
    FastAPI dependency injected into any endpoint that needs the Pipeline.
    Returns the Pipeline if loaded; raises HTTP 503 otherwise.
    """
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not ready. Call POST /train-model first.",
        )
    return model


# Endpoints

@app.get("/health", tags=["ops"])
def health() -> dict:
    """
    Liveness probe.
    Returns 200 as long as the process is running.
    Do NOT check model state here — a load balancer should only use this
    to know whether to restart the container, not whether to send traffic.
    """
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready() -> dict:
    """
    Readiness probe.
    Returns 200 only after the ML Pipeline has been loaded into memory.
    Use this as the Kubernetes readinessProbe or ALB health-check target
    so traffic is only routed to instances that can actually serve predictions.
    """
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ready", "modelVersion": MODEL_VERSION}


@app.get("/status", tags=["ops"])
def status() -> dict:
    """
    Lightweight metadata about the running service.
    Useful for dashboards and debugging without exposing internal state.
    """
    model = get_model()
    return {
        "status": "running",
        "modelLoaded": model is not None,
        "modelVersion": MODEL_VERSION,
        "trainingIntervalSeconds": TRAINING_INTERVAL_SECONDS,
    }


@app.post("/train-model", tags=["training"])
def train_model() -> dict:
    """
    Triggers a synchronous training run.
    Blocks until training completes — suitable for manual retraining.
    The background thread continues its own schedule independently.
    """
    try:
        run_training_pipeline()
        return {"success": True, "message": "Model trained and cached successfully"}
    except Exception:
        logger.exception("POST /train-model failed")
        raise HTTPException(status_code=500, detail="Training failed — check server logs")


@app.post("/cluster-users", tags=["inference"])
def cluster_users(
    request: BatchRequest,
    model=Depends(get_loaded_model),  # 503 if model not ready
) -> dict:
    """
    Runs the full inference pipeline:
    The `model` dependency ensures we never hit this code path with an
    unloaded model — the dependency raises 503 before we get here.
    """
    try:
        logger.info("POST /cluster-users — received %d users", len(request.users))

        # Convert Pydantic models to plain dicts for the pipeline layer.
        input_data = [user.model_dump() for user in request.users]

        results = run_clustering_pipeline(input_data)

        logger.info("POST /cluster-users — processed %d users successfully", len(results))

        return {
            "success": True,
            "count": len(results),
            "meta": {
                "modelVersion": MODEL_VERSION,
                "processedAt": datetime.now(timezone.utc).isoformat(),
            },
            "data": results,
        }

    except ValueError as ve:
        logger.warning("POST /cluster-users — validation error: %s", ve)
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception:
        logger.exception("POST /cluster-users — internal error")
        raise HTTPException(status_code=500, detail="Internal server error")


# Dev runner — not used by uvicorn in production

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)