import logging
from src.pipelines.clustering_pipeline import run_clustering_pipeline
from src.data.load_data import load_user_features_in_batches
from datetime import datetime, timezone
from src.models.model_manager import set_model
from src.models.train_model import train_clustering_model
from src.data.preprocess import preprocess_batch
from src.models.model_manager import get_model
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.pipelines.clustering_pipeline import run_clustering_pipeline
from src.pipelines.training_pipeline import run_training_pipeline
import threading
import time
from src.models.model_persistence import load_model
from src.models.model_manager import set_model
import os
from dotenv import load_dotenv
from src.models.model_manager import get_model

MODEL_VERSION = "v1.0.0"

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """
    Entry point for running clustering pipelines
    """

    logger.info("Starting main pipelines execution")

    all_data = []


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="User Clustering Service",
    version="v1",
    description="AI-powered user segmentation and customer intelligence API"
)


class FeatureSummary(BaseModel):
    sessionCount: float = Field(..., ge=0)
    avgSessionTime: float = Field(..., ge=0)
    pagesPerSession: float = Field(..., ge=0)
    clicksPerSession: float = Field(..., ge=0)
    bounceRate: float = Field(..., ge=0, le=1)
    recencyDays: float = Field(..., ge=0)


class UserInput(BaseModel):
    userId: str
    tenantId: str
    featureSummary: FeatureSummary


class BatchRequest(BaseModel):
    users: List[UserInput]

@app.on_event("startup")
def start_background_jobs():

    # Load model from disk FIRST
    model = load_model(MODEL_PATH)
    if model:
        set_model(model)
        logger.info("Model loaded into memory on startup")

    # Start training loop
    thread = threading.Thread(target=background_training_loop, daemon=True)
    thread.start()

    logger.info("Background training job started")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/train-model")
def train_model_endpoint(request: BatchRequest):
    """
    Train clustering model using batch data
    """

    try:
        logger.info(f"Training request received with {len(request.users)} users")

        input_data = [user.dict() for user in request.users]

        # preprocess
        X, _ = preprocess_batch(input_data)

        if X.shape[0] < 10:
            raise HTTPException(
                status_code=400,
                detail="At least 10 users required to train model"
            )

        # train model
        model = train_clustering_model(X)

        # cache model
        set_model(model)

        logger.info("Model trained and cached successfully")

        return {
            "success": True,
            "message": "Model trained successfully",
            "trainedOn": X.shape[0]
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail="Training failed")
    
    
def background_training_loop():
    """
    Runs training job every fixed interval
    """

    while True:
        try:
            run_training_pipeline()
        except Exception as e:
            logger.exception(f"Background training failed: {e}")

        # run every 10 minutes (configurable)
        time.sleep(600)


@app.get("/status")
def system_status():
    model = get_model()

    return {
        "status": "running",
        "modelLoaded": model is not None
    }
    
       
@app.post("/cluster-users")
def cluster_users(request: BatchRequest):
    """
    Cluster users and return customer intelligence
    """

    try:
        logger.info(f"Received request with {len(request.users)} users")

        # convert Pydantic models → dict
        input_data = [user.dict() for user in request.users]

        # run pipelines
        model = get_model()

        if model is None:
            raise HTTPException(
                status_code=400,
                detail="Model not trained yet. Call /train-model first."
            )

        results = run_clustering_pipeline(input_data)

        logger.info(f"Processed {len(results)} users successfully")

        return {
            "success": True,
            "count": len(results),
            "meta": {
                "modelVersion": MODEL_VERSION,
                "processedAt": datetime.now(timezone.utc).isoformat()
            },
            "data": results
        }

    except ValueError as ve:
        logger.warning(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")
    for batch in load_user_features_in_batches(batch_size=1000):
        all_data.extend(batch)

    if not all_data:
        logger.warning("No data found. Exiting.")
        return

    results = run_clustering_pipeline(all_data)

    print("\nFINAL RESULTS:")
    for r in results:
        print(r)


if __name__ == "__main__":  
    main()