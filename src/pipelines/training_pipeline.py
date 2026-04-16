import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.data.load_data import load_user_features_in_batches
from src.data.preprocess import preprocess_batch
from src.models.model_manager import set_model
from src.models.model_persistence import save_model
from src.models.train_model import train_clustering_model

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_PATH: str = os.getenv("MODEL_PATH", "models/clustering_model.pkl")
MIN_USERS_FOR_TRAINING: int = int(os.getenv("TRAINING_MIN_USERS", "50"))
MAX_USERS_FOR_TRAINING: int = int(os.getenv("TRAINING_MAX_USERS", "50000"))


def run_training_pipeline(
    batch_size: int = 1000,
) -> None:
    """
    Runs the full training pipeline.

    Streams data from MongoDB, preprocesses, trains a new Pipeline, saves
    it to disk, and updates the in-memory singleton.

    """
    start_time = datetime.now(timezone.utc)
    logger.info("Training pipeline: starting at %s", start_time.isoformat())

    all_data = []

    try:
        # Stream from MongoDB. Stop early if we hit the user cap.
        for batch in load_user_features_in_batches(batch_size=batch_size):
            all_data.extend(batch)

            if len(all_data) >= MAX_USERS_FOR_TRAINING:
                logger.info(
                    "Training pipeline: reached MAX_USERS_FOR_TRAINING=%d — "
                    "stopping data load early to control memory",
                    MAX_USERS_FOR_TRAINING,
                )
                all_data = all_data[:MAX_USERS_FOR_TRAINING]
                break

        total_users = len(all_data)
        logger.info("Training pipeline: loaded %d users", total_users)

        if total_users < MIN_USERS_FOR_TRAINING:
            logger.warning(
                "Training pipeline: only %d users loaded, need at least %d — "
                "skipping training. Populate the database first.",
                total_users,
                MIN_USERS_FOR_TRAINING,
            )
            return

        # Build feature matrix + log transforms (no scaling — done by Pipeline)
        X, _ = preprocess_batch(all_data)

        if X.shape[0] == 0:
            logger.warning("Training pipeline: empty feature matrix after preprocessing")
            return

        # Train the unified Pipeline (StandardScaler → MiniBatchKMeans)
        model = train_clustering_model(X)

        # Persist to disk first, then update in-memory singleton.
        save_model(model, MODEL_PATH)
        set_model(model)

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(
            "Training pipeline: complete in %.1fs — trained on %d users, "
            "model saved to %s",
            elapsed,
            X.shape[0],
            MODEL_PATH,
        )

    except Exception:
        logger.exception("Training pipeline: unhandled exception")
        raise
