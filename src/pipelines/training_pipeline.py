import logging
from datetime import datetime
from src.data.load_data import load_user_features_in_batches
from src.data.preprocess import preprocess_batch
from src.models.train_model import train_clustering_model
from src.models.model_manager import set_model
from src.models.model_persistence import save_model
import os
from dotenv import load_dotenv

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")

logger = logging.getLogger(__name__)


def run_training_pipeline(batch_size: int = 1000, min_users: int = 50):
    """
    Automated training pipeline:
    - Loads data from MongoDB
    - Trains clustering model
    - Caches model in memory
    """

    logger.info("Starting automated training job")

    all_data = []

    try:
        # Load data
        for batch in load_user_features_in_batches(batch_size=batch_size):
            all_data.extend(batch)

        total_users = len(all_data)

        logger.info(f"Total users loaded: {total_users}")

        if total_users < min_users:
            logger.warning("Not enough data to train model")
            return

        # Preprocess
        X, _ = preprocess_batch(all_data)

        if X.shape[0] == 0:
            logger.warning("No valid data after preprocessing")
            return

        # Train model
        model = train_clustering_model(X)
        save_model(model, MODEL_PATH)

        # Cache model
        set_model(model)

        logger.info(
            f"Training job completed successfully at {datetime.utcnow().isoformat()} "
            f"on {X.shape[0]} users"
        )

    except Exception as e:
        logger.exception(f"Training job failed: {e}")
        
        
