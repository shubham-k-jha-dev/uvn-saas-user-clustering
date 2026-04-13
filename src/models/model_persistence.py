import os
import joblib
import logging

logger = logging.getLogger(__name__)


def save_model(model, path: str):
    """
    Save trained model to disk
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.exception(f"Failed to save model: {e}")


def load_model(path: str):
    """
    Load model from disk
    """
    try:
        if not os.path.exists(path):
            logger.warning("Model file not found")
            return None

        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model

    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        return None