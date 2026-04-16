import logging
import threading
from typing import Optional

from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Module-level state — one Pipeline for the entire process lifetime.
_model: Optional[Pipeline] = None
_model_lock = threading.Lock()


def get_model() -> Optional[Pipeline]:
    """
    Returns the currently cached Pipeline, or None if not yet trained.
    Thread-safe: reads are atomic on CPython due to the GIL.
    """
    return _model


def set_model(model: Pipeline) -> None:
    """
    Atomically replaces the cached Pipeline.
    Acquiring the lock ensures that concurrent set_model calls from multiple
    threads (unlikely but possible if two training jobs run simultaneously)
    do not interleave.
    """
    global _model
    with _model_lock:
        _model = model
    logger.info("Model manager: new Pipeline cached in memory")