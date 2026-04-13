import logging
from typing import Optional

from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)


_model: Optional[MiniBatchKMeans] = None


# get model
def get_model() -> Optional[MiniBatchKMeans]:
    return _model

# set model
def set_model(model: MiniBatchKMeans) -> None:
    global _model
    _model = model
    logger.info("Model cached in memory")