import logging
from typing import List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "sessionCount",
    "avgSessionTime",
    "pagesPerSession",
    "clicksPerSession",
    "bounceRate",
    "recencyDays"
]

DEFAULT_VALUES = {
    "sessionCount": 0,
    "avgSessionTime": 0,
    "pagesPerSession": 0,
    "clicksPerSession": 0,
    "bounceRate": 1,
    "recencyDays": 999
}


def get_feature_value(feature_dict: Dict[str, Any], key: str) -> float:
    value = feature_dict.get(key, DEFAULT_VALUES[key])

    if value is None:
        return DEFAULT_VALUES[key]

    try:
        value = float(value)
    except Exception:
        return DEFAULT_VALUES[key]

    # prevent negative weird values
    return max(value, 0)


# Build feature matrix
def build_feature_matrix(
    batch: List[Dict[str, Any]]
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Convert Mongo documents → feature matrix

    Returns:
        X: numpy array
        metadata: list of user info
    """

    features = []
    metadata = []

    for doc in batch:
        fs = doc.get("featureSummary", {})

        row = [
            get_feature_value(fs, "sessionCount"),
            get_feature_value(fs, "avgSessionTime"),
            get_feature_value(fs, "pagesPerSession"),
            get_feature_value(fs, "clicksPerSession"),
            get_feature_value(fs, "bounceRate"),
            get_feature_value(fs, "recencyDays"),
        ]

        features.append(row)

        metadata.append({
            "userId": doc.get("userId"),
            "tenantId": doc.get("tenantId"),
            "_id": str(doc.get("_id"))
        })

    X = np.array(features, dtype=np.float64)

    logger.info(f"Built feature matrix with shape: {X.shape}")

    return X, metadata

def apply_log_transform(X: np.ndarray) -> np.ndarray:
    """
    Apply log1p to skewed columns only.
    Does NOT scale — scaling is done by the sklearn Pipeline.
    """

    if X.shape[0] == 0:
        logger.warning("Empty feature matrix received")
        return X

    X_transformed = X.copy()

    # log1p compresses outliers on high-cardinality columns
    X_transformed[:, 0] = np.log1p(X_transformed[:, 0])  # sessionCount
    X_transformed[:, 1] = np.log1p(X_transformed[:, 1])  # avgSessionTime
    X_transformed[:, 3] = np.log1p(X_transformed[:, 3])  # clicksPerSession

    logger.info("Log transform applied")

    return X_transformed


def preprocess_batch(
    batch: List[Dict[str, Any]]
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Builds the raw feature matrix with log transforms applied.
    StandardScaler is applied externally by the sklearn Pipeline
    so the same mean/std from training is reused during inference.
    """

    X, metadata = build_feature_matrix(batch)
    X = apply_log_transform(X)

    return X, metadata