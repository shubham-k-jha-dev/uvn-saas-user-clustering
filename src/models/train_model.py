import logging
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)

# CONSTANTS
MIN_CLUSTERS = 2
MAX_CLUSTERS = 6
BATCH_SIZE = 1024
RANDOM_STATE = 42


# optimal k
def find_optimal_k(X: np.ndarray) -> int:
    """
    Find optimal number of clusters using elbow method
    """

    if X.shape[0] < MIN_CLUSTERS:
        logger.warning("Too few samples, defaulting to 1 cluster")
        return 1

    inertias = []

    k_range = range(MIN_CLUSTERS, min(MAX_CLUSTERS, X.shape[0]) + 1)

    for k in k_range:
        model = MiniBatchKMeans(
            n_clusters=k,
            batch_size=BATCH_SIZE,
            random_state=RANDOM_STATE
        )
        model.fit(X)
        inertias.append(model.inertia_)

    # simple elbow: pick k where drop slows
    optimal_k = k_range[0]

    for i in range(1, len(inertias)):
        drop_prev = inertias[i - 1] - inertias[i]
        drop_next = inertias[i] - inertias[i + 1] if i + 1 < len(inertias) else 0

        if drop_next < drop_prev * 0.5:
            optimal_k = k_range[i]
            break

    logger.info(f"Optimal clusters selected: {optimal_k}")
    return optimal_k


# train model
def train_clustering_model(X: np.ndarray) -> MiniBatchKMeans:
    """
    Train MiniBatchKMeans model
    """

    if X.shape[0] == 0:
        raise ValueError("Empty dataset provided to model")

    k = find_optimal_k(X)

    model = MiniBatchKMeans(
        n_clusters=k,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE
    )

    model.fit(X)

    logger.info("Clustering model trained successfully")

    return model


# predict clusters
def predict_clusters(
    model: MiniBatchKMeans,
    X: np.ndarray,
    metadata: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Assign cluster IDs to users
    """

    if X.shape[0] == 0:
        return []

    cluster_ids = model.predict(X)

    results = []

    for i, cluster_id in enumerate(cluster_ids):
        results.append({
            "userId": metadata[i]["userId"],
            "tenantId": metadata[i]["tenantId"],
            "clusterId": int(cluster_id)
        })

    logger.info(f"Assigned clusters to {len(results)} users")

    return results