import logging
from typing import List, Dict, Any
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# CONSTANTS
MIN_CLUSTERS = 2
MAX_CLUSTERS = 6
BATCH_SIZE = 1024
RANDOM_STATE = 42


def find_optimal_k(X: np.ndarray) -> int:
    """
    Find optimal number of clusters using silhouette score + inertia fallback
    Production-grade selection
    """

    n_samples = X.shape[0]

    if n_samples < MIN_CLUSTERS:
        logger.warning("Too few samples, defaulting to 1 cluster")
        return 1

    # Bound k properly
    max_k = min(MAX_CLUSTERS, n_samples - 1)

    best_k = MIN_CLUSTERS
    best_score = -1

    scores = []

    for k in range(MIN_CLUSTERS, max_k + 1):
        try:
            model = MiniBatchKMeans(
                n_clusters=k,
                batch_size=BATCH_SIZE,
                random_state=RANDOM_STATE
            )

            labels = model.fit_predict(X)

            # silhouette requires >1 cluster and no empty clusters
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                scores.append((k, score))

                if score > best_score:
                    best_score = score
                    best_k = k

        except Exception as e:
            logger.warning(f"Skipping k={k} due to error: {e}")
            continue

    # fallback if silhouette fails
    if best_score == -1:
        logger.warning("Silhouette failed, falling back to inertia method")

        inertias = []
        k_range = range(MIN_CLUSTERS, max_k + 1)

        for k in k_range:
            model = MiniBatchKMeans(
                n_clusters=k,
                batch_size=BATCH_SIZE,
                random_state=RANDOM_STATE
            )
            model.fit(X)
            inertias.append(model.inertia_)

        best_k = k_range[0]

        for i in range(1, len(inertias) - 1):
            drop_prev = inertias[i - 1] - inertias[i]
            drop_next = inertias[i] - inertias[i + 1]

            if drop_next < drop_prev * 0.5:
                best_k = k_range[i]
                break

    logger.info(f"Optimal clusters selected: {best_k}")
    return best_k


# train model
def train_clustering_model(X: np.ndarray) -> Pipeline:
    """
    Train a unified sklearn Pipeline:
    StandardScaler -> MiniBatchKMeans

    Saving this single Pipeline object guarantees that inference
    uses the exact same scaling parameters as training.
    """

    if X.shape[0] == 0:
        raise ValueError("Empty dataset provided to model")

    k = find_optimal_k(X)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", MiniBatchKMeans(
            n_clusters=k,
            batch_size=BATCH_SIZE,
            random_state=RANDOM_STATE
        ))
    ])

    pipeline.fit(X)

    logger.info(f"Pipeline trained successfully (k={k})")

    return pipeline


# predict clusters
def predict_clusters(
    pipeline: Pipeline,
    X: np.ndarray,
    metadata: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Assign cluster IDs to users.
    X must be the log-transformed (pre-scaling) matrix.
    StandardScaler is applied internally by the Pipeline.
    """

    if X.shape[0] == 0:
        return []

    # Pipeline handles scaling then predicting in one step
    cluster_ids = pipeline.predict(X)

    results = []

    for i, cluster_id in enumerate(cluster_ids):
        results.append({
            "userId": metadata[i]["userId"],
            "tenantId": metadata[i]["tenantId"],
            "clusterId": int(cluster_id)
        })

    logger.info(f"Assigned clusters to {len(results)} users")

    return results