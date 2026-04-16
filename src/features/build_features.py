import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

_IDX = {
    "sessionCount": 0,
    "avgSessionTime": 1,
    "pagesPerSession": 2,
    "clicksPerSession": 3,
    "bounceRate": 4,
    "recencyDays": 5,
}

# Weights for engagement scoring.
# Positive: more is better. Negative: more is worse.
# Weights sum to ~1.0 for interpretability.
_WEIGHTS = {
    "sessionCount": 0.25,
    "avgSessionTime": 0.20,
    "pagesPerSession": 0.20,
    "clicksPerSession": 0.20,
    "bounceRate": -0.10,   # high bounce = low engagement
    "recencyDays": -0.05,  # stale user = lower engagement
}


def score_centroid(center: np.ndarray) -> float:
    """
    Compute a single engagement score for a cluster centroid.
    """
    score = 0.0
    for feature, weight in _WEIGHTS.items():
        score += center[_IDX[feature]] * weight
    return score


def label_clusters(pipeline) -> Dict[int, str]:
    """
    Assign a semantic label to every cluster based on relative engagement rank.

    """
    kmeans = pipeline.named_steps["kmeans"]
    centroids = kmeans.cluster_centers_
    k = len(centroids)

    # Score each centroid
    scored = [(cluster_id, score_centroid(center)) for cluster_id, center in enumerate(centroids)]

    # Sort descending: rank 0 is the most engaged cluster
    scored.sort(key=lambda x: x[1], reverse=True)

    label_map: Dict[int, str] = {}

    for rank, (cluster_id, score) in enumerate(scored):
        if k == 1:
            # Only one cluster — no meaningful differentiation
            label = "casual_user"
        elif rank == 0:
            label = "high_value_user"
        elif rank == k - 1:
            # Always label the absolute worst cluster as drop_off,
            # regardless of K. This guarantees drop_off_user is reachable.
            label = "drop_off_user"
        elif rank == 1:
            label = "engaged_user"
        else:
            # Any middle cluster (rank 2..K-2) is casual
            label = "casual_user"

        label_map[cluster_id] = label
        logger.debug("Cluster %d → rank %d → %s (score=%.4f)", cluster_id, rank, label, score)

    logger.info("label_clusters: K=%d, mapping=%s", k, label_map)
    return label_map


def attach_cluster_labels(
    cluster_results: List[Dict],
    cluster_labels: Dict[int, str],
) -> List[Dict]:
    """
    Joins cluster prediction results with the label map.
    """
    enriched = []
    for item in cluster_results:
        cluster_id = item["clusterId"]
        label = cluster_labels.get(cluster_id, "unknown")
        enriched.append({**item, "clusterLabel": label})
    return enriched