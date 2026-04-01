import logging
from typing import Dict, List

import numpy as np


logger = logging.getLogger(__name__)

IDX = {
    "sessionCount": 0,
    "avgSessionTime": 1,
    "pagesPerSession": 2,
    "clicksPerSession": 3,
    "bounceRate": 4,
    "recencyDays": 5
}


def score_cluster(center: np.ndarray) -> float:
    """
    Compute engagement score for cluster
    Higher = better user
    """

    score = 0

    # positive signals
    score += center[IDX["sessionCount"]]
    score += center[IDX["avgSessionTime"]]
    score += center[IDX["clicksPerSession"]]
    score += center[IDX["pagesPerSession"]]

    # negative signals
    score -= center[IDX["bounceRate"]]
    score -= center[IDX["recencyDays"]]

    return score

def label_clusters(model) -> Dict[int, str]:
    """
    Assign business labels to clusters
    """

    centers = model.cluster_centers_

    cluster_scores = []

    for i, center in enumerate(centers):
        score = score_cluster(center)
        cluster_scores.append((i, score))

    # sort by score descending
    cluster_scores.sort(key=lambda x: x[1], reverse=True)

    labels = {}

    for rank, (cluster_id, _) in enumerate(cluster_scores):

        if rank == 0:
            labels[cluster_id] = "high_value_user"
        elif rank == 1:
            labels[cluster_id] = "engaged_user"
        elif rank == 2:
            labels[cluster_id] = "casual_user"
        else:
            labels[cluster_id] = "drop_off_user"

    logger.info(f"Cluster labeling completed: {labels}")

    return labels


def attach_cluster_labels(
    cluster_results: List[Dict],
    cluster_labels: Dict[int, str]
) -> List[Dict]:
    """
    Add labels to user cluster results
    """

    enriched = []

    for item in cluster_results:
        cluster_id = item["clusterId"]

        enriched.append({
            **item,
            "clusterLabel": cluster_labels.get(cluster_id, "unknown")
        })

    return enriched