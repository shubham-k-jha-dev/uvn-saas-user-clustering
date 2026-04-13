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

def label_clusters(model) -> dict:
    """
    Assign meaningful labels to clusters based on cluster centers
    """

    import numpy as np

    centers = model.cluster_centers_

    cluster_labels = {}

    for i, center in enumerate(centers):

        session_count = center[0]
        avg_time = center[1]
        pages = center[2]
        clicks = center[3]
        bounce = center[4]
        recency = center[5]

        # High value users
        if session_count > 20 and avg_time > 10 and bounce < 0.3:
            label = "high_value_user"

        # Drop-off users
        elif session_count < 3 and bounce > 0.7 and recency > 10:
            label = "drop_off_user"

        # Engaged users
        elif session_count > 10 and bounce < 0.5:
            label = "engaged_user"

        # Default
        else:
            label = "casual_user"

        cluster_labels[i] = label

    return cluster_labels


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