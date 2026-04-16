import logging
from typing import List, Dict, Any

from src.features.intelligence_rules import (
    CLUSTER_INTELLIGENCE_MAP,
    DEFAULT_INTELLIGENCE
)

logger = logging.getLogger(__name__)


def enrich_with_customer_intelligence(
    clustered_users: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Returns:
        List of users with added intelligence fields
    """

    if not isinstance(clustered_users, list):
        raise ValueError("clustered_users must be a list")

    enriched_results: List[Dict[str, Any]] = []

    for idx, user in enumerate(clustered_users):

        try:
            cluster_label = user.get("clusterLabel")

            if not cluster_label:
                logger.warning(f"Missing clusterLabel for user index {idx}")
                intelligence = DEFAULT_INTELLIGENCE
            else:
                intelligence = CLUSTER_INTELLIGENCE_MAP.get(
                    cluster_label,
                    DEFAULT_INTELLIGENCE
                )

            enriched_user = {
                **user,
                "customerIntelligence": {
                    "engagementLevel": intelligence["engagementLevel"],
                    "churnRisk": intelligence["churnRisk"],
                    "recommendedAction": intelligence["action"]
                }
            }

            enriched_results.append(enriched_user)

        except Exception as e:
            logger.error(f"Error processing user index {idx}: {e}")
            continue

    logger.info(f"Customer intelligence added for {len(enriched_results)} users")

    return enriched_results

def override_high_value_user(user, current_label):
    """
    Rule-based override for strong signals
    """

    fs = user["featureSummary"]

    if (
        fs["sessionCount"] > 50 and
        fs["avgSessionTime"] > 20 and
        fs["pagesPerSession"] > 10 and
        fs["bounceRate"] < 0.2 and
        fs["recencyDays"] <= 2
    ):
        return "high_value_user"

    return current_label