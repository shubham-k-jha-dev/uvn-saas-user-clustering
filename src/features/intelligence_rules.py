# src/features/intelligence_rules.py

from typing import Dict


# cluster_id -> intelligence attributes

CLUSTER_INTELLIGENCE_MAP: Dict[str, Dict[str, str]] = {
    "high_value_user": {
        "engagementLevel": "high",
        "churnRisk": "low",
        "action": "Upsell premium features"
    },
    "engaged_user": {
        "engagementLevel": "medium",
        "churnRisk": "low",
        "action": "Encourage conversion"
    },
    "casual_user": {
        "engagementLevel": "medium",
        "churnRisk": "medium",
        "action": "Increase engagement via notifications"
    },
    "drop_off_user": {
        "engagementLevel": "low",
        "churnRisk": "high",
        "action": "Send re-engagement campaign"
    }
}


DEFAULT_INTELLIGENCE = {
    "engagementLevel": "unknown",
    "churnRisk": "unknown",
    "action": "No action available"
}