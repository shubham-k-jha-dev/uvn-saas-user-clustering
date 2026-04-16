import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def aggregate_user_sessions(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collapses multiple session/feature documents into one per-user feature dict.
    """
    # Build a per-userId accumulator map
    user_map: Dict[str, Dict[str, Any]] = {}

    for doc in docs:
        user_id = doc.get("userId")
        if not user_id:
            logger.warning("user_aggregator: skipping doc with missing userId")
            continue

        fs = doc.get("featureSummary", {})
        doc_sessions = max(float(fs.get("sessionCount", 1)), 1.0)

        if user_id not in user_map:
            user_map[user_id] = {
                "userId": user_id,
                "tenantId": doc.get("tenantId"),
                "totalSessions": 0.0,
                "totalSessionTime": 0.0,   
                "totalPages": 0.0,         
                "totalClicks": 0.0,        
                "totalBounceWeight": 0.0,  
                "recencyDays": float(fs.get("recencyDays", 0)),
            }

        acc = user_map[user_id]

        acc["totalSessions"] += doc_sessions
        acc["totalSessionTime"] += fs.get("avgSessionTime", 0.0) * doc_sessions
        acc["totalPages"] += fs.get("pagesPerSession", 0.0) * doc_sessions
        acc["totalClicks"] += fs.get("clicksPerSession", 0.0) * doc_sessions
        acc["totalBounceWeight"] += fs.get("bounceRate", 0.0) * doc_sessions

        acc["recencyDays"] = min(acc["recencyDays"], float(fs.get("recencyDays", 0)))

    aggregated: List[Dict[str, Any]] = []

    for acc in user_map.values():
        total_sessions = max(acc["totalSessions"], 1.0)  

        aggregated.append({
            "userId": acc["userId"],
            "tenantId": acc["tenantId"],
            "featureSummary": {
                "sessionCount": total_sessions,
                "avgSessionTime": acc["totalSessionTime"] / total_sessions,
                "pagesPerSession": acc["totalPages"] / total_sessions,
                "clicksPerSession": acc["totalClicks"] / total_sessions,
                "bounceRate": acc["totalBounceWeight"] / total_sessions,
                "recencyDays": acc["recencyDays"],
            },
        })

    logger.info(
        "user_aggregator: %d input docs → %d unique users",
        len(docs),
        len(aggregated),
    )
    return aggregated