import logging
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)

# A session lasting this many seconds or more is treated as a non-bounce.
_BOUNCE_DECAY_SECONDS: float = 30.0


def extract_feature_summary(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts raw telemetry JSON into a six-feature ML-ready featureSummary.
    """
    try:
        session = raw.get("session_context", {})
        engagement = raw.get("engagement_metrics", {})
        user = raw.get("user_identity", {})

        session_duration_sec: float = float(session.get("session_duration_sec", 0))
        session_count: int = int(user.get("lifetime_sessions", 1))

        # avgSessionTime: single-session approximation (see module docstring)
        avg_session_time: float = session_duration_sec

        pages_per_session: float = float(engagement.get("pageviews", 1))
        clicks_per_session: float = float(engagement.get("click_count", 0))

        # Continuous bounce rate: decays from 1.0 (0s session) to 0.0 (≥30s).
        # This preserves gradient signal unlike the old binary step function.
        bounce_rate: float = max(0.0, 1.0 - session_duration_sec / _BOUNCE_DECAY_SECONDS)

        # Recency: days since last_seen. Stays 0 if timestamp is missing/invalid.
        recency_days: int = 0
        last_seen = user.get("last_seen")
        if last_seen:
            try:
                last_seen_dt = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
                recency_days = max(0, (datetime.now(timezone.utc) - last_seen_dt).days)
            except ValueError:
                logger.warning("feature_extractor: unparseable last_seen value: %r", last_seen)

        return {
            "sessionCount": session_count,
            "avgSessionTime": avg_session_time,
            "pagesPerSession": pages_per_session,
            "clicksPerSession": clicks_per_session,
            "bounceRate": round(bounce_rate, 4),
            "recencyDays": recency_days,
        }

    except Exception:
        logger.exception("feature_extractor: extraction failed for document")
        return {}