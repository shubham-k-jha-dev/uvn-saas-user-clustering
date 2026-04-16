import logging
import os
from typing import Any, Dict, Generator, List

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from src.features.feature_extractor import extract_feature_summary

logger = logging.getLogger(__name__)

load_dotenv()


# Configuration — fail fast at module load if required vars are missing

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
USER_FEATURES_COLLECTION = os.getenv("USER_FEATURES_COLLECTION")

if not MONGO_URI or not DB_NAME or not USER_FEATURES_COLLECTION:
    raise EnvironmentError(
        "Missing required environment variables: "
        "MONGO_URI, DB_NAME, USER_FEATURES_COLLECTION. "
        "Check your .env file."
    )

# Default tenant for raw telemetry docs that don't carry tenant context.
_DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "app_001")


# Connection
def get_mongo_client() -> MongoClient:
    """
    Creates and validates a MongoDB client.
    Uses a 5-second timeout so slow connections fail fast rather than hanging.

    Raises:
        ConnectionFailure: if the server is unreachable.
    """
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        logger.info("load_data: MongoDB connection established")
        return client
    except ConnectionFailure:
        logger.exception("load_data: failed to connect to MongoDB at %s", MONGO_URI)
        raise


def get_user_features_collection():
    """Returns the configured MongoDB collection handle."""
    client = get_mongo_client()
    return client[DB_NAME][USER_FEATURES_COLLECTION]


# Document validation (for CASE 2 — raw telemetry)
def _validate_raw_document(doc: Dict[str, Any]) -> bool:
    """
    Returns True if the document has the raw telemetry structure required
    by extract_feature_summary().
    """
    if not isinstance(doc, dict):
        return False
    return all(
        key in doc
        for key in ("user_identity", "session_context", "engagement_metrics")
    )


# Batch loader (generator)

def load_user_features_in_batches(
    batch_size: int = 1000,
) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Streams all documents from MongoDB and yields ML-ready user feature dicts
    in batches of `batch_size`.

    CASE 1 — Document already has 'featureSummary':
      Passed through directly after extracting userId / tenantId.

    CASE 2 — Raw telemetry document:
      Passed through extract_feature_summary() to compute the six features.
      tenantId falls back to DEFAULT_TENANT_ID if not present in the document.

    Documents that match neither case are silently skipped and counted.

    Args:
        batch_size: number of documents per yielded batch.

    Yields:
        List of dicts: [{userId, tenantId, featureSummary}, ...]
    """
    collection = get_user_features_collection()
    total_docs = collection.count_documents({})
    logger.info("load_data: %d documents in collection", total_docs)

    if total_docs == 0:
        logger.warning("load_data: collection is empty — nothing to load")
        return

    cursor = collection.find({}, no_cursor_timeout=True)
    batch: List[Dict[str, Any]] = []
    skipped = 0

    try:
        for doc in cursor:
            # CASE 1: pre-aggregated document
            if "featureSummary" in doc:
                user_id = doc.get("userId")
                if not user_id:
                    skipped += 1
                    continue

                batch.append({
                    "userId": user_id,
                    "tenantId": doc.get("tenantId", _DEFAULT_TENANT_ID),
                    "featureSummary": doc["featureSummary"],
                })

            # CASE 2: raw telemetry
            elif _validate_raw_document(doc):
                user_id = doc.get("user_identity", {}).get("anonymous_id")
                if not user_id:
                    skipped += 1
                    continue

                feature_summary = extract_feature_summary(doc)
                if not feature_summary:
                    skipped += 1
                    continue

                # Use tenant from document if present, else fall back to default
                tenant_id = doc.get("tenantId") or doc.get("tenant_id") or _DEFAULT_TENANT_ID

                batch.append({
                    "userId": user_id,
                    "tenantId": tenant_id,
                    "featureSummary": feature_summary,
                })

            else:
                skipped += 1
                continue

            if len(batch) >= batch_size:
                logger.debug("load_data: yielding batch of %d", len(batch))
                yield batch
                batch = []

        if batch:
            logger.debug("load_data: yielding final batch of %d", len(batch))
            yield batch

        logger.info("load_data: finished streaming. skipped=%d", skipped)

    finally:
        cursor.close()
        logger.debug("load_data: cursor closed")