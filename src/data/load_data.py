import os
import logging
from typing import Generator, Dict, Any, List

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
USER_FEATURES_COLLECTION = os.getenv("USER_FEATURES_COLLECTION")

if not MONGO_URI or not DB_NAME or not USER_FEATURES_COLLECTION:
    raise ValueError("Missing required envt. variables in .env")


# Database connection
def get_mongo_client() -> MongoClient:
    """Establishes connection to MongoDB"""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS = 5000)
        client.admin.command("ping")
        logger.info("MongoDB connection established successfully")
        return client
    except ConnectionFailure as e:
        logger.error("Failed to connect to MongoDB", exc_info = True)
        raise e


def get_user_features_collection():
    """Returns mongodb collection object"""
    client = get_mongo_client()
    db = client[DB_NAME]
    collection = db[USER_FEATURES_COLLECTION]
    return collection

# Basic validation
def validate_document(doc: Dict[str, Any]) -> bool:
    """
    Validate if document has required fields
    """
    if "userId" not in doc:
        return False

    if "featureSummary" not in doc:
        return False

    if not isinstance(doc["featureSummary"], dict):
        return False

    return True


def load_user_features_in_batches(
    batch_size: int = 1000
) -> Generator[List[Dict[str, Any]], None, None]:
    """Load user features in batches"""
    collection = get_user_features_collection()

    total_docs = collection.count_documents({})
    logger.info(f"Total documents in collection: {total_docs}")

    if total_docs == 0:
        logger.warning("No documents found in user_features collection")
        return

    cursor = collection.find({}, no_cursor_timeout=True)

    batch = []
    skipped = 0

    try:
        for doc in cursor:
            if not validate_document(doc):
                skipped += 1
                continue

            batch.append(doc)

            if len(batch) >= batch_size:
                logger.info(f"Yielding batch of size {len(batch)}")
                yield batch
                batch = []

        # last batch
        if batch:
            logger.info(f"Yielding final batch of size {len(batch)}")
            yield batch

        logger.info(f"Skipped invalid documents: {skipped}")

    finally:
        cursor.close()

# Test function
def test_data_loading():
    """
    Debug function to test loading
    """
    logger.info("Starting test data loading...")

    for i, batch in enumerate(load_user_features_in_batches(batch_size=2)):
        logger.info(f"Batch {i + 1}: {len(batch)} records")

        for doc in batch:
            logger.info(doc)


if __name__ == "__main__":
    test_data_loading()