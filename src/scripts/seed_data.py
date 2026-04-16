import logging
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "user_clustering_uvn")
COLLECTION_NAME = os.getenv("USER_FEATURES_COLLECTION", "user_features")


# Profile generators
def _generate_user(user_id: int) -> Dict[str, Any]:
    """
    Generates a single synthetic user document.

    Profile distribution: 25% each (high, medium, low, drop).
    The feature ranges are intentionally spread apart so the clustering
    model can find meaningful separations during training.
    """
    profile_type = random.choice(["high", "medium", "low", "drop"])
    tenant_id = f"app_{random.randint(1, 3):03d}"

    if profile_type == "high":
        feature_summary = {
            "sessionCount": random.randint(30, 100),
            "avgSessionTime": random.uniform(20, 50),
            "pagesPerSession": random.uniform(8, 20),
            "clicksPerSession": random.uniform(15, 40),
            "bounceRate": random.uniform(0.01, 0.20),
            "recencyDays": random.randint(0, 2),
        }
    elif profile_type == "medium":
        feature_summary = {
            "sessionCount": random.randint(10, 30),
            "avgSessionTime": random.uniform(8, 20),
            "pagesPerSession": random.uniform(4, 10),
            "clicksPerSession": random.uniform(5, 15),
            "bounceRate": random.uniform(0.20, 0.50),
            "recencyDays": random.randint(2, 7),
        }
    elif profile_type == "low":
        feature_summary = {
            "sessionCount": random.randint(3, 10),
            "avgSessionTime": random.uniform(3, 10),
            "pagesPerSession": random.uniform(2, 5),
            "clicksPerSession": random.uniform(1, 5),
            "bounceRate": random.uniform(0.50, 0.80),
            "recencyDays": random.randint(5, 20),
        }
    else:  # drop
        feature_summary = {
            "sessionCount": random.randint(1, 2),
            "avgSessionTime": random.uniform(1, 3),
            "pagesPerSession": 1.0,
            "clicksPerSession": 0.0,
            "bounceRate": random.uniform(0.85, 0.99),
            "recencyDays": random.randint(20, 90),
        }

    return {
        "userId": f"u{user_id}",
        "tenantId": tenant_id,
        "featureSummary": feature_summary,
        "createdAt": datetime.now(timezone.utc),
    }


# Seeder

def seed_data(n_users: int = 10_000, batch_size: int = 1_000) -> None:
    """
    Inserts `n_users` synthetic documents into MongoDB.

    Args:
        n_users:    Total number of user documents to insert.
        batch_size: Number of documents per insert_many call.
    """
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    logger.info("Seeding %d users into %s.%s in batches of %d", n_users, DB_NAME, COLLECTION_NAME, batch_size)

    total_inserted = 0

    for i in range(0, n_users, batch_size):
        end = min(i + batch_size, n_users)
        batch = [_generate_user(j) for j in range(i, end)]
        collection.insert_many(batch)
        total_inserted += len(batch)
        logger.info("Inserted users %d → %d", i, total_inserted)

    logger.info("Seeding complete: %d users inserted", total_inserted)
    client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed synthetic user data into MongoDB")
    parser.add_argument("--n", type=int, default=10_000, help="Number of users to seed")
    parser.add_argument("--batch", type=int, default=1_000, help="Insert batch size")
    args = parser.parse_args()

    seed_data(n_users=args.n, batch_size=args.batch)