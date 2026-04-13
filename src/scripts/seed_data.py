import random
from pymongo import MongoClient
from datetime import datetime

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "user_clustering_uvn"
COLLECTION_NAME = "user_features"


def generate_user(user_id: int):
    profile_type = random.choice(["high", "medium", "low", "drop"])

    if profile_type == "high":
        return {
            "userId": f"u{user_id}",
            "tenantId": "app_001",
            "featureSummary": {
                "sessionCount": random.randint(30, 100),
                "avgSessionTime": random.uniform(20, 50),
                "pagesPerSession": random.uniform(8, 20),
                "clicksPerSession": random.uniform(15, 40),
                "bounceRate": random.uniform(0.01, 0.2),
                "recencyDays": random.randint(0, 2)
            },
            "createdAt": datetime.utcnow()
        }

    elif profile_type == "drop":
        return {
            "userId": f"u{user_id}",
            "tenantId": "app_001",
            "featureSummary": {
                "sessionCount": random.randint(1, 2),
                "avgSessionTime": random.uniform(1, 3),
                "pagesPerSession": 1,
                "clicksPerSession": 0,
                "bounceRate": random.uniform(0.85, 0.99),
                "recencyDays": random.randint(20, 90)
            },
            "createdAt": datetime.utcnow()
        }

    elif profile_type == "medium":
        return {
            "userId": f"u{user_id}",
            "tenantId": "app_001",
            "featureSummary": {
                "sessionCount": random.randint(10, 30),
                "avgSessionTime": random.uniform(8, 20),
                "pagesPerSession": random.uniform(4, 10),
                "clicksPerSession": random.uniform(5, 15),
                "bounceRate": random.uniform(0.2, 0.5),
                "recencyDays": random.randint(2, 7)
            },
            "createdAt": datetime.utcnow()
        }

    else:  # low
        return {
            "userId": f"u{user_id}",
            "tenantId": "app_001",
            "featureSummary": {
                "sessionCount": random.randint(3, 10),
                "avgSessionTime": random.uniform(3, 10),
                "pagesPerSession": random.uniform(2, 5),
                "clicksPerSession": random.uniform(1, 5),
                "bounceRate": random.uniform(0.5, 0.8),
                "recencyDays": random.randint(5, 20)
            },
            "createdAt": datetime.utcnow()
        }


def seed_data(n_users=10000, batch_size=1000):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    print(f"Seeding {n_users} users in batches of {batch_size}...")

    for i in range(0, n_users, batch_size):
        batch = [
            generate_user(user_id=j)
            for j in range(i, min(i + batch_size, n_users))
        ]

        collection.insert_many(batch)

        print(f"Inserted users {i} → {i + len(batch)}")

    print("Seeding completed successfully!")


if __name__ == "__main__":
    seed_data(10000)