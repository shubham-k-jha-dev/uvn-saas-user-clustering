import sys
import os

# Ensure the project root is on the path when running from the tests/ dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

# Fixtures

@pytest.fixture(scope="module")
def trained_pipeline():
    """
    Builds and returns a real sklearn Pipeline trained on synthetic
    four-profile data. Used by all tests that need a model.

    The pipeline is trained ONCE per test module to keep the suite fast.
    """
    rng = np.random.default_rng(42)

    # Simulate four distinct user profiles
    n = 200  # 50 per profile
    high = rng.uniform([30, 20, 8, 15, 0.01, 0], [100, 50, 20, 40, 0.20, 2], (n // 4, 6))
    medium = rng.uniform([10, 8, 4, 5, 0.20, 2], [30, 20, 10, 15, 0.50, 7], (n // 4, 6))
    low = rng.uniform([3, 3, 2, 1, 0.50, 5], [10, 10, 5, 5, 0.80, 20], (n // 4, 6))
    drop = rng.uniform([1, 1, 1, 0, 0.85, 20], [2, 3, 1, 0, 0.99, 90], (n // 4, 6))
    X = np.vstack([high, medium, low, drop])

    # Apply log1p to skewed columns (same as preprocess.py)
    X[:, 0] = np.log1p(X[:, 0])  # sessionCount
    X[:, 1] = np.log1p(X[:, 1])  # avgSessionTime
    X[:, 3] = np.log1p(X[:, 3])  # clicksPerSession

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=200)),
    ])
    pipeline.fit(X)
    return pipeline


@pytest.fixture(scope="module")
def api_client(trained_pipeline):
    """
    Returns a FastAPI TestClient with the trained Pipeline pre-loaded into
    model_manager. This simulates a fully-started server without needing
    MongoDB or a real HTTP server.
    """
    from src.models.model_manager import set_model
    set_model(trained_pipeline)

    from main import app
    return TestClient(app)



# Unit tests — user_aggregator.py

class TestUserAggregator:

    def test_single_doc_passes_through_unchanged(self):
        """A single document should come out with identical feature values."""
        from src.features.user_aggregator import aggregate_user_sessions

        docs = [{
            "userId": "u1",
            "tenantId": "app_001",
            "featureSummary": {
                "sessionCount": 80,
                "avgSessionTime": 40,
                "pagesPerSession": 15,
                "clicksPerSession": 30,
                "bounceRate": 0.05,
                "recencyDays": 1,
            }
        }]

        result = aggregate_user_sessions(docs)
        assert len(result) == 1
        fs = result[0]["featureSummary"]

        assert fs["sessionCount"] == 80
        assert abs(fs["avgSessionTime"] - 40) < 0.01, (
            f"avgSessionTime should be 40, got {fs['avgSessionTime']}. "
            "This confirms the weighted-average division bug is fixed."
        )
        assert abs(fs["bounceRate"] - 0.05) < 0.01

    def test_two_docs_same_user_weighted_average(self):
        """Two docs for the same user must produce a weighted average."""
        from src.features.user_aggregator import aggregate_user_sessions

        docs = [
            {
                "userId": "u1",
                "tenantId": "app_001",
                "featureSummary": {
                    "sessionCount": 40,
                    "avgSessionTime": 20,
                    "pagesPerSession": 5,
                    "clicksPerSession": 10,
                    "bounceRate": 0.1,
                    "recencyDays": 5,
                }
            },
            {
                "userId": "u1",
                "tenantId": "app_001",
                "featureSummary": {
                    "sessionCount": 40,
                    "avgSessionTime": 60,
                    "pagesPerSession": 15,
                    "clicksPerSession": 30,
                    "bounceRate": 0.5,
                    "recencyDays": 1,
                }
            }
        ]

        result = aggregate_user_sessions(docs)
        assert len(result) == 1, "Two docs for same user must produce one row"

        fs = result[0]["featureSummary"]
        # Both docs have equal sessionCount=40, so averages should be equal-weight means
        assert abs(fs["avgSessionTime"] - 40.0) < 0.01, f"Expected 40.0, got {fs['avgSessionTime']}"
        assert abs(fs["bounceRate"] - 0.30) < 0.01
        assert fs["recencyDays"] == 1, "Recency should be the minimum (most recent)"

    def test_two_separate_users_stay_separate(self):
        """Two different users must produce two separate output rows."""
        from src.features.user_aggregator import aggregate_user_sessions

        docs = [
            {"userId": "u1", "tenantId": "a", "featureSummary": {
                "sessionCount": 10, "avgSessionTime": 5, "pagesPerSession": 2,
                "clicksPerSession": 1, "bounceRate": 0.3, "recencyDays": 3,
            }},
            {"userId": "u2", "tenantId": "a", "featureSummary": {
                "sessionCount": 50, "avgSessionTime": 30, "pagesPerSession": 10,
                "clicksPerSession": 20, "bounceRate": 0.05, "recencyDays": 0,
            }},
        ]

        result = aggregate_user_sessions(docs)
        assert len(result) == 2

    def test_missing_userId_is_skipped(self):
        """Docs with no userId must be silently dropped."""
        from src.features.user_aggregator import aggregate_user_sessions

        docs = [
            {"tenantId": "a", "featureSummary": {"sessionCount": 5, "avgSessionTime": 5,
             "pagesPerSession": 1, "clicksPerSession": 1, "bounceRate": 0.5, "recencyDays": 2}},
        ]

        result = aggregate_user_sessions(docs)
        assert len(result) == 0



# Unit tests — build_features.py (label_clusters)

class TestLabelClusters:

    def _make_pipeline(self, k: int, centroids: np.ndarray) -> Pipeline:
        """Helper: build a Pipeline with manually set cluster_centers_."""
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("kmeans", MiniBatchKMeans(n_clusters=k, random_state=42)),
        ])
        # Train on dummy data large enough
        dummy_X = np.random.randn(k * 20, 6)
        pipeline.fit(dummy_X)
        # Override centroids with our controlled values
        pipeline.named_steps["kmeans"].cluster_centers_ = centroids
        return pipeline

    def test_k2_always_has_high_and_dropoff(self):
        """With K=2, the best cluster must be high_value and the worst drop_off."""
        from src.features.build_features import label_clusters

        # Cluster 0: high engagement (high sessionCount/time, low bounce/recency in scaled space)
        # Cluster 1: low engagement
        centroids = np.array([
            [2.0, 2.0, 2.0, 2.0, -2.0, -2.0],   # cluster 0: high value
            [-2.0, -2.0, -2.0, -2.0, 2.0, 2.0],  # cluster 1: drop off
        ])
        pipeline = self._make_pipeline(2, centroids)
        label_map = label_clusters(pipeline)

        assert label_map[0] == "high_value_user", f"Got: {label_map}"
        assert label_map[1] == "drop_off_user", f"Got: {label_map}"

    def test_k4_full_spectrum(self):
        """With K=4, all four labels must appear."""
        from src.features.build_features import label_clusters

        centroids = np.array([
            [3.0, 3.0, 3.0, 3.0, -3.0, -3.0],   # rank 0 → high_value
            [1.0, 1.0, 1.0, 1.0, -1.0, -1.0],   # rank 1 → engaged
            [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0], # rank 2 → casual
            [-3.0, -3.0, -3.0, -3.0, 3.0, 3.0], # rank 3 → drop_off
        ])
        pipeline = self._make_pipeline(4, centroids)
        label_map = label_clusters(pipeline)

        labels = set(label_map.values())
        assert "high_value_user" in labels
        assert "engaged_user" in labels
        assert "casual_user" in labels
        assert "drop_off_user" in labels

    def test_k3_no_engaged_label_missing(self):
        """With K=3, drop_off must still be assigned (not stuck at engaged_user)."""
        from src.features.build_features import label_clusters

        centroids = np.array([
            [2.0, 2.0, 2.0, 2.0, -2.0, -2.0],   # rank 0 → high_value
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],     # rank 1 → engaged
            [-2.0, -2.0, -2.0, -2.0, 2.0, 2.0], # rank 2 → drop_off (K-1)
        ])
        pipeline = self._make_pipeline(3, centroids)
        label_map = label_clusters(pipeline)

        assert label_map is not None
        values = list(label_map.values())
        assert "drop_off_user" in values, (
            f"drop_off_user must be reachable with K=3. Got: {label_map}"
        )
        assert "high_value_user" in values


# Unit tests — feature_extractor.py

class TestFeatureExtractor:

    def test_zero_duration_gives_full_bounce(self):
        """A 0-second session should produce bounceRate = 1.0."""
        from src.features.feature_extractor import extract_feature_summary

        raw = {
            "user_identity": {"anonymous_id": "u1", "lifetime_sessions": 5, "last_seen": None},
            "session_context": {"session_duration_sec": 0},
            "engagement_metrics": {"pageviews": 1, "click_count": 0},
        }
        result = extract_feature_summary(raw)
        assert result["bounceRate"] == 1.0

    def test_30s_session_gives_zero_bounce(self):
        """A 30-second session should produce bounceRate = 0.0."""
        from src.features.feature_extractor import extract_feature_summary

        raw = {
            "user_identity": {"anonymous_id": "u1", "lifetime_sessions": 5, "last_seen": None},
            "session_context": {"session_duration_sec": 30},
            "engagement_metrics": {"pageviews": 3, "click_count": 5},
        }
        result = extract_feature_summary(raw)
        assert result["bounceRate"] == 0.0

    def test_15s_session_gives_half_bounce(self):
        """A 15-second session should produce bounceRate ≈ 0.5."""
        from src.features.feature_extractor import extract_feature_summary

        raw = {
            "user_identity": {"anonymous_id": "u1", "lifetime_sessions": 5, "last_seen": None},
            "session_context": {"session_duration_sec": 15},
            "engagement_metrics": {"pageviews": 2, "click_count": 2},
        }
        result = extract_feature_summary(raw)
        assert abs(result["bounceRate"] - 0.5) < 0.01

    def test_malformed_doc_returns_empty(self):
        """A totally malformed document should return {} without raising."""
        from src.features.feature_extractor import extract_feature_summary

        result = extract_feature_summary({})
        # Not a crash — returns empty dict
        assert isinstance(result, dict)


# Unit tests — business override

class TestOverrideHighValueUser:

    def _make_user(self, session_count, avg_time, pages, bounce, recency):
        return {
            "featureSummary": {
                "sessionCount": session_count,
                "avgSessionTime": avg_time,
                "pagesPerSession": pages,
                "clicksPerSession": 10,
                "bounceRate": bounce,
                "recencyDays": recency,
            }
        }

    def test_qualifies_for_override(self):
        """User meeting all five criteria must be forced to high_value_user."""
        from src.features.customer_intelligence import override_high_value_user

        user = self._make_user(60, 25, 12, 0.1, 1)
        result = override_high_value_user(user, "engaged_user")
        assert result == "high_value_user"

    def test_does_not_override_if_bounce_too_high(self):
        """High bounce rate (>= 0.2) must prevent the override."""
        from src.features.customer_intelligence import override_high_value_user

        user = self._make_user(60, 25, 12, 0.25, 1)
        result = override_high_value_user(user, "engaged_user")
        assert result == "engaged_user"

    def test_does_not_override_if_recency_too_old(self):
        """recencyDays > 2 must prevent the override."""
        from src.features.customer_intelligence import override_high_value_user

        user = self._make_user(60, 25, 12, 0.1, 5)
        result = override_high_value_user(user, "engaged_user")
        assert result == "engaged_user"

    def test_already_high_value_stays_high_value(self):
        """If ML already assigned high_value_user, label must not change."""
        from src.features.customer_intelligence import override_high_value_user

        user = self._make_user(60, 25, 12, 0.1, 1)
        result = override_high_value_user(user, "high_value_user")
        assert result == "high_value_user"


# API integration tests


class TestClusterUsersEndpoint:

    def test_health_always_200(self, api_client):
        """Liveness probe must return 200 unconditionally."""
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ready_200_when_model_loaded(self, api_client):
        """Readiness probe must return 200 when Pipeline is in memory."""
        response = api_client.get("/ready")
        assert response.status_code == 200

    def test_high_value_user_gets_correct_label(self, api_client):
        """
        A user with high sessions, long time, low bounce, recent activity
        must receive high_value_user (via business override or ML).
        """
        response = api_client.post("/cluster-users", json={"users": [{
            "userId": "test_high",
            "tenantId": "app_001",
            "featureSummary": {
                "sessionCount": 80,
                "avgSessionTime": 40,
                "pagesPerSession": 15,
                "clicksPerSession": 30,
                "bounceRate": 0.05,
                "recencyDays": 1,
            }
        }]})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        user_result = data["data"][0]
        assert user_result["clusterLabel"] == "high_value_user", (
            f"Expected high_value_user, got {user_result['clusterLabel']}"
        )
        assert user_result["customerIntelligence"]["engagementLevel"] == "high"
        assert user_result["customerIntelligence"]["churnRisk"] == "low"

    def test_drop_off_user_gets_correct_label(self, api_client):
        """
        A user with minimal sessions, short time, near-100% bounce, and
        long inactivity must receive drop_off_user.
        """
        response = api_client.post("/cluster-users", json={"users": [{
            "userId": "test_drop",
            "tenantId": "app_001",
            "featureSummary": {
                "sessionCount": 1,
                "avgSessionTime": 2,
                "pagesPerSession": 1,
                "clicksPerSession": 0,
                "bounceRate": 0.95,
                "recencyDays": 60,
            }
        }]})
        assert response.status_code == 200
        user_result = response.json()["data"][0]
        assert user_result["clusterLabel"] == "drop_off_user", (
            f"Expected drop_off_user, got {user_result['clusterLabel']}"
        )
        assert user_result["customerIntelligence"]["churnRisk"] == "high"

    def test_batch_two_extremes_get_different_clusters(self, api_client):
        """
        The core regression test: a high-value and drop-off user in the same
        batch must be assigned DIFFERENT clusterIds and DIFFERENT labels.
        This is what the original bug broke — both users got the same cluster.
        """
        response = api_client.post("/cluster-users", json={"users": [
            {
                "userId": "batch_high",
                "tenantId": "app_001",
                "featureSummary": {
                    "sessionCount": 90,
                    "avgSessionTime": 45,
                    "pagesPerSession": 18,
                    "clicksPerSession": 35,
                    "bounceRate": 0.03,
                    "recencyDays": 0,
                }
            },
            {
                "userId": "batch_low",
                "tenantId": "app_001",
                "featureSummary": {
                    "sessionCount": 1,
                    "avgSessionTime": 1,
                    "pagesPerSession": 1,
                    "clicksPerSession": 0,
                    "bounceRate": 0.99,
                    "recencyDays": 90,
                }
            }
        ]})

        assert response.status_code == 200
        results = response.json()["data"]
        assert len(results) == 2

        labels = {r["userId"]: r["clusterLabel"] for r in results}
        assert labels["batch_high"] != labels["batch_low"], (
            f"REGRESSION: both users got the same label: {labels}. "
            "This means the scaler bug has returned."
        )
        assert labels["batch_high"] == "high_value_user"
        assert labels["batch_low"] == "drop_off_user"

    def test_zero_activity_user_does_not_crash(self, api_client):
        """A user with all-zero features must return 200, not a 500."""
        response = api_client.post("/cluster-users", json={"users": [{
            "userId": "test_zero",
            "tenantId": "app_001",
            "featureSummary": {
                "sessionCount": 0,
                "avgSessionTime": 0,
                "pagesPerSession": 0,
                "clicksPerSession": 0,
                "bounceRate": 1.0,
                "recencyDays": 999,
            }
        }]})
        assert response.status_code == 200
        user_result = response.json()["data"][0]
        # Should be drop-off, not crash
        assert user_result["clusterLabel"] in ("drop_off_user", "casual_user")

    def test_extreme_power_user_does_not_crash(self, api_client):
        """Extreme feature values (5000 sessions) must not cause errors."""
        response = api_client.post("/cluster-users", json={"users": [{
            "userId": "test_power",
            "tenantId": "app_001",
            "featureSummary": {
                "sessionCount": 5000,
                "avgSessionTime": 120,
                "pagesPerSession": 80,
                "clicksPerSession": 200,
                "bounceRate": 0.0,
                "recencyDays": 0,
            }
        }]})
        assert response.status_code == 200

    def test_empty_users_list_returns_422(self, api_client):
        """
        An empty users list must be rejected at the Pydantic layer with 422.
        BatchRequest has min_length=1 on the users field.
        """
        response = api_client.post("/cluster-users", json={"users": []})
        assert response.status_code == 422

    def test_missing_feature_field_returns_422(self, api_client):
        """A featureSummary missing bounceRate must return 422 (Pydantic validation)."""
        response = api_client.post("/cluster-users", json={"users": [{
            "userId": "u1",
            "tenantId": "app_001",
            "featureSummary": {
                "sessionCount": 10,
                "avgSessionTime": 5,
                "pagesPerSession": 3,
                "clicksPerSession": 2,
                # bounceRate missing
                "recencyDays": 4,
            }
        }]})
        assert response.status_code == 422

    def test_bounce_rate_out_of_range_returns_422(self, api_client):
        """bounceRate > 1.0 must be rejected by Pydantic (Field le=1)."""
        response = api_client.post("/cluster-users", json={"users": [{
            "userId": "u1",
            "tenantId": "app_001",
            "featureSummary": {
                "sessionCount": 5,
                "avgSessionTime": 3,
                "pagesPerSession": 2,
                "clicksPerSession": 1,
                "bounceRate": 1.5,  # invalid
                "recencyDays": 10,
            }
        }]})
        assert response.status_code == 422

    def test_response_structure_is_complete(self, api_client):
        """Every response must include success, count, meta, and data fields."""
        response = api_client.post("/cluster-users", json={"users": [{
            "userId": "structure_check",
            "tenantId": "app_001",
            "featureSummary": {
                "sessionCount": 5,
                "avgSessionTime": 5,
                "pagesPerSession": 3,
                "clicksPerSession": 2,
                "bounceRate": 0.4,
                "recencyDays": 7,
            }
        }]})
        assert response.status_code == 200
        body = response.json()
        assert "success" in body
        assert "count" in body
        assert "meta" in body
        assert "data" in body
        assert "modelVersion" in body["meta"]
        assert "processedAt" in body["meta"]

        user = body["data"][0]
        assert "userId" in user
        assert "tenantId" in user
        assert "clusterId" in user
        assert "clusterLabel" in user
        assert "customerIntelligence" in user

        ci = user["customerIntelligence"]
        assert "engagementLevel" in ci
        assert "churnRisk" in ci
        assert "recommendedAction" in ci