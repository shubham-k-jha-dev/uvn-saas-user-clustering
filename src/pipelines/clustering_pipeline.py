import logging
from typing import Any, Dict, List

import numpy as np

from src.data.preprocess import preprocess_batch
from src.features.build_features import attach_cluster_labels, label_clusters
from src.features.customer_intelligence import (
    enrich_with_customer_intelligence,
    override_high_value_user,
)
from src.features.user_aggregator import aggregate_user_sessions
from src.models.train_model import predict_clusters

logger = logging.getLogger(__name__)


# Input validation

def validate_input(data: List[Dict[str, Any]]) -> None:
    """
    Validates the structure of incoming user dicts.

    """
    if not isinstance(data, list):
        raise ValueError("Input data must be a list of user dicts")

    if len(data) == 0:
        raise ValueError("Input data must not be empty")

    required_keys = {"userId", "tenantId", "featureSummary"}
    for i, item in enumerate(data):
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(
                f"User at index {i} is missing required keys: {missing}"
            )


# Business-rule override layer
def _apply_business_overrides(
    final_results: List[Dict[str, Any]],
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Applies deterministic business rules on top of ML cluster labels.
    This layer sits AFTER label assignment so the ML output is always
    visible in logs and the override is a deliberate, transparent correction.
    """
    corrected = []
    for i, result in enumerate(final_results):
        original_label = result["clusterLabel"]
        corrected_label = override_high_value_user(data[i], original_label)

        if corrected_label != original_label:
            logger.info(
                "Business override for user %s: %s → %s",
                result.get("userId"),
                original_label,
                corrected_label,
            )

        corrected.append({**result, "clusterLabel": corrected_label})
    return corrected


# Main pipeline

def run_clustering_pipeline(
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Runs the full inference pipeline for a batch of users.
    """
    from src.models.model_manager import get_model

    logger.info("Clustering pipeline: starting for %d users", len(data))

    # Validate
    validate_input(data)

    # Aggregate multi-session docs per user
    aggregated = aggregate_user_sessions(data)

    # Build raw feature matrix + log transforms (no scaling here)
    X, metadata = preprocess_batch(aggregated)

    if X.shape[0] == 0:
        logger.warning("Clustering pipeline: no valid rows after preprocessing")
        return []

    # Load the cached Pipeline (scaler + kmeans) from the singleton.
    #    We do NOT train here. Training is the responsibility of
    #    training_pipeline.py. If the model is absent, fail fast.
    model = get_model()
    if model is None:
        raise ValueError(
            "No model loaded. Call POST /train-model before running inference."
        )

    # Scale → predict (both handled by the Pipeline in one call)
    cluster_results = predict_clusters(model, X, metadata)

    # Score centroids → rank → assign semantic labels
    cluster_labels = label_clusters(model)

    # Attach labels to each user
    final_results = attach_cluster_labels(cluster_results, cluster_labels)

    # Apply business-rule overrides (transparent log on every correction)
    final_results = _apply_business_overrides(final_results, aggregated)

    # Enrich with customer intelligence
    final_results = enrich_with_customer_intelligence(final_results)

    logger.info(
        "Clustering pipeline: complete for %d users",
        len(final_results),
    )
    return final_results