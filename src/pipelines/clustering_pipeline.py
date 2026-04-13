import logging
from typing import List, Dict, Any

import numpy as np
from src.models.model_manager import get_model, set_model
from src.data.preprocess import preprocess_batch
from src.models.train_model import train_clustering_model, predict_clusters
from src.features.build_features import label_clusters, attach_cluster_labels
from src.features.customer_intelligence import enrich_with_customer_intelligence


logger = logging.getLogger(__name__)

def validate_input(data: List[Dict[str, Any]]) -> None:
    if not isinstance(data, list):
        raise ValueError("Input data must be a list of dictionaries")

    if len(data) == 0:
        raise ValueError("Input data is empty")

    required_keys = {"userId", "tenantId", "featureSummary"}

    for i, item in enumerate(data):
        if not required_keys.issubset(item.keys()):
            raise ValueError(f"Missing required keys in item index {i}")


def run_clustering_pipeline(
    data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Main clustering pipeline
    """

    logger.info("Starting clustering pipeline")

    # validate input
    validate_input(data)

    # preprocess
    X, metadata = preprocess_batch(data)

    if X.shape[0] == 0:
        logger.warning("No valid data after preprocessing")
        return []

    # train model
    model = get_model()

    # Only train if enough data
    if model is None:
        if X.shape[0] < 10:
            logger.warning("Not enough data to train model. Need at least 10 users.")
            raise ValueError("Insufficient data for training model")

        logger.info("Training model with sufficient data...")
        model = train_clustering_model(X)
        set_model(model)
    else:
        logger.info("Using cached model")

    # predict clusters
    cluster_results = predict_clusters(model, X, metadata)

    # label clusters
    cluster_labels = label_clusters(model)

    # attach labels
    final_results = attach_cluster_labels(cluster_results, cluster_labels)

    # add customer intelligence layer
    final_results = enrich_with_customer_intelligence(final_results)

    logger.info(
        "Clustering + intelligence pipeline completed successfully "
        f"for {len(final_results)} users"
    )

    return final_results