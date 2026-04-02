import logging
from src.pipeline.clustering_pipeline import run_clustering_pipeline
from src.data.load_data import load_user_features_in_batches


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """
    Entry point for running clustering pipeline
    """

    logger.info("Starting main pipeline execution")

    all_data = []

    # Load from Mongo (adapter layer usage)
    for batch in load_user_features_in_batches(batch_size=1000):
        all_data.extend(batch)

    if not all_data:
        logger.warning("No data found. Exiting.")
        return

    # Run ML pipeline (core logic)
    results = run_clustering_pipeline(all_data)

    print("\nFINAL RESULTS:")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()