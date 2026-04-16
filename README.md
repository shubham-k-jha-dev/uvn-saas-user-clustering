# User Clustering Module (v3.0)

A production-ready, scalable user segmentation engine for behavioral analytics systems.

---

## What's New in Version 3

This major upgrade transitions the system from a working prototype to a highly robust, production-grade ML microservice.

- **Unified ML Pipeline**: Training and Inference exactly match via a unified `sklearn.Pipeline` (bundling `StandardScaler` + `MiniBatchKMeans`), eliminating feature scaling corruption.
- **Thread-safe In-Memory Model**: The model is cached in-memory and protected with `threading.Lock()` against background re-training, allowing safe, lock-free reads during inference.
- **Dynamic Cluster Labeling**: Semantic labels (High Value, Engaged, Drop-off) are now dynamically assigned regardless of the final 'K' chosen by the model. 
- **Production-grade API**: Features FastAPI `/ready` and `/health` endpoints with model-state dependencies, ensuring traffic is only served when the model is initialized.
- **Mathematical Correctness**: `user_aggregator` now correctly calculates true weighted averages on pre-aggregated data, replacing standard division.
- **Comprehensive Testing Suite**: Added a robust 26-test suite using `pytest` verifying everything from data transformations to entire endpoint flows, no database needed.

---

## Overview

This module performs user clustering based on behavioral features such as:

- session activity
- engagement patterns
- bounce behavior
- recency

It is designed to be:

- Database-agnostic (can work with any kind of database)
- Scalable (batch-based training to prevent OOM)
- Easily integrable with backend systems via REST API
- Interpretable via deterministic Business Rules

---

## Architecture

```text
Data Source (Mongo / Raw Telemetry Data)
↓
Data Loader & Preprocessor (Weighted Aggregation, Log Transforms)
↓
Sklearn ML Pipeline (StandardScaler → MiniBatchKMeans)
↓
Dynamic Cluster Labeling & Business Override Layer
↓
Customer Intelligence Generation
↓
API Response (via FastAPI)
```

---

## Input Format

```json
[
  {
    "userId": "u1",
    "tenantId": "app_001",
    "featureSummary": {
      "sessionCount": 3,
      "avgSessionTime": 5,
      "pagesPerSession": 1.3,
      "clicksPerSession": 2,
      "bounceRate": 0.3,
      "recencyDays": 0
    }
  }
]
```

## Output Format

```json
[
  {
    "userId": "u1",
    "tenantId": "app_001",
    "clusterId": 0,
    "clusterLabel": "high_value_user",
    "customerIntelligence": {
      "engagementLevel": "high",
      "churnRisk": "low",
      "recommendedAction": "Upsell premium features"
    }
  }
]
```

## Cluster Labels

- **high_value_user**: Highly engaged, frequent users
- **engaged_user**: Active but not top-tier
- **casual_user**: Moderate usage
- **drop_off_user**: Low activity, high churn risk

## How to Run

1. Copy `.env.example` to `.env` and fill the variables.
2. Run tests to confirm integrity:
   ```bash
   python -m pytest tests/
   ```
3. Run the development server:
   ```bash
   python main.py
   ```

## Integration (Backend)

The service runs via FastAPI. You can POST to `/cluster-users` or call the pipeline directly in python:

```python
from src.pipelines.clustering_pipeline import run_clustering_pipeline

results = run_clustering_pipeline(data)
```

## Core Features

- **Automated ML pipeline:** MongoDB → Training → Persisted Model → Fast Inference
- **User clustering:** Using `MiniBatchKMeans` with automatic K-selection.
- **Customer intelligence generation:** Built-in recommendations.
- **Background model retraining:** Daemon thread handles continuous learning without blocking UI.
- **Model persistence:** Joblib Serialization + Thread-Safe App Memory
- **REST API:** Built with FastAPI `lifespan` architecture.
- **Continuous "Bounce Rate" Decay:** Features gradients instead of binary step functions.

## Future Improvements

- Customer context enrichment (Geo, Device)
- Model drift detection to alert scaling changes
- Support for out-of-core online clustering loops (`partial_fit`)

## Author

**Shubham Kumar Jha**
