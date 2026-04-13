# User Clustering Module (v1)

A production-ready, scalable user segmentation engine for behavioral analytics systems.

---

## Overview

This module performs user clustering based on behavioral features such as:

- session activity
- engagement patterns
- bounce behavior
- recency

It is designed to be:

- Database-agnostic(can work with any kind of database)
- Scalable (batch-based)  
- Easily integrable with backend systems  

---

## Architecture

```text
Data Source (Mongo / API / Warehouse)
↓
Preprocessing Layer
↓
Feature Transformation
↓
Clustering Model (MiniBatchKMeans)
↓
Cluster Labeling
↓
Output (JSON / API / DB)
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
    "clusterLabel": "high_value_user",
    "customerIntelligence": {
      "engagementLevel": "high",
      "churnRisk": "low",
      "action": "Upsell premium features"
    }
  }
]
```

## Cluster Labels

- high_value_user: highly engaged, frequent users
- engaged_user: active but not top-tier
- casual_user: moderate usage
- drop_off_user: low activity, high churn risk

## How to Run

```bash
python main.py
```

## Core API

```python
from src.pipelines.clustering_pipeline import run_clustering_pipeline

results = run_clustering_pipeline(data)
```

## Integration (Backend)

1. Pass user feature data into the pipelines
2. Store results in `user_clusters` collection
3. Inject into API response:

```json
"userCluster": {
  "clusterId": 0,
  "label": "high_value_user"
}
```

## Key Design Principles

- Separation of concerns
- Database-independent ML pipelines
- Batch processing (scalable)
- Interpretable clustering

## Features

- Automated ML pipeline (MongoDB → Training → API)
- User clustering using MiniBatchKMeans
- Customer intelligence generation
- Background model retraining
- Model persistence (disk + memory)
- REST API using FastAPI

## Future Improvements

- Customer intelligence layer (churn, retention)
- Model persistence
- Incremental clustering
- Real-time inference API

## Author

Shubham Kumar Jha

---

## Architecture Diagram (Simplified)

```text
SDK Events
↓
MongoDB / Data Source
↓
Aggregation → user_features
↓
Clustering pipelines
↓
user_clusters (output)
↓
Backend API
↓
Frontend Dashboard
```
