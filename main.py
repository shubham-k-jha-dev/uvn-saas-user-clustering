from src.data.load_data import load_user_features_in_batches
from src.data.preprocess import preprocess_batch
from src.models.train_model import train_clustering_model, predict_clusters
from src.features.build_features import label_clusters, attach_cluster_labels

all_X = []
all_meta = []

# collect all data (small dataset for now)
for batch in load_user_features_in_batches(batch_size=2):
    X, meta = preprocess_batch(batch)
    all_X.append(X)
    all_meta.extend(meta)

# combine batches
import numpy as np
X_full = np.vstack(all_X)

# train model
model = train_clustering_model(X_full)

# predict
results = predict_clusters(model, X_full, all_meta)

print("\nCLUSTER RESULTS:")
for r in results:
    print(r)


# after prediction
cluster_labels = label_clusters(model)

final_results = attach_cluster_labels(results, cluster_labels)

print("\nFINAL LABELED RESULTS:")
for r in final_results:
    print(r)