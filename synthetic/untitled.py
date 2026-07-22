import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

all_results = []



f = open('clustering_results.txt', 'w')
for COUNT in range(1000):
    suffix = f'COUNT{COUNT}'

    # Load data
    X_train = joblib.load(f'/data4/meerak/onevar_data/X_train_{suffix}.joblib').reshape(-1, 1)
    y_train = joblib.load(f'/data4/meerak/onevar_data/y_train_{suffix}.joblib')

    X_val = joblib.load(f'/data4/meerak/onevar_data/X_val_{suffix}.joblib').reshape(-1, 1)
    orig_y_val = joblib.load(f'/data4/meerak/onevar_data/orig_y_val_{suffix}.joblib')
    y_val = joblib.load(f'/data4/meerak/onevar_data/y_val_{suffix}.joblib')
    binary_y_val = joblib.load(f'/data4/meerak/onevar_data/binary_y_val_{suffix}.joblib')
    
    X_test = joblib.load(f'/data4/meerak/onevar_data/X_test_{suffix}.joblib').reshape(-1, 1)
    y_test = joblib.load(f'/data4/meerak/onevar_data/y_test_{suffix}.joblib')
    binary_y_test = joblib.load(f'/data4/meerak/onevar_data/actual_binary_y_test{suffix}.joblib')

    # --- Step 1: Use elbow method to find best k on training ---
    k_values = range(2, 50)
    inertia_scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_train)
        inertia_scores.append(kmeans.inertia_)

    inertia_diff = np.diff(inertia_scores)
    inertia_diff2 = np.diff(inertia_diff)
    elbow_k = k_values[np.argmin(inertia_diff2) + 1]  # +1 to index properly

    print(f"[Dataset {COUNT}] Selected elbow_k = {elbow_k}")

    # --- Step 2: Fit with elbow_k and apply to test ---
    kmeans = KMeans(n_clusters=elbow_k, random_state=42, n_init=10)
    categories_train = kmeans.fit_predict(X_train)
    categories_val = kmeans.predict(X_val)
    categories_test = kmeans.predict(X_test)

    # --- Step 3: Cluster-wise analysis on test ---

    diff_var = []
    min_gaps = []   
     
    for cluster_id in range(elbow_k):
        cluster_mask = categories_val == cluster_id
        if not np.any(cluster_mask):
            continue
    
        y_cluster = y_val[cluster_mask]
        orig_y_cluster = orig_y_val[cluster_mask]
        binary_cluster = binary_y_val[cluster_mask]

        
        cens_times = y_cluster[binary_cluster == 0]
        uncens_times = y_cluster[binary_cluster == 1]
        orig_cens_times = orig_y_cluster[binary_cluster == 0]

        if len(cens_times) > 0:
            diff_var.append(np.var(cens_times) - np.var(orig_cens_times))
            min_gaps.append(np.min(np.abs(cens_times - orig_cens_times)))


    f.write('%0.3f, %0.3f\n'%(np.min(diff_var), np.max(min_gaps)))
    f.flush()