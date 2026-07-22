import numpy as np
import joblib
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
from cwite_paths import data_path, output_path, output_dir

def save_clusters(name, categories_val):
    out_path = output_path('clustering_results', f'{name}.joblib')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(categories_val, out_path)

def save_variances(name, categories_val, y_val, orig_y_val, binary_y_val):
    result = {}
    categories_val = np.array(categories_val)
    for c in np.unique(categories_val):
        cluster_mask = (categories_val == c) & (binary_y_val == 0)
        if np.sum(cluster_mask) < 2:
            continue  # Skip degenerate clusters
        var_y = np.var(y_val[cluster_mask])
        var_orig_y = np.var(orig_y_val[cluster_mask])
        result[int(c)] = {
            'var_y_val': float(var_y),
            'var_orig_y_val': float(var_orig_y)
        }
    out_path = output_path('clustering_results', f'{name}_variances.joblib')
    joblib.dump(result, out_path)

suffix = 'covariate_low_resource'
for folder in ['support50_propbin']:
    try:
        path = data_path(folder)
        X_train = joblib.load(f'{path}_data/X_train_{suffix}.joblib')
        X_val = joblib.load(f'{path}_data/X_val_{suffix}.joblib')

        orig_y_val = joblib.load(f'{path}_data/orig_y_val_{suffix}.joblib')
        y_val = joblib.load(f'{path}_data/y_val_{suffix}.joblib')
        binary_y_val = joblib.load(f'{path}_data/binary_y_val_{suffix}.joblib')

        ### --- KMeans elbow ---
        k_values = range(2, min(10 * X_train.shape[1], 100))
        inertia_scores = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_train)
            inertia_scores.append(kmeans.inertia_)
        inertia_diff2 = np.diff(np.diff(inertia_scores))
        elbow_k = k_values[np.argmin(inertia_diff2) + 1]

        kmeans = KMeans(n_clusters=elbow_k, random_state=42, n_init=10)
        kmeans.fit(X_train)
        kmeans_val = kmeans.predict(X_val)
        save_clusters(f'{folder}_kmeans_val', kmeans_val)
        save_variances(f'{folder}_kmeans_val', kmeans_val, y_val, orig_y_val, binary_y_val)

        ### --- KMeans + PCA ---
        pca = PCA(n_components=min(10, X_train.shape[1]))
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)

        best_k, best_score = None, -1
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(X_train_pca)
            score = silhouette_score(X_train_pca, labels)
            if score > best_score:
                best_k, best_score = k, score
                best_km = km
        kmeans_pca_val = best_km.predict(X_val_pca)
        save_clusters(f'{folder}_kmeans_pca_val', kmeans_pca_val)
        save_variances(f'{folder}_kmeans_pca_val', kmeans_pca_val, y_val, orig_y_val, binary_y_val)

        ### --- GMMs ---
        best_k, best_score = None, -1
        for k in range(2, 11):
            gmm = GaussianMixture(n_components=k, random_state=42)
            labels = gmm.fit_predict(X_train)
            score = silhouette_score(X_train, labels)
            if score > best_score:
                best_k, best_score = k, score
                best_gmm = gmm
        gmm_val = best_gmm.predict(X_val)
        save_clusters(f'{folder}_gmm_val', gmm_val)
        save_variances(f'{folder}_gmm_val', gmm_val, y_val, orig_y_val, binary_y_val)

        ### --- DBSCAN ---
        eps_values = [0.3, 0.5, 0.7]
        min_samples_values = [3, 5, 10]
        best_score = -1
        best_dbscan = None
        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X_train)
                if len(set(labels)) > 1 and -1 not in set(labels):
                    score = silhouette_score(X_train, labels)
                    if score > best_score:
                        best_score = score
                        best_dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X_train)
        if best_dbscan is not None:
            dbscan_val = best_dbscan.fit_predict(X_val)
            save_clusters(f'{folder}_dbscan_val', dbscan_val)
            save_variances(f'{folder}_dbscan_val', dbscan_val, y_val, orig_y_val, binary_y_val)

        ### --- Spectral Clustering ---
        best_k, best_score = None, -1
        for k in range(2, 11):
            sc = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
            labels = sc.fit_predict(X_train)
            score = silhouette_score(X_train, labels)
            if score > best_score:
                best_k, best_score = k, score
                best_sc = sc
        spectral_val = best_sc.fit_predict(X_val)
        save_clusters(f'{folder}_spectral_val', spectral_val)
        save_variances(f'{folder}_spectral_val', spectral_val, y_val, orig_y_val, binary_y_val)

    except Exception as e:
        print(f"Failed for {folder} {suffix}: {str(e)}")
        continue
