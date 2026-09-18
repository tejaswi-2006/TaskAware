import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from modules.schema_analyzer import analyze_schema
from modules.preprocessing import build_preprocessing_pipeline
from modules.cluster_profiler import (
    build_cluster_profiles,
    find_distinguishing_features,
    representative_text_terms,
)


# ============================================================
# K CANDIDATE EVALUATION
# ============================================================

def evaluate_k_values(
    X_transformed,
    k_range,
    n_init=10,
    random_state=42
):
    """
    Evaluate multiple K values for K-Means.

    Returns per-K metrics: silhouette score, Davies-Bouldin
    index and Calinski-Harabasz index.
    """

    evaluations = []

    n_samples = X_transformed.shape[0]

    max_k = min(
        max(k_range) + 1,
        n_samples - 1
    )

    for k in k_range:

        if k >= n_samples:
            continue

        try:

            model = KMeans(
                n_clusters=k,
                n_init=n_init,
                random_state=random_state
            )

            labels = model.fit_predict(
                X_transformed
            )

            unique_labels = np.unique(labels)

            if len(unique_labels) < 2:
                continue

            silhouette = silhouette_score(
                X_transformed,
                labels
            )

            db_index = davies_bouldin_score(
                X_transformed,
                labels
            )

            ch_index = calinski_harabasz_score(
                X_transformed,
                labels
            )

            evaluations.append({
                "k": k,
                "silhouette": round(
                    float(silhouette),
                    4
                ),
                "davies_bouldin": round(
                    float(db_index),
                    4
                ),
                "calinski_harabasz": round(
                    float(ch_index),
                    2
                ),
                "cluster_sizes": [
                    int((labels == i).sum())
                    for i in range(k)
                ]
            })

        except Exception:
            continue

    return evaluations


# ============================================================
# BEST K SELECTION
# ============================================================

def select_best_k(evaluations):
    """
    Select the most appropriate K.

    Primary: highest silhouette score.
    Tie-break: fewer clusters (simpler interpretation) is
    preferred when scores are very close.
    """

    if not evaluations:
        return None

    best = max(
        evaluations,
        key=lambda e: e["silhouette"]
    )

    return best


# ============================================================
# STABILITY CHECK
# ============================================================

def check_stability(
    X_transformed,
    n_clusters,
    seeds=None,
    n_init=10
):
    """
    Run K-Means with multiple random seeds and compare
    the resulting labelings. Higher agreement means the
    clustering is more stable.
    """

    if seeds is None:
        seeds = [1, 2, 3, 42, 99]

    labelings = []

    for seed in seeds:

        model = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=seed
        )

        labels = model.fit_predict(
            X_transformed
        )

        labelings.append(labels)

    if len(labelings) < 2:
        return {
            "stable": True,
            "agreement": 1.0,
            "seeds_tested": len(labelings)
        }

    # Pairwise agreement using cluster membership overlaps
    n = len(labelings)
    total = 0
    matched = 0

    for i in range(n):

        for j in range(i + 1, n):

            agreement = _label_agreement(
                labelings[i],
                labelings[j]
            )

            total += 1
            matched += agreement

    mean_agreement = (
        matched / total
        if total > 0
        else 1.0
    )

    return {
        "stable": mean_agreement >= 0.80,
        "agreement": round(
            float(mean_agreement),
            3
        ),
        "seeds_tested": n
    }


def _label_agreement(labels_a, labels_b):
    """
    Compute the fraction of row pairs that are grouped
    consistently across two clusterings.
    """

    n = len(labels_a)

    if n < 2:
        return 1.0

    sample = min(n, 1000)

    idx = np.random.RandomState(0).choice(
        n,
        size=sample,
        replace=False
    )

    a = labels_a[idx]
    b = labels_b[idx]

    consistent = 0
    total = 0

    for i in range(sample):

        for j in range(i + 1, sample):

            same_a = (a[i] == a[j])
            same_b = (b[i] == b[j])

            if same_a == same_b:
                consistent += 1

            total += 1

    return (
        consistent / total
        if total > 0
        else 1.0
    )


# ============================================================
# VISUALIZATION
# ============================================================

def generate_pca_chart(
    X_transformed,
    labels,
    n_clusters,
    output_path
):
    """
    Project features to 2D and color points by cluster label.

    Uses TruncatedSVD for sparse matrices and PCA for dense
    matrices.
    """

    if X_transformed.shape[0] < 2:
        return False

    n_features = X_transformed.shape[1]

    if n_features < 2:
        # Cannot project with fewer than 2 features
        return False

    if hasattr(X_transformed, "toarray"):
        reducer = TruncatedSVD(
            n_components=2,
            random_state=42,
        )
    else:
        reducer = PCA(
            n_components=2,
            random_state=42,
        )

    reduced = reducer.fit_transform(X_transformed)

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    colors = plt.cm.tab10(
        np.linspace(
            0,
            1,
            max(n_clusters, 1)
        )
    )

    for cluster in range(n_clusters):

        mask = (labels == cluster)

        if mask.sum() == 0:
            continue

        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            s=25,
            alpha=0.6,
            color=colors[cluster % 10],
            label=f"Cluster {cluster}"
        )

    ax.set_title(
        "Cluster Visualization (PCA Projection)"
    )

    ax.set_xlabel(
        "Principal Component 1"
    )

    ax.set_ylabel(
        "Principal Component 2"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=100
    )

    plt.close(fig)

    return True


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def perform_clustering(
    df,
    objective=None,
    charts_dir=None,
    chart_id=None,
    k_range=None,
    n_init=10,
    random_state=42
):
    """
    Full clustering workflow:
        feature prep -> K evaluation -> best K ->
        final clustering -> profiles -> stability -> chart.
    """

    if k_range is None:
        k_range = [2, 3, 4, 5, 6]

    schema = analyze_schema(
        df,
        target_column=None,
    )

    # No usable features -> clear error
    if not schema["feature_columns"]:

        return {
            "success": False,
            "message": (
                "Clustering could not be performed: the "
                "dataset contains no usable feature "
                "columns. All columns were detected as "
                "identifiers, constants, or unsupported "
                "types."
            ),
        }

    numerical_columns = schema["numerical_columns"]
    categorical_columns = schema["categorical_columns"]
    boolean_columns = schema["boolean_columns"]
    text_columns = schema["text_columns"]
    date_columns = schema["date_columns"]

    feature_columns = list(schema["feature_columns"])

    X = df[feature_columns]

    pipeline = build_preprocessing_pipeline(
        numerical_columns,
        categorical_columns,
        text_columns,
        boolean_columns=boolean_columns,
        date_columns=date_columns,
    )

    X_transformed = pipeline.fit_transform(X)

    # ------------------------------------------------
    # DIMENSIONALITY REDUCTION FOR TEXT
    # ------------------------------------------------

    representation_note = None

    if text_columns:

        n_features = X_transformed.shape[1]

        is_sparse = hasattr(X_transformed, "toarray")

        # Keep at most 200 dense components for K-Means
        if n_features > 200:

            svd = TruncatedSVD(
                n_components=min(200, X_transformed.shape[0] - 1),
                random_state=42,
            )

            X_transformed = svd.fit_transform(X_transformed)

            representation_note = (
                f"Text features were transformed with TF-IDF "
                f"({n_features} terms) and reduced with "
                f"TruncatedSVD to {X_transformed.shape[1]} "
                "components for clustering."
            )

        elif is_sparse:

            X_transformed = X_transformed.toarray()

            representation_note = (
                f"Text features were transformed with TF-IDF "
                f"({n_features} terms)."
            )

        else:

            representation_note = (
                f"Text features were transformed with TF-IDF "
                f"({n_features} terms)."
            )

    # ------------------------------------------------
    # K EVALUATION
    # ------------------------------------------------

    evaluations = evaluate_k_values(
        X_transformed,
        k_range,
        n_init=n_init,
        random_state=random_state
    )

    best = select_best_k(
        evaluations
    )

    # ------------------------------------------------
    # FINAL CLUSTERING
    # ------------------------------------------------

    if best is None:

        return {
            "success": False,
            "message": (
                "Unable to find a suitable number of "
                "clusters for this dataset."
            )
        }

    n_clusters = best["k"]

    final_model = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state
    )

    labels = final_model.fit_predict(
        X_transformed
    )

    # ------------------------------------------------
    # PROFILES + STABILITY + CHART
    # ------------------------------------------------

    representative_terms = representative_text_terms(
        X_transformed,
        labels,
        pipeline,
        n_clusters,
        text_columns=text_columns,
    )

    profiles = build_cluster_profiles(
        df,
        feature_columns,
        labels,
        n_clusters,
        text_columns=text_columns,
        representative_terms=representative_terms,
    )

    distinguishing = find_distinguishing_features(
        profiles,
        feature_columns
    )

    stability = check_stability(
        X_transformed,
        n_clusters
    )

    chart_path = None

    if charts_dir:

        os.makedirs(
            charts_dir,
            exist_ok=True
        )

        base = (
            chart_id
            if chart_id
            else "clustering"
        )

        chart_path = os.path.join(
            charts_dir,
            f"{base}_clusters.png"
        )

        generate_pca_chart(
            X_transformed,
            labels,
            n_clusters,
            chart_path
        )

    # ------------------------------------------------
    # RESULT
    # ------------------------------------------------

    return {
        "success": True,
        "k_range_tested": k_range,
        "evaluations": evaluations,
        "best": best,
        "n_clusters": n_clusters,
        "cluster_labels": labels.tolist(),
        "profiles": profiles,
        "distinguishing_features": distinguishing,
        "stability": stability,
        "chart_file": (
            os.path.basename(chart_path)
            if chart_path
            else None
        ),
        "n_features": len(feature_columns),
        "n_samples": int(X_transformed.shape[0]),
        "feature_columns": feature_columns,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "boolean_columns": boolean_columns,
        "text_columns": text_columns,
        "date_columns": date_columns,
        "id_columns": schema["id_columns"],
        "constant_columns": schema["constant_columns"],
        "unsupported_columns": schema["unsupported_columns"],
        "representation_note": representation_note,
        "objective": objective
    }