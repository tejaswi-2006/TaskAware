import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from modules.preprocessing import (
    build_preprocessing_pipeline,
    identify_column_roles
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
# CLUSTER PROFILES
# ============================================================

def build_cluster_profiles(
    df,
    feature_columns,
    labels,
    n_clusters
):
    """
    Build interpretable profiles for each cluster using the
    original feature columns.

    Numerical features: mean and median.
    Categorical features: dominant category and percentage.

    IMPORTANT: profiles describe observed differences in the
    data. They are not ground-truth labels.
    """

    if df is None or len(df) == 0:
        return []

    working = df[feature_columns].copy()

    working["__cluster__"] = labels

    profiles = []

    for cluster in range(n_clusters):

        subset = working[
            working["__cluster__"] == cluster
        ]

        if len(subset) == 0:
            continue

        features = []

        for column in feature_columns:

            series = subset[column].dropna()

            if len(series) == 0:
                continue

            if pd.api.types.is_numeric_dtype(series):

                features.append({
                    "column": column,
                    "type": "numerical",
                    "mean": round(
                        float(series.mean()),
                        3
                    ),
                    "median": round(
                        float(series.median()),
                        3
                    )
                })

            else:

                counts = series.value_counts()

                dominant = counts.index[0]

                features.append({
                    "column": column,
                    "type": "categorical",
                    "dominant": str(dominant),
                    "dominant_ratio": round(
                        float(
                            counts.iloc[0] / len(series)
                        ),
                        3
                    )
                })

        profiles.append({
            "cluster": int(cluster),
            "size": int(len(subset)),
            "size_percentage": round(
                len(subset) / len(working) * 100,
                2
            ),
            "features": features
        })

    return profiles


# ============================================================
# CLUSTER DIFFERENCES
# ============================================================

def find_distinguishing_features(
    profiles,
    feature_columns,
    limit=5
):
    """
    Identify the features that differ most between
    clusters. Used for the explanation section.
    """

    if len(profiles) < 2:
        return []

    differences = []

    for column in feature_columns:

        values = []

        for profile in profiles:

            for feature in profile["features"]:

                if feature["column"] != column:
                    continue

                if feature["type"] == "numerical":
                    values.append(
                        feature["mean"]
                    )
                else:
                    values.append(
                        feature["dominant"]
                    )

                break

        if len(values) < 2:
            continue

        # Only numerical features support a spread measure
        if all(
            isinstance(v, (int, float))
            for v in values
        ):

            spread = max(values) - min(values)

            relative_spread = (
                spread / abs(np.mean(values))
                if abs(np.mean(values)) > 0
                else 0
            )

            differences.append({
                "column": column,
                "spread": round(spread, 3),
                "relative_spread": round(
                    float(relative_spread),
                    3
                ),
                "cluster_values": values
            })

    differences.sort(
        key=lambda item: item["relative_spread"],
        reverse=True
    )

    return differences[:limit]


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
    Project features to 2D with PCA and color points by
    cluster label.
    """

    if X_transformed.shape[0] < 2:
        return False

    pca = PCA(
        n_components=2,
        random_state=42
    )

    reduced = pca.fit_transform(X_transformed)

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

    roles = identify_column_roles(
        df,
        target_column=None
    )

    numerical_columns = roles["numerical_columns"]
    categorical_columns = roles["categorical_columns"]

    feature_columns = (
        numerical_columns + categorical_columns
    )

    X = df[feature_columns]

    pipeline = build_preprocessing_pipeline(
        numerical_columns,
        categorical_columns
    )

    X_transformed = pipeline.fit_transform(X)

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

    profiles = build_cluster_profiles(
        df,
        feature_columns,
        labels,
        n_clusters
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
        "id_columns": roles["id_columns"],
        "constant_columns": roles["constant_columns"],
        "objective": objective
    }